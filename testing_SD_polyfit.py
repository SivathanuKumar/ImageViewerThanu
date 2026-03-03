import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import skfuzzy as fuzz
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, disk, binary_closing
from skimage.measure import label
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import math
import os


# ==========================================
# 1. Dark Center Score & Compensation Trigger
# ==========================================
def calculate_dark_center_score(img, center_diam_mm=3.0, scan_size_mm=6.0):
    """Calculates the ratio of peripheral to central intensity."""
    h, w = img.shape
    center_y, center_x = h // 2, w // 2

    # Calculate radii in pixels
    px_per_mm = w / scan_size_mm
    r_center_px = (center_diam_mm / 2) * px_per_mm
    r_total_px = (scan_size_mm / 2) * px_per_mm  # Should be w/2

    # Create coordinate grids
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    # Define masks
    center_mask = dist_from_center <= r_center_px
    periphery_mask = (dist_from_center > r_center_px) & (dist_from_center <= r_total_px)

    # Calculate means
    mean_center = np.mean(img[center_mask])
    mean_periphery = np.mean(img[periphery_mask])

    # Avoid division by zero
    if mean_center < 1e-5: mean_center = 1e-5

    score = mean_periphery / mean_center
    return score, mean_center, mean_periphery


# ==========================================
# 2. Non-Linear (Polynomial) Background Correction
# ==========================================
def polynomial_background_correction(img, poly_degree=2, blur_sigma=50):
    """
    Corrects non-uniform background using a 2D polynomial surface fit.
    """
    h, w = img.shape
    img_float = img.astype(float)

    # A. Estimate crude background (heavy blur to remove vessels)
    # This creates a map of the low-frequency intensity trends
    bg_estimate = gaussian_filter(img_float, sigma=blur_sigma)

    # B. Prepare data for fitting
    # Create X, Y coordinate grids
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    # Flatten arrays for sklearn
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = bg_estimate.flatten()

    # Stack coordinates: [x, y] points
    coords = np.vstack((X_flat, Y_flat)).T

    # C. Polynomial Feature Generation & Fitting
    # Generate polynomial features (e.g., [1, x, y, x^2, xy, y^2] for degree 2)
    poly = PolynomialFeatures(degree=poly_degree)
    coords_poly = poly.fit_transform(coords)

    # Fit a linear model to the polynomial features
    model = LinearRegression()
    model.fit(coords_poly, Z_flat)

    # Predict the smooth surface across the image
    fitted_surface_flat = model.predict(coords_poly)
    fitted_surface = fitted_surface_flat.reshape(h, w)

    # D. Apply Correction (Normalization)
    # Avoid division by zero
    fitted_surface[fitted_surface < 1e-5] = 1e-5

    global_mean = np.mean(img_float)
    corrected_img = global_mean * (img_float / fitted_surface)

    # Clip to valid range
    corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)

    # Calculate factor map for visualization
    correction_factor = global_mean / fitted_surface

    return corrected_img, fitted_surface, correction_factor


# ==========================================
# 3. Fuzzy C-Means Logic (Standalone)
# ==========================================
def bestK(img, pr):
    """Determine best K for FCM."""
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    X = img.flatten()
    X = np.random.choice(X, 10000)
    X = X.reshape(-1, 1)
    distortions = []
    # Adjusted range: 3 to 6 clusters is typical for OCTA
    for k_temp in range(3, 7):
        kmeanModel = KMeans(n_clusters=k_temp, n_init=10, random_state=42).fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)))

    # Handle edge cases for variance calculation
    if len(distortions) < 2 or distortions[0] == distortions[-1]:
        print("BestK: Not enough variance, defaulting to K=3")
        return 3

    jk1 = np.asarray(distortions[0:-1])
    jk2 = np.asarray(distortions[1::])
    variance = jk1 - jk2

    denominator = distortions[0] - distortions[-1]
    distortion_percent = np.cumsum(variance) / denominator

    try:
        res = list(map(lambda i: i > pr, distortion_percent)).index(True)
        K = res + 3  # +3 because range starts at 3
    except (ValueError, IndexError):
        K = 3  # Fallback

    print('Best K calculated:', K)
    return K


def fuzz_CC_thresholding_standalone(img_retina, img_flow, threshold_largevessel=0.68,
                                    scansize=6, k_val=None, CCthresAdj=2.0, img_mask=None):
    """Simplified FCM thresholding."""
    # 1. Preprocessing
    if img_flow.ndim == 3: img_flow = img_flow[:, :, 0]
    if img_retina.ndim == 3: img_retina = img_retina[:, :, 0]

    target_h, target_w = img_flow.shape
    if img_retina.shape != img_flow.shape:
        img_retina = cv2.resize(img_retina, (target_w, target_h))

    # 2. Large Vessel Masking (Frangi)
    print("Generating Vessel Mask (Frangi)...")
    img_norm = img_retina.astype(float)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-5)

    sigmas = np.linspace(1.0, 4.0, 4)
    vesselness = frangi(img_norm, sigmas=sigmas, black_ridges=False)

    thresh_val = 0.12 * vesselness.max()
    binary = vesselness > thresh_val

    footprint = disk(2)
    bridged_mask = binary_closing(binary, footprint)
    retina_mask = remove_small_objects(bridged_mask, min_size=200)

    # 3. Combine Masks
    if img_mask is None:
        img_mask = np.zeros(img_flow.shape, dtype=bool)

    total_mask = np.logical_or(retina_mask, img_mask)

    # 4. Prepare Data for FCM
    CC_f = np.copy(img_flow).astype(float)

    # IMPORTANT: For FCM on corrected images, we don't set to 0.
    # We extract only the valid pixels for clustering.
    valid_pixels = CC_f[~total_mask]

    if len(valid_pixels) == 0:
        print("Error: No valid pixels for FCM.")
        return np.zeros_like(CC_f, dtype=bool), total_mask

    X = valid_pixels.reshape(-1, 1)

    # 5. Fuzzy C-Means
    if k_val is None:
        k_val = bestK(CC_f, 0.96)

    print(f"Running FCM with K={k_val}...")
    # Fuzz cmeans might fail if X is too homogeneous
    try:
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X.T, k_val, 2, error=0.005, maxiter=1000, seed=42)
    except Exception as e:
        print(f"FCM Failed: {e}. Defaulting to simple threshold.")
        # Fallback: simple percentile threshold
        CC_thres = np.percentile(X, 10)
        k_val = -1  # Indicator

    if k_val != -1:
        FCM_labels = np.argmax(u, axis=0)
        min_ind = np.argmin(cntr)
        CC_group = X[FCM_labels == min_ind]
        CC_thres = CC_group.max() * CCthresAdj

    print(f"FCM Threshold: {CC_thres:.2f}")

    # 6. Generate Binary Map
    binary_ccfd = np.zeros_like(CC_f, dtype=bool)
    # Apply threshold to the whole image
    binary_ccfd[CC_f <= CC_thres] = True

    # Re-apply mask to remove false positives in vessels/GA
    binary_ccfd[total_mask] = False

    # Small object removal
    pixelsize = scansize * 1000 / target_h
    thres_size = round(((24 / 2) ** 2 * math.pi) / pixelsize / pixelsize)
    final_ccfd_mask = remove_small_objects(binary_ccfd, min_size=thres_size)

    return final_ccfd_mask, total_mask, CC_thres


# ==========================================
# 4. Main Workflow & Visualization
# ==========================================
def run_ccfd_pipeline(struct_path, flow_path, dcs_threshold=1.15, poly_degree=2):
    # A. Load Images
    struct_img = np.array(Image.open(struct_path).convert('L'))
    flow_img_orig = np.array(Image.open(flow_path).convert('L'))  # Already Zhang-compensated

    # B. Calculate Dark Center Score
    dcs, mean_c, mean_p = calculate_dark_center_score(flow_img_orig)
    print(f"Dark Center Score: {dcs:.3f} (Center: {mean_c:.1f}, Periphery: {mean_p:.1f})")

    # C. Conditional Correction
    if dcs > dcs_threshold:
        print(f"DCS > {dcs_threshold}. Applying Polynomial Background Correction (Degree {poly_degree})...")
        # Use a large blur sigma to capture only the broad trend
        flow_img_final, fitted_surface, factor_map = polynomial_background_correction(
            flow_img_orig, poly_degree=poly_degree, blur_sigma=80
        )
        corrected_status = "Corrected (Poly Fit)"
    else:
        print(f"DCS <= {dcs_threshold}. Proceeding with original image.")
        flow_img_final = flow_img_orig
        # Create dummy maps for plotting
        fitted_surface = np.full_like(flow_img_orig, np.mean(flow_img_orig), dtype=float)
        factor_map = np.ones_like(flow_img_orig, dtype=float)
        corrected_status = "Original (No Extra Correction)"

    # D. Run FCM Detection on Final Image
    # Pass the original structure image for vessel masking
    ccfd_mask, vessel_mask, final_thres = fuzz_CC_thresholding_standalone(
        img_retina=struct_img,
        img_flow=flow_img_final
    )

    # E. Visualization
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    # 1. Original Flow (Input)
    ax[0, 0].imshow(flow_img_orig, cmap='gray', vmin=0, vmax=255)
    ax[0, 0].set_title(f"1. Input Flow (Zhang Comp.)\nDCS: {dcs:.3f}")

    # 2. Fitted Background Surface
    im2 = ax[0, 1].imshow(fitted_surface, cmap='jet')
    ax[0, 1].set_title(f"2. Fitted Background Surface\n(Poly Degree {poly_degree})")
    plt.colorbar(im2, ax=ax[0, 1], fraction=0.046)

    # 3. Final Corrected Flow
    ax[0, 2].imshow(flow_img_final, cmap='gray', vmin=0, vmax=255)
    ax[0, 2].set_title(f"3. Final Flow for FCM\nStatus: {corrected_status}")

    # 4. Intensity Histograms
    ax[1, 0].hist(flow_img_orig.flatten(), bins=50, alpha=0.5, label='Original', color='b', density=True)
    if dcs > dcs_threshold:
        ax[1, 0].hist(flow_img_final.flatten(), bins=50, alpha=0.5, label='Corrected', color='g', density=True)
    ax[1, 0].axvline(final_thres, color='r', linestyle='--', label=f'FCM Thres ({final_thres:.1f})')
    ax[1, 0].set_title("4. Intensity Histograms")
    ax[1, 0].legend()
    ax[1, 0].set_xlim(0, 255)

    # 5. Final Detection Map
    ax[1, 1].imshow(ccfd_mask, cmap='gray')
    ax[1, 1].set_title("5. Final CCFD Mask (Binary)")

    # 6. Composite Overlay
    overlay = np.dstack((flow_img_final, flow_img_final, flow_img_final)).astype(float)
    red_mask = np.zeros_like(overlay);
    red_mask[:, :, 0] = 255
    ylo_mask = np.zeros_like(overlay);
    ylo_mask[:, :, 0] = 255;
    ylo_mask[:, :, 1] = 255

    overlay[ccfd_mask > 0] = overlay[ccfd_mask > 0] * 0.4 + red_mask[ccfd_mask > 0] * 0.6
    overlay[vessel_mask > 0] = overlay[vessel_mask > 0] * 0.6 + ylo_mask[vessel_mask > 0] * 0.4

    ax[1, 2].imshow(overlay.astype(np.uint8))
    ax[1, 2].set_title("6. Final Overlay\n(Red=CCFD, Yellow=Vessel)")

    plt.suptitle(f"CCFD Pipeline Results\n{os.path.basename(flow_path)}", fontsize=16)
    plt.show()


# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    # UPDATE PATHS to your actual images
    # Note: Use the image that has already gone through Zhang compensation as input.
    path_struct = '/mnt/Data3/Eye/Yi/OAC_Newmethod/Paired/SD/P4063_Angiography 6x6 mm_7-23-2021_2-30-12_OD_sn26696_raw/P4063_Angiography 6x6 mm_7-23-2021_2-30-12_OD_sn26696_ZeissFlow/Results_2026-02-03_12-38-35/Proj_0_Stru_max_L6-6.png'
    path_flow = '/mnt/Data3/Eye/Yi/OAC_Newmethod/Paired/SD/P4063_Angiography 6x6 mm_7-23-2021_2-30-12_OD_sn26696_raw/P4063_Angiography 6x6 mm_7-23-2021_2-30-12_OD_sn26696_ZeissFlow/Results_2026-02-03_12-38-35/Compensated_Flow_Raw.png'

    if os.path.exists(path_struct) and os.path.exists(path_flow):
        # Set threshold: e.g., 1.15 means trigger if periphery is >15% brighter than center
        # Set poly_degree: 2 is usually sufficient for a simple "bowl" shape.
        run_ccfd_pipeline(path_struct, path_flow, dcs_threshold=1.15, poly_degree=2)
    else:
        print("Image files not found. Please check paths.")