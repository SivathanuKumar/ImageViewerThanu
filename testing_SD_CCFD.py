import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from PIL import Image
import cv2
import skfuzzy as fuzz
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, disk, binary_closing
from skimage.measure import label
import math
import os


# ==========================================
# 1. Compensation Logic (Refined)
# ==========================================
def compensate_flow_image(struct_path, flow_path, roi_size=50):
    """
    Performs regression-based compensation and returns debug data for plotting.
    """
    # Load Images
    struct_img = np.array(Image.open(struct_path).convert('L')).astype(np.float32)
    flow_img = np.array(Image.open(flow_path).convert('L')).astype(np.float32)

    height, width = struct_img.shape

    # Extract Reference Region (Top-Right)
    # ROI: y=[0:roi_size], x=[width-roi_size:width]
    roi_struct = struct_img[0:roi_size, width - roi_size:width]
    roi_flow = flow_img[0:roi_size, width - roi_size:width]

    # Linear Regression
    flat_struct = roi_struct.flatten()
    flat_flow = roi_flow.flatten()
    slope, intercept, r_value, _, _ = linregress(flat_struct, flat_flow)

    print(f"Regression Fit: Flow = {slope:.4f} * Struct + {intercept:.4f} (R={r_value:.4f})")

    # Predict Background & Calculate Factor
    predicted_background = slope * struct_img + intercept
    predicted_background = np.maximum(predicted_background, 1.0)  # Safety floor
    target_background = np.mean(roi_flow)

    correction_factor = target_background / predicted_background

    # Apply Compensation
    compensated_flow = flow_img * correction_factor
    compensated_flow = np.clip(compensated_flow, 0, 255).astype(np.uint8)

    # Pack debug info for visualization
    debug_info = {
        'struct_img': struct_img,
        'flow_img': flow_img,
        'roi_struct': roi_struct,
        'roi_flow': roi_flow,
        'flat_struct': flat_struct,
        'flat_flow': flat_flow,
        'slope': slope,
        'intercept': intercept,
        'roi_coords': (width - roi_size, 0),  # x, y for plotting box
        'roi_dim': roi_size
    }

    return compensated_flow, correction_factor, debug_info


# ==========================================
# 2. Fuzzy C-Means Logic (Standalone)
# ==========================================
def bestK(img, pr):
    """Determine best K for FCM."""
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    X = img.flatten()
    X = np.random.choice(X, 10000)  # Subsample for speed
    X = X.reshape(-1, 1)
    distortions = []
    for k_temp in range(2, 8):  # Reduced range for speed in demo
        kmeanModel = KMeans(n_clusters=k_temp, n_init=10).fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)))

    jk1 = np.asarray(distortions[0:-1])
    jk2 = np.asarray(distortions[1::])
    variance = jk1 - jk2

    # Handle edge case where variance is zero
    denom = (distortions[0] - distortions[-1])
    if denom == 0: denom = 1

    distortion_percent = np.cumsum(variance) / denom

    try:
        res = list(map(lambda i: i > pr, distortion_percent)).index(True)
        K = res + 2  # Adjusted index
    except ValueError:
        K = 3  # Fallback

    print('Best K calculated:', K)
    return K


def fuzz_CC_thresholding_standalone(img_retina, img_comp, threshold_largevessel=0.68,
                                    scansize=6, k_val=None, CCthresAdj=2.0, img_mask=None):
    """
    Simplified version of your FCM thresholding for standalone use.
    """
    # 1. Preprocessing
    if img_comp.ndim == 3: img_comp = img_comp[:, :, 0]
    if img_retina.ndim == 3: img_retina = img_retina[:, :, 0]

    target_h, target_w = img_comp.shape
    if img_retina.shape != img_comp.shape:
        img_retina = cv2.resize(img_retina, (target_w, target_h))

    # 2. Large Vessel Masking (Retina)
    # Using Frangi (New Method)
    print("Generating Vessel Mask (Frangi)...")
    img_norm = img_retina.astype(float)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-5)

    sigmas = np.linspace(1.0, 4.0, 4)  # Adjusted sigmas for typical OCTA
    vesselness = frangi(img_norm, sigmas=sigmas, black_ridges=False)

    thresh_val = 0.12 * vesselness.max()
    binary = vesselness > thresh_val

    footprint = disk(2)
    bridged_mask = binary_closing(binary, footprint)
    retina_mask = remove_small_objects(bridged_mask, min_size=200)

    # 3. Combine Masks
    if img_mask is None:
        img_mask = np.zeros(img_comp.shape, dtype=bool)

    total_mask = np.logical_or(retina_mask, img_mask)

    # 4. Prepare Data for FCM
    CC_f = np.copy(img_comp).astype(float)
    CC_f[total_mask] = 0  # Mask out vessels

    X = CC_f.flatten()
    X = X[X > 0]  # Only analyze flow pixels
    X = X.reshape(-1, 1)

    # 5. Fuzzy C-Means
    if k_val is None:
        k_val = bestK(CC_f, 0.96)

    print(f"Running FCM with K={k_val}...")
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X.T, k_val, 2, error=0.005, maxiter=1000, seed=42)

    # 6. Thresholding
    FCM_labels = np.argmax(u, axis=0)
    min_ind = np.argmin(cntr)

    # Get the cluster with the lowest intensity (Flow Deficits)
    # Note: In compensated images, flow deficits are DARK (low values)
    CC_group = X[FCM_labels == min_ind]
    CC_thres = CC_group.max() * CCthresAdj
    print(f"FCM Threshold: {CC_thres:.2f}")

    # Generate Binary Map
    # Pixels <= Threshold are Deficits (1), others are Flow (0)
    CC_m = np.copy(CC_f)
    binary_ccfd = np.zeros_like(CC_m, dtype=bool)
    binary_ccfd[CC_m <= CC_thres] = True
    binary_ccfd[CC_m > CC_thres] = False

    # Clean up
    binary_ccfd[total_mask] = False  # Ensure vessels are not counted as deficits

    # Small object removal
    pixelsize = scansize * 1000 / target_h
    thres_size = round(((24 / 2) ** 2 * math.pi) / pixelsize / pixelsize)
    final_ccfd_mask = remove_small_objects(binary_ccfd, thres_size)

    return final_ccfd_mask, total_mask


# ==========================================
# 3. Visualization (The 6 Plots)
# ==========================================
def visualize_analysis(struct_path, flow_path):
    # A. Run Compensation
    comp_flow, comp_factor, debug = compensate_flow_image(struct_path, flow_path)

    # B. Run Detection
    # Using the compensated flow and the ORIGINAL Structure (as retina proxy for vessel masking)
    # Note: Ideally you pass the Retina layer for vessel masking, but Structure works for demo
    ccfd_mask, vessel_mask = fuzz_CC_thresholding_standalone(
        img_retina=debug['struct_img'],
        img_comp=comp_flow
    )

    # C. Plotting
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    # 1. Original Structure with ROI
    ax[0, 0].imshow(debug['struct_img'], cmap='gray')
    ax[0, 0].set_title("1. Original Structure (Reflectance)")
    # Draw ROI Box
    roi_x, roi_y = debug['roi_coords']
    rect = plt.Rectangle((roi_x, roi_y), debug['roi_dim'], debug['roi_dim'],
                         edgecolor='red', facecolor='none', lw=2)
    ax[0, 0].add_patch(rect)
    ax[0, 0].text(roi_x - 10, roi_y + 20, "Ref ROI", color='red', fontweight='bold', ha='right')

    # 2. Original Flow with ROI
    ax[0, 1].imshow(debug['flow_img'], cmap='gray')
    ax[0, 1].set_title("2. Original Flow (Attenuated Center)")
    rect2 = plt.Rectangle((roi_x, roi_y), debug['roi_dim'], debug['roi_dim'],
                          edgecolor='red', facecolor='none', lw=2)
    ax[0, 1].add_patch(rect2)

    # 3. Regression Analysis
    ax[0, 2].scatter(debug['flat_struct'], debug['flat_flow'], alpha=0.3, s=2, label='ROI Pixels')

    # Plot Fit Line
    x_space = np.linspace(debug['flat_struct'].min(), debug['flat_struct'].max(), 100)
    y_pred = debug['slope'] * x_space + debug['intercept']
    ax[0, 2].plot(x_space, y_pred, color='red', lw=2, label='Noise Floor Fit')

    ax[0, 2].set_xlabel("Structure Intensity")
    ax[0, 2].set_ylabel("Flow Intensity")
    ax[0, 2].set_title(f"3. ROI Regression\ny = {debug['slope']:.2f}x + {debug['intercept']:.2f}")
    ax[0, 2].legend()
    ax[0, 2].grid(True, alpha=0.3)

    # 4. Compensation Factor Map
    im4 = ax[1, 0].imshow(comp_factor, cmap='jet')
    ax[1, 0].set_title("4. Compensation Factor Map")
    plt.colorbar(im4, ax=ax[1, 0], fraction=0.046, pad=0.04)

    # 5. Compensated Flow
    ax[1, 1].imshow(comp_flow, cmap='gray')
    ax[1, 1].set_title("5. Compensated Flow")

    # 6. Final CCFD Detection
    # Create an overlay
    overlay = np.dstack((comp_flow, comp_flow, comp_flow))

    # Tint Deficits Red
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = 255  # R

    # Tint Vessels/Excluded Yellow
    yellow_mask = np.zeros_like(overlay)
    yellow_mask[:, :, 0] = 255  # R
    yellow_mask[:, :, 1] = 255  # G

    # Apply tints
    # Where CCFD is True, blend with Red
    # Where Vessel is True, blend with Yellow

    # Convert to float for blending
    overlay = overlay.astype(float)

    # Overlay Deficits (Red)
    mask_indices = ccfd_mask > 0
    overlay[mask_indices] = overlay[mask_indices] * 0.5 + red_mask[mask_indices] * 0.5

    # Overlay Exclusions (Yellow)
    vessel_indices = vessel_mask > 0
    overlay[vessel_indices] = overlay[vessel_indices] * 0.7 + yellow_mask[vessel_indices] * 0.3

    ax[1, 2].imshow(overlay.astype(np.uint8))
    ax[1, 2].set_title("6. Final Detection (Red=Deficit, Yellow=Vessel)")

    plt.suptitle(f"CCFD Analysis Dashboard\n{os.path.basename(struct_path)}", fontsize=16)
    plt.show()


# ==========================================
# 4. Execution
# ==========================================
if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    path_struct = '/mnt/Data3/Eye/Yi/OAC_Newmethod/Paired/SD/P4063_Angiography 6x6 mm_7-23-2021_2-30-12_OD_sn26696_raw/P4063_Angiography 6x6 mm_7-23-2021_2-30-12_OD_sn26696_ZeissFlow/Results_2026-02-03_12-38-35/Proj_0_Stru_max_L6-6.png'
    path_flow = '/mnt/Data3/Eye/Yi/OAC_Newmethod/Paired/SD/P4063_Angiography 6x6 mm_7-23-2021_2-30-12_OD_sn26696_raw/P4063_Angiography 6x6 mm_7-23-2021_2-30-12_OD_sn26696_ZeissFlow/Results_2026-02-03_12-38-35/Compensated_Flow_Raw.png'

    # Check if files exist before running to avoid errors
    if os.path.exists(path_struct) and os.path.exists(path_flow):
        visualize_analysis(path_struct, path_flow)
    else:
        print("Please update the file paths in the '__main__' section.")
