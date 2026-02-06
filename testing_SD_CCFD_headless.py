import os
import sys
import numpy as np
import pandas as pd
import cv2
from cv2 import imread, imwrite
from skimage.transform import resize
from skimage.filters import frangi
from skimage.morphology import disk, binary_closing, remove_small_objects
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import traceback

# =========================================================
# 1. HELPER FUNCTIONS (Imports from your local package)
# =========================================================
try:
    from Analysis.CCFD_Standalone.ccquan import cc_comp, fuzz_CC_thresholding
except ImportError:
    print("WARNING: Could not import 'cc_comp' or 'fuzz_CC_thresholding'.")


    # Mock functions to prevent immediate crash if imports fail
    def cc_comp(*args, **kwargs):
        return args[0], args[0]


    def fuzz_CC_thresholding(*args, **kwargs):
        return np.zeros_like(args[1]), None, 0, 0, 6, None, [0, 0, 0], [0, 0, 0], None


# =========================================================
# 2. LOGIC: Non-Linear Background Correction (Updated)
# =========================================================
def calculate_dark_center_score(img, center_diam_mm=3.0, scan_size_mm=6.0):
    """Calculates DCS to trigger correction."""
    h, w = img.shape
    center_y, center_x = h // 2, w // 2
    px_per_mm = w / scan_size_mm
    r_center_px = (center_diam_mm / 2) * px_per_mm
    r_total_px = (scan_size_mm / 2) * px_per_mm

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    center_mask = dist <= r_center_px
    periphery_mask = (dist > r_center_px) & (dist <= r_total_px)

    mean_c = np.mean(img[center_mask])
    mean_p = np.mean(img[periphery_mask])

    if mean_c < 1e-5: mean_c = 1e-5
    score = mean_p / mean_c
    return score


def polynomial_background_correction(img, poly_degree=16, blur_sigma=16):
    """
    Fits a 4th-degree polynomial to flatten non-linear background.
    - Higher degree (4) captures complex roll-off better than linear/quadratic.
    - Lower sigma (40) keeps the correction 'tighter' to the center.
    """
    h, w = img.shape
    img_f = img.astype(float)

    # 1. Estimate Background
    # Using a smaller sigma (40) to keep the "dark center" estimate focused
    bg_estimate = gaussian_filter(img_f, sigma=blur_sigma)

    # 2. Coordinate Grid
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), bg_estimate.flatten()

    # Center coordinates to stabilize higher-order fitting
    X_cent = (X_flat - w / 2) / (w / 2)
    Y_cent = (Y_flat - h / 2) / (h / 2)
    coords = np.vstack((X_cent, Y_cent)).T

    # 3. Fit Polynomial Surface (Degree 4 for robust non-linear fit)
    poly = PolynomialFeatures(degree=poly_degree)
    coords_poly = poly.fit_transform(coords)

    # Fit model to the estimated background
    reg = LinearRegression()
    reg.fit(coords_poly, Z_flat)

    # 4. Predict Correction Surface
    fitted_flat = reg.predict(coords_poly)
    fitted_surface = fitted_flat.reshape(h, w)

    # Safety floor to prevent division by zero/negative
    fitted_surface = np.maximum(fitted_surface, 1.0)

    # 5. Apply Correction
    # This acts like a 'flat field' correction
    global_mean = np.mean(img_f)
    correction_map = global_mean / fitted_surface
    corrected_img = img_f * correction_map

    return np.clip(corrected_img, 0, 255).astype(np.uint8), fitted_surface, correction_map


# =========================================================
# 3. THE COMPARATOR CLASS
# =========================================================
class CCFD_Comparator:
    def __init__(self, output_dir="CCFD_Comparison_Results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # --- CONSTANTS ---
        self.TailPara = 0.001
        self.CompPara = 1.5
        self.BestK = 6  # Fixed K=6
        self.CCthres = 1.1  # Fixed Threshold=1.1

        self.CompPara2 = 1.0  # Auto-optimized
        self.ScanSize = 6
        self.target_shape = (500, 500)

        # --- CORRECTION TRIGGERS ---
        self.DCS_THRESHOLD = 1.15
        self.POLY_DEGREE = 4  # Increased to 4 for non-linear robustness
        self.BLUR_SIGMA = 40  # Reduced to 40 for tighter focus

        self.log_file = os.path.join(output_dir, "process_log.txt")

    def log(self, text):
        print(f"[CCFD] {text}")
        with open(self.log_file, "a") as f:
            f.write(f"{text}\n")

    def load_and_resize(self, path):
        if not os.path.exists(path):
            self.log(f"ERROR: File not found {path}")
            return None
        img = imread(path)
        if img.ndim == 3: img = img[:, :, 0]
        if img.shape != self.target_shape:
            img = cv2.resize(img, (self.target_shape[1], self.target_shape[0]))
        return img

    def find_optimal_gamma(self, oct_cc, oct_retina, exclusion_mask):
        """Standard Zhang optimization loop."""
        self.log("Optimizing Gamma (Method 1)...")
        best_std = float('inf')
        best_gamma = 1.0

        # Quick search
        for gamma in np.arange(0.5, 3.5, 0.5):
            img_comp, _ = cc_comp(oct_cc, oct_cc, oct_retina,
                                  para1=self.CompPara, para2=gamma, para3=self.TailPara,
                                  exclusion_mask=exclusion_mask, hyperTD_mask=None)

            img_blur = gaussian_filter(img_comp.astype(float) * 255, sigma=15)
            if exclusion_mask is not None:
                valid_pixels = img_blur[~exclusion_mask]
            else:
                valid_pixels = img_blur.flatten()

            if len(valid_pixels) > 0:
                curr_std = np.std(valid_pixels)
                if curr_std < best_std:
                    best_std = curr_std
                    best_gamma = gamma

        self.log(f"Optimal Gamma: {best_gamma}")
        return best_gamma

    def run_pipeline(self, path_flow, path_retina, path_struct, patient_id="Case_01"):
        self.log(f"--- Processing {patient_id} ---")

        # 1. Load
        img_flow = self.load_and_resize(path_flow)
        img_retina = self.load_and_resize(path_retina)
        img_struct = self.load_and_resize(path_struct)

        if img_flow is None: return

        # 2. Vessel Mask
        img_norm = img_retina.astype(float)
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())
        vesselness = frangi(img_norm, sigmas=np.linspace(1, 4, 4), black_ridges=False)
        vessel_mask = vesselness > (0.12 * vesselness.max())
        vessel_mask = remove_small_objects(binary_closing(vessel_mask, disk(2)), 200)

        # 3. METHOD 1 (Zhang)
        best_gamma = self.find_optimal_gamma(img_struct, img_retina, vessel_mask)
        comp_m1, _ = cc_comp(img_flow, img_struct, img_retina,
                             para1=self.CompPara, para2=best_gamma, para3=self.TailPara,
                             exclusion_mask=vessel_mask, hyperTD_mask=None)
        img_m1_uint8 = np.uint8(np.clip(comp_m1 * 255, 0, 255))

        # 4. METHOD 2 CHECK
        dcs = calculate_dark_center_score(img_m1_uint8, 3.0, self.ScanSize)
        self.log(f"DCS: {dcs:.3f}")

        img_m2_uint8 = None
        poly_surface_map = None
        correction_factor_map = None

        if dcs > self.DCS_THRESHOLD:
            self.log(f"TRIGGERED (>{self.DCS_THRESHOLD}). Applying Poly Correction (Deg {self.POLY_DEGREE})...")
            img_m2_uint8, poly_surface_map, correction_factor_map = polynomial_background_correction(
                img_m1_uint8, poly_degree=self.POLY_DEGREE, blur_sigma=self.BLUR_SIGMA
            )
        else:
            self.log("Skipping Method 2.")
            img_m2_uint8 = img_m1_uint8.copy()

        # 5. RUN FCM QUANTIFICATION
        def run_fcm(image, tag):
            self.log(f"Running FCM on {tag}...")
            img_float = image.astype(float) / 255.0
            try:
                # Calls your internal function
                ccfd_img, _, size, den, k, _, vols, sizes, _ = fuzz_CC_thresholding(
                    img_retina, img_float, 0.68, scansize=self.ScanSize,
                    CCthresAdj=self.CCthres, k_val=self.BestK,
                    img_mask=None, scaleX=1.0, scaleY=1.0,
                    save_path=os.path.join(self.output_dir, f"{patient_id}_{tag}_FCM")
                )
                return {'CCFDD': den, 'CCFDSize': size, 'Img': ccfd_img}
            except Exception as e:
                self.log(f"FCM Failed {tag}: {e}")
                traceback.print_exc()
                return None

        res_m1 = run_fcm(img_m1_uint8, "1_ZhangOnly")
        res_m2 = run_fcm(img_m2_uint8, "2_ZhangPoly")

        # 6. SAVE DETAILED VISUALIZATIONS
        self.save_visualization_suite(patient_id, img_m1_uint8, img_m2_uint8,
                                      poly_surface_map, correction_factor_map,
                                      res_m1, res_m2)

    def save_visualization_suite(self, pid, img1, img2, surface, factor_map, res1, res2):
        # A. Main Output Images
        imwrite(os.path.join(self.output_dir, f"{pid}_A_Method1_Flow.png"), img1)
        imwrite(os.path.join(self.output_dir, f"{pid}_B_Method2_Flow.png"), img2)

        # B. Compensation Maps (The requested "Entire Maps")
        if surface is not None:
            # 1. The Fitted Surface (The "Shape" of the darkness)
            norm_surf = (surface - surface.min()) / (surface.max() - surface.min()) * 255
            imwrite(os.path.join(self.output_dir, f"{pid}_C_Poly_Fit_Surface.png"), np.uint8(norm_surf))

            # 2. The Correction Factor Map (What we multiplied by)
            # Normalize for visibility (Yellow = High Boost, Blue = Low Boost)
            norm_factor = (factor_map - factor_map.min()) / (factor_map.max() - factor_map.min()) * 255
            # Save as grayscale (can apply heatmap in viewer)
            imwrite(os.path.join(self.output_dir, f"{pid}_D_Correction_Factor_Map.png"), np.uint8(norm_factor))

            # 3. Difference Map (Method 2 - Method 1)
            # Shows exactly where brightness was added
            diff = img2.astype(float) - img1.astype(float)
            diff_norm = np.clip(diff * 5, 0, 255)  # Amplify diff x5 for visibility
            imwrite(os.path.join(self.output_dir, f"{pid}_E_Difference_Map_Amplified.png"), np.uint8(diff_norm))

        # C. CCFD Binary Maps
        if res1: imwrite(os.path.join(self.output_dir, f"{pid}_F_Method1_CCFD_Map.png"), np.uint8(res1['Img']) * 255)
        if res2: imwrite(os.path.join(self.output_dir, f"{pid}_G_Method2_CCFD_Map.png"), np.uint8(res2['Img']) * 255)

        # D. Save Summary CSV
        data = {
            'PatientID': pid,
            'M1_CCFDD': res1['CCFDD'] if res1 else 0,
            'M2_CCFDD': res2['CCFDD'] if res2 else 'N/A',
            'Diff_CCFDD': (res1['CCFDD'] - res2['CCFDD']) if (res1 and res2) else 0
        }
        df = pd.DataFrame([data])
        csv_path = os.path.join(self.output_dir, "Summary.csv")
        df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        self.log(f"Saved suite for {pid}")


# =========================================================
# 4. EXECUTION
# =========================================================
if __name__ == "__main__":
    # --- UPDATE PATHS ---
    path_flow = "/home/skthanu/Documents/Dancing_Lion/CCFD compare/SD/103844/OCTACC.png"
    path_retina = "/home/skthanu/Documents/Dancing_Lion/CCFD compare/SD/103844/Retina.png"
    path_struct = "/home/skthanu/Documents/Dancing_Lion/CCFD compare/SD/103844/OCTCC.png"

    comp = CCFD_Comparator(output_dir="Run_Focused_Poly16Sig16")

    if os.path.exists(path_flow):
        comp.run_pipeline(path_flow, path_retina, path_struct, patient_id="Test_Case_01")
    else:
        print("Please check file paths.")