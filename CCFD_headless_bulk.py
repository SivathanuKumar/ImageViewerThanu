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

# --- IMPORT HELPERS ---
try:
    from Analysis.CCFD_Standalone.ccquan import cc_comp, fuzz_CC_thresholding
except ImportError:
    print("WARNING: Could not import 'cc_comp' or 'fuzz_CC_thresholding'.")
    sys.exit(1)

# --- CONSTANTS ---
SCAN_SIZE = 6  # mm
TARGET_SHAPE = (500, 500)
TAIL_PARA = 0.001
COMP_PARA = 1.5
BEST_K = 6
CC_THRES = 1.1
DCS_THRESHOLD = 1.15
POLY_DEGREE = 16
BLUR_SIGMA = 16


class CCFD_Comparator:
    def __init__(self, output_dir="CCFD_Multi_Output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        self.log_file = os.path.join(output_dir, "process_log.txt")
        self.processed_cases_data = []  # Stores image columns

    def log(self, text):
        print(f"[CCFD] {text}")
        with open(self.log_file, "a") as f: f.write(f"{text}\n")

    def load_img(self, path):
        if not os.path.exists(path): return None
        img = imread(path)
        if img is None: return None
        if img.ndim == 3: img = img[:, :, 0]
        if img.shape != TARGET_SHAPE:
            img = cv2.resize(img, (TARGET_SHAPE[1], TARGET_SHAPE[0]))
        return img

    def draw_overlay(self, img_gray, halo_val=None, color=(0, 255, 255)):
        """Draws 3mm yellow circle and optional text."""
        if img_gray.ndim == 2:
            img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img_gray.copy()

        h, w = img_bgr.shape[:2]
        center = (w // 2, h // 2)
        radius = int((3.0 / 6.0) * w / 2)  # 3mm radius in pixels

        cv2.circle(img_bgr, center, radius, color, 2)

        if halo_val is not None:
            text = f"Halo: {halo_val:.2f}"
            # Draw black outline for text
            cv2.putText(img_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            # Draw yellow text
            cv2.putText(img_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return img_bgr

    def calculate_dcs(self, img):
        h, w = img.shape
        cy, cx = h // 2, w // 2
        px_mm = w / SCAN_SIZE
        r3 = (3.0 / 2) * px_mm
        r6 = (6.0 / 2) * px_mm

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        mask_c = dist <= r3
        mask_p = (dist > r3) & (dist <= r6)

        mean_c = np.mean(img[mask_c])
        mean_p = np.mean(img[mask_p])
        return mean_p / (mean_c + 1e-6)

    def polynomial_correction(self, img):
        # 1. Blur
        img_f = img.astype(float)
        bg = gaussian_filter(img_f, sigma=BLUR_SIGMA)

        # 2. Fit Poly
        h, w = img.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))

        # Center coords
        Xc = (X - w / 2) / (w / 2)
        Yc = (Y - h / 2) / (h / 2)

        coords = np.vstack((Xc.flatten(), Yc.flatten())).T
        poly = PolynomialFeatures(degree=POLY_DEGREE)
        feats = poly.fit_transform(coords)

        reg = LinearRegression()
        reg.fit(feats, bg.flatten())

        # 3. Predict & Correct
        surf = reg.predict(feats).reshape(h, w)
        surf = np.maximum(surf, 1.0)

        corr = img_f * (np.mean(img_f) / surf)
        return np.clip(corr, 0, 255).astype(np.uint8)

    def find_gamma(self, struct, retina, mask):
        best_g = 1.0
        best_std = 999999
        for g in np.arange(0.5, 3.5, 0.5):
            res, _ = cc_comp(struct, struct, retina, para1=COMP_PARA, para2=g, para3=TAIL_PARA, exclusion_mask=mask)
            # Check flatness
            blur = gaussian_filter(res.astype(float) * 255, 15)
            vals = blur[~mask] if mask is not None else blur.flatten()
            if len(vals) > 0:
                s = np.std(vals)
                if s < best_std:
                    best_std = s
                    best_g = g
        return best_g

    def run_case(self, case_path):
        case_id = os.path.basename(case_path)
        self.log(f"Processing {case_id}...")

        # 1. Load
        f_flow = self.load_img(os.path.join(case_path, "OCTACC.png"))
        f_stru = self.load_img(os.path.join(case_path, "OCTCC.png"))
        f_ret = self.load_img(os.path.join(case_path, "Retina.png"))

        if f_flow is None:
            self.log(f"Skipping {case_id} (Missing images)")
            return

        # 2. Vessel Mask
        v_norm = f_ret.astype(float) / 255.0
        vness = frangi(v_norm, sigmas=np.linspace(1, 4, 4), black_ridges=False)
        vmask = vness > (0.12 * vness.max())
        vmask = remove_small_objects(binary_closing(vmask, disk(2)), 200)

        # 3. Method 1 (Zhang)
        gamma = self.find_gamma(f_stru, f_ret, vmask)
        m1_img, _ = cc_comp(f_flow, f_stru, f_ret, para1=COMP_PARA, para2=gamma, para3=TAIL_PARA, exclusion_mask=vmask)
        m1_u8 = np.uint8(np.clip(m1_img * 255, 0, 255))

        # 4. Halo Score
        dcs = self.calculate_dcs(m1_u8)

        # 5. Method 2 (Double Comp)
        m2_u8 = self.polynomial_correction(m1_u8)

        # 6. FCM Quantification
        def get_ccfd(img_in, tag):
            # Pass to CCQuan
            # It now returns: CCMask1 (Binary), img_color (Overlay), Size, Den, K, ...
            mask, ov, sz, den, k, ori, dens, sizes, tot = fuzz_CC_thresholding(
                f_ret, img_in.astype(float) / 255.0, 0.68,
                scansize=SCAN_SIZE, CCthresAdj=CC_THRES, k_val=BEST_K,
                img_mask=None, scaleX=1.0, scaleY=1.0,
                save_path=os.path.join(self.output_dir, f"{case_id}_{tag}")
            )
            return {'map': mask, 'dens': dens, 'sizes': sizes}  # dens/sizes are lists [1,3,5mm]

        res_m1 = get_ccfd(m1_u8, "M1")
        res_m2 = get_ccfd(m2_u8, "M2")

        # 7. Save Stats TXT
        with open(os.path.join(self.output_dir, f"{case_id}_Stats.txt"), "w") as f:
            f.write(f"Case: {case_id}\n")
            f.write(f"Halo Ratio: {dcs:.4f}\n")
            f.write("Method 1 (Zhang):\n")
            f.write(f"  CCFDD (1,3,5): {res_m1['dens']}\n")
            f.write(f"  Area  (1,3,5): {res_m1['sizes']}\n")
            f.write("Method 2 (Poly):\n")
            f.write(f"  CCFDD (1,3,5): {res_m2['dens']}\n")
            f.write(f"  Area  (1,3,5): {res_m2['sizes']}\n")

        # 8. BUILD IMAGE COLUMN (With white dividers)
        # Create Divider
        div_h = 10  # Height of horizontal divider
        div_w = f_flow.shape[1]
        divider = np.ones((div_h, div_w, 3), dtype=np.uint8) * 255

        # Row 1: Raw Flow + Circle
        r1 = self.draw_overlay(f_flow)

        # Row 2: Zhang Comp (Plain)
        r2 = cv2.cvtColor(m1_u8, cv2.COLOR_GRAY2BGR)

        # Row 3: Double Comp + Circle + Halo Text
        r3 = self.draw_overlay(m2_u8, halo_val=dcs)

        # Row 4: CCFD Map M1 (Full Map)
        # res_m1['map'] is boolean/binary. Convert to Red on Black or White on Black
        # Let's do White on Black for clarity
        r4 = np.uint8(res_m1['map']) * 255
        r4 = cv2.cvtColor(r4, cv2.COLOR_GRAY2BGR)

        # Row 5: CCFD Map M2 (Full Map)
        r5 = np.uint8(res_m2['map']) * 255
        r5 = cv2.cvtColor(r5, cv2.COLOR_GRAY2BGR)

        # Stack vertically with dividers
        col = np.vstack([r1, divider, r2, divider, r3, divider, r4, divider, r5])

        # Add Header (Case ID)
        head_h = 60
        header = np.ones((head_h, col.shape[1], 3), dtype=np.uint8) * 0  # Black background
        cv2.putText(header, case_id, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        final_col = np.vstack([header, col])
        self.processed_cases_data.append(final_col)

    def finalize_montage(self):
        if not self.processed_cases_data: return
        self.log("Building Final Montage...")

        # Vertical divider (between columns)
        h = self.processed_cases_data[0].shape[0]
        v_div_w = 20
        v_divider = np.ones((h, v_div_w, 3), dtype=np.uint8) * 255  # White vertical strip

        # Combine all columns with dividers
        final_img = self.processed_cases_data[0]
        for i in range(1, len(self.processed_cases_data)):
            final_img = np.hstack([final_img, v_divider, self.processed_cases_data[i]])

        imwrite(os.path.join(self.output_dir, "FINAL_Comparison_Montage.png"), final_img)
        self.log("Done.")


# --- RUNNER ---
if __name__ == "__main__":
    PARENT_DIR = "/home/skthanu/Documents/Dancing_Lion/ARVOIITE2026"

    comp = CCFD_Comparator(output_dir=os.path.join(PARENT_DIR, "Comparison_Results"))

    if os.path.exists(PARENT_DIR):
        folders = sorted(
            [f for f in os.listdir(PARENT_DIR) if f.startswith('1') and os.path.isdir(os.path.join(PARENT_DIR, f))])
        for f in folders:
            comp.run_case(os.path.join(PARENT_DIR, f))
        comp.finalize_montage()
    else:
        print("Path not found.")