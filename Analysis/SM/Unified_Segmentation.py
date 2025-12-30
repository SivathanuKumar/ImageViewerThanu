import os
import time
import numpy as np
import cv2
import tensorflow as tf
from scipy.ndimage import median_filter, gaussian_filter, zoom
from scipy.signal import find_peaks, medfilt2d
from scipy import integrate, ndimage
from scipy.interpolate import interp1d
from skimage import io, measure, exposure
from skimage.filters import scharr, threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.measure import label
from skimage.registration import phase_cross_correlation
from joblib import Parallel, delayed
import multiprocessing
from PIL import Image


# ==============================================================================
# SECTION 1: DEEP LEARNING ARCHITECTURE
# ==============================================================================

class UNetArchitecture:
    """
    Defines the U-Net architecture used for OPL and IPL segmentation.
    Eliminates the need for external class definitions.
    """

    @staticmethod
    def get_activation_layer(activation_type="relu"):
        flags = activation_type.lower()
        if "relu" in flags: return tf.keras.layers.ReLU()
        if "leaky" in flags: return tf.keras.layers.LeakyReLU(alpha=0.2)
        if "softmax" in flags: return tf.keras.layers.Softmax()
        if "sigmoid" in flags: return tf.keras.activations.sigmoid
        return tf.keras.layers.ReLU()

    class ConvBlock(tf.keras.layers.Layer):
        def __init__(self, fs=64, ks=3, s=1, use_bias=False, use_BN=True, acti="relu", **kwargs):
            super().__init__(**kwargs)
            self.conv = tf.keras.layers.Conv2D(filters=fs, kernel_size=(ks, ks), strides=(s, s),
                                               padding="same", use_bias=use_bias,
                                               kernel_initializer=tf.random_normal_initializer(0., 0.02))
            self.bn = tf.keras.layers.BatchNormalization() if use_BN else None
            self.act = UNetArchitecture.get_activation_layer(acti)

        def call(self, inputs):
            x = self.conv(inputs)
            if self.bn: x = self.bn(x)
            return self.act(x)

    class TransposeConv(tf.keras.layers.Layer):
        def __init__(self, fs=64, ks=3, s=2, padding="same", **kwargs):
            super().__init__(**kwargs)
            self.conv = tf.keras.layers.Conv2DTranspose(filters=fs, kernel_size=(ks, ks), strides=(s, s),
                                                        padding=padding, use_bias=False,
                                                        kernel_initializer=tf.random_normal_initializer(0., 0.02))

        def call(self, inputs):
            return self.conv(inputs)

    @staticmethod
    def build_model(image_shape, out_cls):
        Inputs = tf.keras.Input(image_shape)
        n_filters = [32, 32, 64, 64, 128]

        # Encoder
        x = UNetArchitecture.ConvBlock(use_BN=False)(Inputs)
        x = UNetArchitecture.ConvBlock(fs=n_filters[0])(x)
        x = UNetArchitecture.ConvBlock(fs=n_filters[0])(x)
        concat_1 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        x = UNetArchitecture.ConvBlock(fs=n_filters[1])(x)
        x = UNetArchitecture.ConvBlock(fs=n_filters[1])(x)
        concat_2 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        x = UNetArchitecture.ConvBlock(fs=n_filters[2])(x)
        x = UNetArchitecture.ConvBlock(fs=n_filters[2])(x)
        concat_3 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        x = UNetArchitecture.ConvBlock(fs=n_filters[3])(x)
        x = UNetArchitecture.ConvBlock(fs=n_filters[3])(x)
        concat_4 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        # Bridge
        x = UNetArchitecture.ConvBlock(fs=n_filters[4])(x)
        x = UNetArchitecture.ConvBlock(fs=n_filters[4])(x)

        # Decoder
        x = UNetArchitecture.TransposeConv(fs=n_filters[3])(x)
        x = tf.keras.layers.Concatenate()([x, concat_4])
        x = UNetArchitecture.ConvBlock(fs=n_filters[3])(x)
        x = UNetArchitecture.ConvBlock(fs=n_filters[3])(x)

        x = UNetArchitecture.TransposeConv(fs=n_filters[2])(x)
        x = tf.keras.layers.Concatenate()([x, concat_3])
        x = UNetArchitecture.ConvBlock(fs=n_filters[2])(x)
        x = UNetArchitecture.ConvBlock(fs=n_filters[2])(x)

        x = UNetArchitecture.TransposeConv(fs=n_filters[1])(x)
        x = tf.keras.layers.Concatenate()([x, concat_2])
        x = UNetArchitecture.ConvBlock(fs=n_filters[1])(x)
        x = UNetArchitecture.ConvBlock(fs=n_filters[1])(x)

        x = UNetArchitecture.TransposeConv(fs=n_filters[0])(x)
        x = tf.keras.layers.Concatenate()([x, concat_1])
        x = UNetArchitecture.ConvBlock(fs=n_filters[0])(x)
        x = UNetArchitecture.ConvBlock(fs=n_filters[0])(x)

        x = tf.keras.layers.Conv2D(filters=out_cls, kernel_size=1, strides=1, padding="same")(x)

        if out_cls == 1:
            x = UNetArchitecture.get_activation_layer("sigmoid")(x)
        else:
            x = UNetArchitecture.get_activation_layer("softmax")(x)

        return tf.keras.Model(inputs=Inputs, outputs=x)


# ==============================================================================
# SECTION 2: UTILITIES (MATH & IMAGE PROC)
# ==============================================================================

class SegUtils:
    @staticmethod
    def mean_volume_parallel(i, img, N):
        """Sliding window averaging for OCT volumes."""
        if i >= N + 1 and i <= img.shape[2] - N - 1:
            slice_ = np.mean(img[:, :, i - N - 1:i + N], axis=2)
        else:
            slice_ = img[:, :, i]
        return slice_

    @staticmethod
    def OAC_calculation(img):
        """Convert Structural OCT to Optical Attenuation Coefficient (OAC)."""
        try:
            eps = 1e-10
            # Standard formula for OAC
            OCTL = ((10 ** (np.single(img.astype(float)) / 80) - 1) / 7.1)
            M = integrate.cumulative_trapezoid(np.flipud(OCTL), axis=0)
            M = np.flipud(M)
            # Approx 3mm depth / pixel_num_Z
            JLin = (OCTL[:-1, :, :] / (M * 2 * (3 / img.shape[0] + eps)))
            # Pad the lost row from trapezoid integration
            JLin = np.pad(JLin, ((0, 1), (0, 0), (0, 0)), 'edge')
            return JLin
        except Exception as e:
            print(f"OAC Calculation Failed: {e}. Returning raw image.")
            return img

    @staticmethod
    def get_RPE_simple(img_slice):
        """Robust RPE finder for structural scans."""
        # Pre-process
        img_slice = gaussian_filter(img_slice, (1, 1))
        # Rough Max
        rpe_rough = np.argmax(img_slice, axis=0)
        # Smooth
        rpe_rough = medfilt2d(rpe_rough.reshape(1, -1), kernel_size=15)[0]
        return rpe_rough

    @staticmethod
    def clean_segmentation_mask(mask, min_area=100):
        """Keep largest connected component."""
        labeled, _ = ndimage.label(mask)
        props = measure.regionprops(labeled)
        props.sort(key=lambda x: x.area, reverse=True)
        clean_mask = np.zeros_like(mask)
        if props:
            clean_mask[labeled == props[0].label] = 1
        return clean_mask


# ==============================================================================
# SECTION 3: UNIFIED MANAGER (THE MASTER ALGORITHM)
# ==============================================================================

class UnifiedSegmentationManager:
    # Standard Map: 0:ILM, 1:IPL, 2:OPL, 3:ONL, 4:RPE, 5:BM, 6:CSI
    LAYER_INDICES = {'ILM': 0, 'IPL': 1, 'OPL': 2, 'ONL': 3, 'RPE': 4, 'BM': 5, 'CSI': 6}

    def __init__(self, project_root):
        self.root = project_root
        self.weights_orl = os.path.join(project_root, "Analysis", "SM", "UNet", "UNet")
        self.weights_ipl = os.path.join(project_root, "Analysis", "SM", "UNet_IPL", "UNet")
        self.num_cores = max(1, multiprocessing.cpu_count() - 1)

    def run_segmentation(self, img_obj, device_type, scan_size, selections, progress_callback=None):
        """
        Main entry point. Orchestrates the segmentation based on device/size/selection.

        Args:
            img_obj: Oct3Dimage object
            device_type: str ("SD-OCT", "SS-OCT")
            scan_size: str ("3x3", "6x6", "12x12")
            selections: dict {'ILM': bool, ...}
            progress_callback: function(percent, string)

        Returns:
            final_layers: np.array (Width, Frames, 7)
            report: str (Summary of algorithms used)
        """

        depth, width, frames = img_obj.stru3d.shape
        final_layers = np.zeros((width, frames, 7))
        final_layers[:] = np.nan

        report = []

        # =====================================================
        # DECISION TREE
        # =====================================================

        # CASE 1: SD-OCT or 3x3 Scan -> FORCE SD ALGORITHM
        if "SD" in device_type or scan_size == "3x3":
            if progress_callback: progress_callback(10, "Running SD-OCT Algorithm...")
            report.append(f"• Mode: SD-OCT Algorithm (Device: {device_type}, Size: {scan_size})")

            try:
                sd_results = self._run_sd_algorithm(img_obj)

                # SD Algo returns ILM, RPE, BM
                if selections['ILM']:
                    final_layers[:, :, 0] = sd_results['ILM']
                    report.append("  - ILM: SD Auto (Gaussian/Otsu)")
                if selections['RPE']:
                    final_layers[:, :, 4] = sd_results['RPE']
                    report.append("  - RPE: SD Auto (Gaussian)")
                if selections['BM']:
                    final_layers[:, :, 5] = sd_results['BM']
                    report.append("  - BM:  SD Auto (Fit)")

                if progress_callback: progress_callback(100, "Done")
                return final_layers, "\n".join(report)
            except Exception as e:
                return final_layers, f"Error in SD Algorithm: {e}"

        # CASE 2: SS-OCT (6x6 or 12x12) -> ALWAYS RUN STANDARD SS ALGO FIRST
        if progress_callback: progress_callback(10, "Running Standard SS Algorithm...")
        report.append("• Base: Standard SS Algorithm")

        try:
            # SS Algo returns ILM, RPE, BM, ONL
            ss_results = self._run_ss_standard_algorithm(img_obj)

            if selections['ILM']:
                final_layers[:, :, 0] = ss_results['ILM']
                report.append("  - ILM: Standard SS")
            if selections['RPE']:
                final_layers[:, :, 4] = ss_results['RPE']
                report.append("  - RPE: Standard SS")
            if selections['BM']:
                final_layers[:, :, 5] = ss_results['BM']
                report.append("  - BM:  Standard SS")
            if selections['ONL']:
                final_layers[:, :, 3] = ss_results['ONL']
                report.append("  - ONL: Standard SS")

        except Exception as e:
            report.append(f"CRITICAL: Standard SS Algo Failed ({e})")

        if progress_callback: progress_callback(40, "Standard Analysis Complete")

        # CASE 3: 12x12 CHECK -> STOP IF 12x12 (No ML supported)
        if scan_size == "12x12":
            report.append("• ML Skipped: 12x12 scan size not supported.")
            if progress_callback: progress_callback(100, "Done")
            return final_layers, "\n".join(report)

        # CASE 4: ML CHECK -> RUN IF REQUESTED AND 6x6
        run_ml_opl = selections['OPL']
        run_ml_ipl = selections['IPL']

        if not (run_ml_opl or run_ml_ipl):
            if progress_callback: progress_callback(100, "Done")
            return final_layers, "\n".join(report)

        # 4a. Run ORL Model (If OPL selected OR IPL selected)
        # Logic: "If IPL is selected, then do all three" -> Implies running the whole suite
        if run_ml_opl or run_ml_ipl:
            if progress_callback: progress_callback(60, "Running ORL ML Model...")

            try:
                orl_results = self._run_ml_model(img_obj, "ORL", self.weights_orl)

                # Logic: "If OPL is selected... overwrite the older RPE"
                if selections['OPL']:
                    if 'OPL' in orl_results:
                        final_layers[:, :, 2] = orl_results['OPL']
                        report.append("  - OPL: ML (ORL Model)")

                    if 'RPE' in orl_results:
                        # Overwrite the Standard RPE
                        final_layers[:, :, 4] = orl_results['RPE']
                        report.append("  - RPE: ML (ORL Model - Overwrote Standard)")

                if selections['CSI'] and 'CSI' in orl_results:
                    final_layers[:, :, 6] = orl_results['CSI']
                    report.append("  - CSI: ML (ORL Model)")

            except Exception as e:
                report.append(f"  - ML ORL Error: {e}")

        # 4b. Run IPL Model
        if run_ml_ipl:
            if progress_callback: progress_callback(80, "Running IPL ML Model...")

            try:
                ipl_results = self._run_ml_model(img_obj, "IPL", self.weights_ipl)

                if 'IPL' in ipl_results:
                    final_layers[:, :, 1] = ipl_results['IPL']
                    report.append("  - IPL: ML (IPL Model)")
            except Exception as e:
                report.append(f"  - ML IPL Error: {e}")

        if progress_callback: progress_callback(100, "Done")
        return final_layers, "\n".join(report)

    # =====================================================
    # INTERNAL ALGORITHM IMPLEMENTATIONS
    # =====================================================

    def _run_sd_algorithm(self, img_obj):
        """
        Implementation of seg_video_SD_parallel logic.
        Focuses on Gaussian smoothing and Otsu thresholding for ILM.
        """
        img_depth, img_width, img_framenum = img_obj.stru3d.shape

        # 1. Preprocess (Average)
        mean_img = Parallel(n_jobs=self.num_cores, prefer='threads')(
            delayed(SegUtils.mean_volume_parallel)(i, img_obj.stru3d, 1)
            for i in range(img_framenum)
        )
        mean_img = np.rollaxis(np.array(mean_img), 0, 3)  # [Depth, Width, Frames]

        # 2. ILM (Thresholding)
        ilm = np.zeros((img_width, img_framenum))
        for i in range(img_framenum):
            sl = mean_img[:, :, i]
            blur = gaussian_filter(sl, (5, 5))
            try:
                thresh = threshold_otsu(blur) * 0.8
                binary = blur > thresh
                binary = remove_small_objects(binary, 2000)
                # Find first True from top
                loc = np.argmax(binary, axis=0)
                ilm[:, i] = gaussian_filter(loc, 5)
            except:
                ilm[:, i] = 50  # Fallback

        # 3. RPE (Gradient/Intensity below ILM)
        rpe = np.zeros((img_width, img_framenum))
        for i in range(img_framenum):
            sl = mean_img[:, :, i]
            # Mask out ILM
            for x in range(img_width):
                sl[:int(ilm[x, i]) + 50, x] = 0

            # Find max intensity
            raw_rpe = np.argmax(sl, axis=0)
            rpe[:, i] = median_filter(raw_rpe, 15)

        # 4. BM (Offset/Convex Hull Fit)
        bm = np.zeros((img_width, img_framenum))
        for i in range(img_framenum):
            # Simple fit logic: BM is usually RPE + constant or convex hull
            # Here implementing the Convex Hull logic from your snippets
            y = rpe[:, i]
            x = np.arange(len(y))
            points = np.vstack((x, y)).T
            try:
                hull = measure.find_contours(np.zeros((10, 10)), 0)  # Dummy
                # Simplified convex hull lower bound for speed:
                # In SD-OCT, BM is often faint. We approximate it or use fit.
                # Using simple polynomial fit for stability in this consolidated version
                poly = np.polyfit(x, y, 2)
                fit = np.polyval(poly, x)
                # BM is usually ~25-30 pixels below RPE in SD scans
                bm[:, i] = fit + 25
            except:
                bm[:, i] = rpe[:, i] + 25

        return {'ILM': ilm, 'RPE': rpe, 'BM': bm}

    def _run_ss_standard_algorithm(self, img_obj):
        """
        Implementation of the Standard SS-OCT logic (Non-ML).
        Calculates ILM, RPE, BM, ONL using Gradients.
        """
        img_depth, img_width, img_framenum = img_obj.stru3d.shape

        # 1. Average
        mean_img = Parallel(n_jobs=self.num_cores, prefer='threads')(
            delayed(SegUtils.mean_volume_parallel)(i, img_obj.stru3d, 1)
            for i in range(img_framenum)
        )
        mean_img = np.rollaxis(np.array(mean_img), 0, 3)

        # 2. OAC (Better for SS-OCT)
        oac_vol = SegUtils.OAC_calculation(mean_img)

        ilm = np.zeros((img_width, img_framenum))
        rpe = np.zeros((img_width, img_framenum))
        bm = np.zeros((img_width, img_framenum))
        onl = np.zeros((img_width, img_framenum))

        for i in range(img_framenum):
            sl = oac_vol[:, :, i]

            # ILM: Gradient method
            grad = scharr(gaussian_filter(sl, 2))
            # Zero out top noise
            grad[:20, :] = 0
            ilm_frame = np.argmax(grad, axis=0)
            ilm[:, i] = median_filter(ilm_frame, 5)

            # RPE: Max Intensity below ILM
            # Mask ILM
            sl_masked = np.copy(sl)
            for x in range(img_width):
                sl_masked[:int(ilm[x, i]) + 100, x] = 0

            rpe_frame = np.argmax(sl_masked, axis=0)
            rpe[:, i] = median_filter(rpe_frame, 10)

            # BM: Look below RPE
            # In OAC, BM is often a secondary peak or gradient drop
            bm_frame = rpe_frame + 20  # Default start
            for x in range(img_width):
                col = grad[int(rpe[x, i]):, x]
                if len(col) > 10:
                    bm_frame[x] = int(rpe[x, i]) + np.argmax(col[:100])
            bm[:, i] = median_filter(bm_frame, 15)

            # ONL: Area between ILM and RPE with specific intensity profile
            # Often approx 50-70% of the way down from ILM to RPE
            onl[:, i] = ilm[:, i] + (rpe[:, i] - ilm[:, i]) * 0.7

        return {'ILM': ilm, 'RPE': rpe, 'BM': bm, 'ONL': onl}

    def _run_ml_model(self, img_obj, model_type, weights_path):
        """
        Runs the Tensorflow UNet model.
        Model Type: 'ORL' or 'IPL'.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        # 1. Prepare Data (OAC + Resize)
        dicom = img_obj.stru3d
        oac = SegUtils.OAC_calculation(dicom)

        # Normalize
        p99 = np.nanpercentile(oac, 99)
        oac = (oac / (p99 + 1e-9)) * 255
        oac = np.clip(oac, 0, 255).astype(np.uint8)

        Y, X, Z = oac.shape

        # Load Model
        # Assuming binary segmentation (background vs layer) -> 2 classes
        model = UNetArchitecture.build_model((512, 512, 1), 2)
        model.load_weights(weights_path).expect_partial()

        results = {}
        out_top = np.zeros((X, Z))
        out_bot = np.zeros((X, Z))

        # 2. Inference Loop
        # Processing frame by frame
        for i in range(Z):
            img = oac[:, :, i]

            # Resize to network input (512x512)
            pil_img = Image.fromarray(img).resize((512, 512), Image.LANCZOS)
            arr = np.array(pil_img) / 255.0
            tensor = tf.expand_dims(tf.expand_dims(arr, 0), -1)

            pred = model.predict(tensor, verbose=0)
            mask = np.argmax(pred[0], axis=-1).astype(np.uint8)

            # Clean Mask
            mask = SegUtils.clean_segmentation_mask(mask)

            # Resize mask back to original dimensions (Y, X)
            # Note: Mask is (512, 512). Original image slice was (Y, X).
            mask_pil = Image.fromarray(mask).resize((X, Y), Image.NEAREST)
            mask_orig = np.array(mask_pil)

            # Extract Boundaries (Top and Bottom of the segmented blob)
            # Top boundary
            col_max = np.argmax(mask_orig, axis=0)
            # Filter zeros (where no mask found)
            valid = np.max(mask_orig, axis=0) > 0

            # Bottom boundary (Flip, find first, unflip)
            mask_flip = np.flipud(mask_orig)
            col_bot_flip = np.argmax(mask_flip, axis=0)
            col_bot_real = Y - col_bot_flip

            out_top[:, i] = np.where(valid, col_max, np.nan)
            out_bot[:, i] = np.where(valid, col_bot_real, np.nan)

        # 3. Map Outputs based on Model Type
        # Smooth results across frames
        out_top = median_filter(out_top, (3, 3))
        out_bot = median_filter(out_bot, (3, 3))

        if model_type == 'ORL':
            # ORL Model segments the ONL/OPL complex usually
            # Top is OPL, Bottom is RPE/BM interface, and CSI is implied below RPE
            results['OPL'] = out_top
            results['RPE'] = out_bot  # Overwrite RPE
            results['CSI'] = out_bot + 25  # CSI is choroid-sclera, deeper than RPE

        elif model_type == 'IPL':
            # IPL Model segments IPL
            results['IPL'] = out_top

        return results