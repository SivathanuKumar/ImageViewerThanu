import csv
import os
import sys
import weakref
import fnmatch
import traceback
import io as ioio
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
import tifffile
from PIL import ImageEnhance
from matplotlib.cm import get_cmap
from imageio import imwrite
import cv2
from cv2 import imread

# skimage imports
from skimage.filters import frangi
from skimage.morphology import disk, binary_closing, remove_small_objects
from skimage.transform import resize

# PyQt5 imports
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# Internal module imports (Assumed to be present in your environment)
#import QuanCC_Standalone
from Analysis.CCFD_Standalone.scenes import *
from Analysis.CCFD_Standalone.ccquan import *
from Analysis.CCFD_Standalone.QuanCC_Standalone import Ui_MainWindow

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)


def resize_mask_nearest(mask, target_shape):
    """
    Helper to resize masks without thickening/blurring edges.
    Uses Nearest Neighbor interpolation.
    """
    if mask is None:
        return None
    # cv2.resize expects (width, height), shape is (height, width)
    # target_shape is (H, W)
    dest_size = (target_shape[1], target_shape[0])

    # Ensure input is suitable for cv2 (uint8 preferred for masks)
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255

    resized = cv2.resize(mask, dest_size, interpolation=cv2.INTER_NEAREST)

    # Return as boolean if input was boolean-like logic
    return resized > 127


class CCquanWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """
    CC quantification window
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        self.url_ccfd = None
        self.setupUi(self)
        self.HyperTDmask = None

        self.scene_Mask_HyperTD = GraphicsScene2d()
        self.graphicsView_Mask_HyperTD.setScene(self.scene_Mask_HyperTD)

        self.scene_Mask_ExclusionCombined = GraphicsScene2d()
        self.graphicsView_Mask_ExclusionCombined.setScene(self.scene_Mask_ExclusionCombined)
        self.pushButton_LoadMask_HyperTD.clicked.connect(self.on_button_clicked_load_hyperTD)
        self.pushButton_MaskMerge.clicked.connect(self.on_button_clicked_mask_merge)

        # display
        self.scene_OCTACC = GraphicsScene2d()
        self.graphicsView_OCTACC.setScene(self.scene_OCTACC)

        self.target_shape = (500, 500)
        # self.setupUi(self)

        self.scene_OCTCC = GraphicsScene2d()
        self.graphicsView_OCTCC.setScene(self.scene_OCTCC)

        self.scene_OCTARetina = GraphicsScene2d()
        self.graphicsView_OCTARetina.setScene(self.scene_OCTARetina)

        self.scene_CCflowvoids = GraphicsScene2d()
        self.graphicsView_CCflowvoids.setScene(self.scene_CCflowvoids)

        self.scene_ImageComp = GraphicsScene2d()
        self.graphicsView_ImageComp.setScene(self.scene_ImageComp)

        self.scene_Mask = GraphicsScene2d()
        self.graphicsView_Mask.setScene(self.scene_Mask)

        # push button
        self.pushButton_LoadOCTACC.clicked.connect(self.load_img_OCTACC)
        self.pushButton_LoadOCTCC.clicked.connect(self.load_img_OCTCC)
        self.pushButton_LoadOCTARetina.clicked.connect(self.load_img_OCTARetina)

        self.pushButton_PreviewComp.clicked.connect(self.load_img_comp)
        self.pushButton_Preview.clicked.connect(self.load_img_CC)
        self.pushButton_Save.clicked.connect(self.save_results)
        self.pushButton_ClearAll.clicked.connect(self.clear_all)

        self.GAmask = None
        self.HyperTDmask = None
        self.target_shape = (500, 500)

        self.pushButton_UpdateRule.clicked.connect(
            self.on_button_clicked_open_batch)
        self.pushButton_RunSingle.clicked.connect(
            self.on_button_clicked_run_single)
        self.pushButton_RunBatch.clicked.connect(
            self.on_button_clicked_run_batch)

        self.pushButton_LoadZeissCC.clicked.connect(
            self.on_button_clicked_load_zeiss_CC)

        self.pushButton_LoadMask.clicked.connect(
            self.on_button_clicked_load_mask)

        # scale
        self.global_scale = 1.0
        # self.horizontalSlider_ScaleSlice.valueChanged.connect(self.on_button_clicked_scale_slice)
        self.pushButton_SliceZoomIn.clicked.connect(
            self.on_button_clicked_scale_slice_zoom_in)
        self.pushButton_SliceZoomOut.clicked.connect(
            self.on_button_clicked_scale_slice_zoom_out)
        self.pushButton_ZoomToFit.clicked.connect(
            self.on_button_clicked_scale_to_fit)

        # set the path to the lineEdit
        self.get_current_path()

    def get_current_path(self):
        """ Get the current path of the file, forward it to the lineEdit_OpenPath
        """
        path = os.getcwd()
        self.lineEdit_OpenPath.setText(path)

    def openFileNameDialog(self, url=None, type=None):
        """Open file dialog, return the selected fileName
        Args:
            url:
        """
        print('open file url = ', url)

        url = str(Path(url))
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        if type == 'video':
            fileName, _ = QFileDialog.getOpenFileName(None, "Open file", url,
                                                      "Video Files (*.avi *.dcm *.img)",
                                                      options=options)
        elif type == 'image':
            fileName, _ = QFileDialog.getOpenFileName(None, "Open file", url,
                                                      "Image Files (*.jpg *.bmp *.tif *.png)",
                                                      options=options)
        else:
            fileName, _ = QFileDialog.getOpenFileName(None, "Open file", url,
                                                      "All Files (*.*)",
                                                      options=options)

        if fileName:
            print(fileName)

        return fileName

    def open_folder(self, dir=None):
        '''Open the folder
        '''
        if dir is not None:
            dir = str(Path(dir))
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folder_name = QFileDialog.getExistingDirectory(self, 'Select directory',
                                                       directory=dir,
                                                       options=options)
        return folder_name

    def on_button_clicked_scale_slice_zoom_in(self):
        '''Zoom in
        '''
        scale = 1.3
        self.update_scale_of_view(scale)
        self.global_scale *= scale

    def on_button_clicked_scale_slice_zoom_out(self):
        '''Zoom out
        '''
        scale = 0.7
        self.update_scale_of_view(scale)
        self.global_scale *= scale

    def on_button_clicked_scale_to_fit(self):
        '''Scale to fit
        '''
        scale = 1.0 / self.global_scale
        self.global_scale = 1.0
        self.update_scale_of_view(scale)

    def update_scale_of_view(self, scale):
        ''' Updates all graphics view
        '''
        self.graphicsView_OCTACC.scale(scale, scale)
        self.graphicsView_OCTARetina.scale(scale, scale)
        self.graphicsView_CCflowvoids.scale(scale, scale)
        self.graphicsView_OCTCC.scale(scale, scale)
        self.graphicsView_ImageComp.scale(scale, scale)

    def update_parameters(self):
        """ Update the parameters from the UI
        """
        self.TailPara = float(self.lineEdit_TailPara.text())
        self.CompPara = float(self.lineEdit_CompPara.text())
        # self.CompPara2 = float(self.lineEdit_CompPara2.text())
        self.ScanSize = int(self.lineEdit_ScanSize.text())
        self.BestK = int(self.lineEdit_BestK.text())
        self.CCthres = float(self.lineEdit_CCthres.text())

    def load_img_OCTACC(self):
        """
        Load MAIN image (OCTA CC) - Sets the Master Size
        """
        url = self.lineEdit_OpenPath.text()
        file_url = self.openFileNameDialog(url, 'image')
        if os.path.exists(file_url):
            self.filename = file_url
            self.lineEdit_OpenPath.setText(os.path.dirname(file_url))
            img = imread(file_url)

            # --- NEW: Set Target Shape based on this image ---
            if img.ndim == 3:
                self.target_shape = img.shape[:2]
            else:
                self.target_shape = img.shape

            self.OCTACC = img
            self.scene_OCTACC.update_image(img)
            self.graphicsView_OCTACC.fitInView(self.scene_OCTACC.sceneRect(), Qt.KeepAspectRatio)
            self.send_and_display_the_log(f'Loaded OCTA CC: {os.path.basename(file_url)} | Dims: {self.target_shape}')
        else:
            self.send_and_display_the_log('File does not exist: ' + file_url)

    def load_img_OCTCC(self):
        url = self.lineEdit_OpenPath.text()
        file_url = self.openFileNameDialog(url, 'image')
        self.file_url_for_ref = file_url
        if os.path.exists(file_url):
            img = imread(file_url)
            self.lineEdit_OpenPath.setText(os.path.dirname(file_url))

            # --- NEW: Resize to match OCTA CC ---
            if img.shape[:2] != self.target_shape:
                self.send_and_display_the_log(f"Resizing OCT CC from {img.shape[:2]} to {self.target_shape}")
                # cv2.resize takes (width, height), shape is (height, width)
                img = cv2.resize(img, (self.target_shape[1], self.target_shape[0]))

            self.OCTCC = img
            self.scene_OCTCC.update_image(img)
            self.graphicsView_OCTCC.fitInView(self.scene_OCTCC.sceneRect(), Qt.KeepAspectRatio)
            self.send_and_display_the_log(f'Loaded OCT CC: {os.path.basename(file_url)} | Dims: {img.shape[:2]}')
        else:
            self.send_and_display_the_log('File does not exist: ' + file_url)

    def on_button_clicked_load_mask(self):
        url = self.lineEdit_OpenPath.text()
        file_url = self.openFileNameDialog(url, 'image')
        if os.path.exists(file_url):
            img = imread(file_url)
            self.lineEdit_OpenPath.setText(os.path.dirname(file_url))

            if img.ndim == 3:
                img = img[:, :, 0]

            # --- RESIZE: NEAREST NEIGHBOR TO PREVENT THICKENING ---
            if img.shape != self.target_shape:
                img = resize_mask_nearest(img, self.target_shape)

            self.GAmask = img > 0
            self.scene_Mask.update_image(np.uint8(self.GAmask) * 255)
            self.graphicsView_Mask.fitInView(self.scene_Mask.sceneRect(), Qt.KeepAspectRatio)
            self.send_and_display_the_log(f'Loaded Mask: {os.path.basename(file_url)} | Dims: {img.shape}')
        else:
            self.send_and_display_the_log('File does not exist: ' + file_url)

    def on_button_clicked_load_zeiss_CC(self):
        """
        Load the cc image with compasetion
        Returns:
        """
        if self.checkBox_UseMask.isChecked():

            url = self.lineEdit_OpenPath.text()
            file_url = self.openFileNameDialog(url, 'image')
            if os.path.exists(file_url):
                self.filename = file_url
                img = imread(file_url)
                self.lineEdit_OpenPath.setText(
                    os.path.dirname(file_url))  # set the url to the window
                self.img_comp = img
                self.scene_ImageComp.update_image(img)
                self.graphicsView_ImageComp.fitInView(
                    self.scene_ImageComp.sceneRect(), Qt.KeepAspectRatio)
                self.send_and_display_the_log(
                    'Load Zeiss CC Image: ' + file_url)
            else:
                self.send_and_display_the_log(
                    'File does not exist: ' + file_url)

    def on_button_clicked_load_hyperTD(self):
        url = self.lineEdit_OpenPath.text()
        file_url = self.openFileNameDialog(url, 'image')
        if os.path.exists(file_url):
            img = imread(file_url)
            self.lineEdit_OpenPath.setText(os.path.dirname(file_url))

            if img.ndim == 3:
                img = img[:, :, 0]

            # --- RESIZE: NEAREST NEIGHBOR TO PREVENT THICKENING ---
            if img.shape != self.target_shape:
                img = resize_mask_nearest(img, self.target_shape)

            self.HyperTDmask = img > 0
            self.scene_Mask_HyperTD.update_image(np.uint8(self.HyperTDmask) * 255)
            self.graphicsView_Mask_HyperTD.fitInView(self.scene_Mask_HyperTD.sceneRect(), Qt.KeepAspectRatio)
            self.send_and_display_the_log(f'Loaded HyperTD: {os.path.basename(file_url)} | Dims: {img.shape}')
        else:
            self.send_and_display_the_log('File does not exist: ' + file_url)

    def on_button_clicked_mask_merge(self):
        """
        Merges multiple user masks (HypoTD + GA) and puts result in graphicsView_Mask.
        This is CUMULATIVE: New selections are added to existing masks.
        SAVES INTERMEDIATE STEPS.
        """
        url = self.lineEdit_OpenPath.text()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        files, _ = QFileDialog.getOpenFileNames(self, "Select Exclusion Masks (GA, HypoTD)", url,
                                                "Image Files (*.png *.jpg *.bmp *.tif)", options=options)

        if files:
            # 1. Check for existing mask to preserve previous loads
            if self.GAmask is not None and self.GAmask.shape == self.target_shape:
                combined_mask = self.GAmask  # Start with what we already have
                self.send_and_display_the_log("Adding to existing mask...")
            else:
                combined_mask = np.zeros(self.target_shape, dtype=bool)  # Start blank

            # 2. Prepare Output Directory for Steps
            # We use the directory of the first selected file
            base_dir = os.path.dirname(files[0])
            step_save_dir = os.path.join(base_dir, "Merged_Mask_Steps")

            if not os.path.exists(step_save_dir):
                os.makedirs(step_save_dir)
                self.send_and_display_the_log(f"Created folder for mask steps: {step_save_dir}")

            # 3. Iterate, Merge, and Save Step-by-Step
            for i, file_path in enumerate(files):
                try:
                    img = imread(file_path)

                    # Ensure 2D
                    if img.ndim == 3:
                        img = img[:, :, 0]

                    # Resize to match Master OCTA CC
                    # Use NEAREST to prevent thickening
                    if img.shape != self.target_shape:
                        img = resize_mask_nearest(img, self.target_shape)

                    # Add to combination (Logical OR)
                    current_mask = img > 0
                    combined_mask = np.logical_or(combined_mask, current_mask)

                    filename_only = os.path.basename(file_path)
                    self.send_and_display_the_log(f'Merged: {filename_only}')

                    # --- SAVE INTERMEDIATE STEP ---
                    # Naming: step_01_added_[filename].png
                    step_filename = f"step_{i + 1:02d}_added_{filename_only}.png"
                    save_path = os.path.join(step_save_dir, step_filename)

                    # Convert boolean to uint8 (0-255) for saving
                    mask_to_save = np.uint8(combined_mask) * 255
                    imwrite(save_path, mask_to_save)
                    # ------------------------------

                except Exception as e:
                    print(f"Error merging file {file_path}: {e}")
                    self.send_and_display_the_log(f"Error merging {os.path.basename(file_path)}")

            # Store in the standard Exclusion Variable
            self.GAmask = combined_mask

            # Display in the standard Mask View (User Input)
            display_img = np.uint8(combined_mask) * 255
            self.scene_Mask.update_image(display_img)
            self.graphicsView_Mask.fitInView(self.scene_Mask.sceneRect(), Qt.KeepAspectRatio)

            self.send_and_display_the_log(f'Mask Merge Complete. Intermediate steps saved to {step_save_dir}')
            self.checkBox_UseMask.setChecked(True)

    def load_img_OCTARetina(self):
        url = self.lineEdit_OpenPath.text()
        file_url = self.openFileNameDialog(url, 'image')
        self.fileloc = file_url
        if os.path.exists(file_url):
            img = imread(file_url)
            self.lineEdit_OpenPath.setText(os.path.dirname(file_url))

            # --- NEW: Resize to match OCTA CC ---
            if img.shape[:2] != self.target_shape:
                self.send_and_display_the_log(f"Resizing OCTA Retina from {img.shape[:2]} to {self.target_shape}")
                img = cv2.resize(img, (self.target_shape[1], self.target_shape[0]))

            self.OCTARetina = img
            self.scene_OCTARetina.update_image(img)
            self.graphicsView_OCTARetina.fitInView(self.scene_OCTARetina.sceneRect(), Qt.KeepAspectRatio)
            self.send_and_display_the_log(f'Loaded Retina: {os.path.basename(file_url)} | Dims: {img.shape[:2]}')
        else:
            self.send_and_display_the_log('File does not exist: ' + file_url)

    def load_img_comp(self):
        """
        Preview the compensation with specific masking rules:
        1. Regular Mask (GA): Excluded from analysis.
        2. HyperTD: Excluded from analysis, but merged back UNCOMPENSATED at the end.
        """
        self.send_and_display_the_log('Previewing OCTA CC image compensation...')
        self.update_parameters()

        # Create output directories
        try:
            destination_rawop = os.path.dirname(self.fileloc) + "/CCFD_debug"
            if not os.path.exists(destination_rawop):
                os.makedirs(destination_rawop)

            self.url_ccfd = os.path.dirname(self.fileloc) + '/CCFD_Outputs/'
            if not os.path.exists(self.url_ccfd):
                os.makedirs(self.url_ccfd)
        except Exception as e:
            print('cannot save images: ', e)

        # Initialize User Masks (GA / HyperTD) if missing
        if self.HyperTDmask is None:
            self.HyperTDmask = np.zeros(self.target_shape, dtype=bool)
            self.scene_Mask_HyperTD.update_image(np.uint8(self.HyperTDmask) * 255)
            self.graphicsView_Mask_HyperTD.fitInView(self.scene_Mask_HyperTD.sceneRect(), Qt.KeepAspectRatio)

        if self.GAmask is None:
            self.GAmask = np.zeros(self.target_shape, dtype=bool)
            self.scene_Mask.update_image(np.uint8(self.GAmask) * 255)
            self.graphicsView_Mask.fitInView(self.scene_Mask.sceneRect(), Qt.KeepAspectRatio)

        # ---------------------------------------------
        # OPTIMIZATION & SELECTION LOGIC
        # ---------------------------------------------

        def find_optimal_gamma(method2):
            results = []
            theOCTCC = np.copy(self.OCTCC)

            # --- METHOD A: OLD SIMPLE THRESHOLDING ---
            def generate_simple_mask(enface_image, thresh_val):
                """Original logic: Gaussian Blur -> MinMax Scale -> Simple Threshold"""
                self.send_and_display_the_log(f"Generating mask using Simple Thresholding ({thresh_val})...")

                # Ensure 2D
                if enface_image.ndim == 3:
                    J_input = enface_image[:, :, 0]
                else:
                    J_input = enface_image

                # 1. Blur
                J = cv2.GaussianBlur(J_input, (3, 3), cv2.BORDER_DEFAULT)

                # 2. Scale Min-Max
                min_val = J.min()
                max_val = J.max()
                if max_val - min_val > 0:
                    J = (J - min_val) / (max_val - min_val)

                # 3. Threshold
                binary = J > thresh_val

                # 4. Cleanup
                cleaned_mask = remove_small_objects(binary, 200)

                # Save Debug
                imwrite(os.path.join(destination_rawop, 'Debug_Mask_1_Vessels_OldMethod.png'),
                        np.uint8(cleaned_mask) * 255)

                return cleaned_mask

            # --- METHOD B: NEW FRANGI VESSELNESS ---
            def generate_frangi_mask(enface_image):
                """New logic: Frangi Vesselness -> Threshold -> Morphological Closing"""
                self.send_and_display_the_log("Generating mask using Frangi Vesselness...")

                # Ensure 2D
                if enface_image.ndim == 3:
                    img_2d = enface_image[:, :, 0]
                else:
                    img_2d = enface_image

                # 1. Normalize
                img_norm = img_2d.astype(float)
                if img_norm.max() > img_norm.min():
                    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())

                # 2. Frangi Filter
                sigmas = np.linspace(3.5, 10.0, 4)
                vesselness = frangi(img_norm, sigmas=sigmas, black_ridges=False)

                # 3. Threshold
                thresh_val = 0.12 * vesselness.max()
                binary = vesselness > thresh_val

                # 4. Cleanup (Closing + Remove Small Objects)
                footprint = disk(2)
                bridged_mask = binary_closing(binary, footprint)
                cleaned_mask = remove_small_objects(bridged_mask, min_size=500)

                # Save Debug
                imwrite(os.path.join(destination_rawop, 'Debug_Mask_1_Vessels_Frangi.png'),
                        np.uint8(cleaned_mask) * 255)

                return cleaned_mask

            # --- Helper: Apply Gaussian ---
            def apply_gaussian_filter(image):
                img_float = image.astype(float)
                gaussian_img = cv2.GaussianBlur(img_float, (13, 13), 15)
                return gaussian_img

            # --- Helper: Analyze Stats ---
            def analyze_skewness_and_std(parameters, skewness_values, std_values):
                min_std_index = np.argmin(std_values)
                lowest_std = std_values[min_std_index]
                lowest_std_parameter = parameters[min_std_index]

                crossover_parameter = None
                for i in range(1, len(skewness_values)):
                    if skewness_values[i - 1] < 0 and skewness_values[i] >= 0:
                        x1, x2 = parameters[i - 1], parameters[i]
                        y1, y2 = skewness_values[i - 1], skewness_values[i]
                        crossover_parameter = 0.1 + x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                        crossover_skewness = 0
                        break

                if crossover_parameter is None:
                    min_skew_index = np.argmin(skewness_values)
                    crossover_skewness = skewness_values[min_skew_index]
                    crossover_parameter = parameters[min_skew_index] + 0.2

                return lowest_std, lowest_std_parameter, crossover_skewness, crossover_parameter

            # Helper to save images safely
            def super_imwrite(filename, img):
                if img.dtype != np.uint8: img = np.uint8(np.clip(img, 0, 255))
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                cv2.imwrite(filename, img)

            # =========================================================
            # STEP 1: CONSTRUCT TOTAL EXCLUSION MASK (Vessels + GA + HyperTD)
            # =========================================================

            # A. Generate Vessel Mask
            if self.checkBox_oldthresh.isChecked():
                vessel_mask = generate_simple_mask(self.OCTARetina, 0.6)
            else:
                vessel_mask = generate_frangi_mask(self.OCTARetina)

            # B. Save User Masks to Debug for confirmation
            if hasattr(self, 'GAmask') and self.GAmask is not None:
                super_imwrite(os.path.join(destination_rawop, 'Debug_Mask_2_GA.png'),
                              self.GAmask.astype(np.uint8) * 255)

            if hasattr(self, 'HyperTDmask') and self.HyperTDmask is not None:
                super_imwrite(os.path.join(destination_rawop, 'Debug_Mask_3_HyperTD.png'),
                              self.HyperTDmask.astype(np.uint8) * 255)

            # C. Combine Everything
            # This 'total_exclusion_mask' represents every pixel we should IGNORE when calculating statistics
            total_exclusion_mask = np.copy(vessel_mask)

            if hasattr(self, 'GAmask') and self.GAmask is not None:
                # Resize check handled in load_mask, but double check
                if self.GAmask.shape == total_exclusion_mask.shape:
                    total_exclusion_mask = np.logical_or(total_exclusion_mask, self.GAmask)

            if hasattr(self, 'HyperTDmask') and self.HyperTDmask is not None:
                if self.HyperTDmask.shape == total_exclusion_mask.shape:
                    total_exclusion_mask = np.logical_or(total_exclusion_mask, self.HyperTDmask)

            # D. Save the Final Mask used for Optimization
            super_imwrite(os.path.join(destination_rawop, 'Debug_Mask_4_Total_Exclusion_Used_For_Stats.png'),
                          total_exclusion_mask.astype(np.uint8) * 255)

            parameter_values = []
            skewness_values = []
            std_values = []
            gaussian_std_values = []

            # =========================================================
            # STEP 2: LOOP THROUGH GAMMA PARAMETERS
            # =========================================================
            # [Inside find_optimal_gamma loop]
            # [Inside find_optimal_gamma loop]
            for parameter in np.arange(0.2, 4.01, 0.2):

                # 1. Define what to exclude from STATS (Normalization)
                # We want to exclude GA, Vessels, AND HyperTD from the statistics
                stats_mask = np.copy(vessel_mask)
                if hasattr(self, 'GAmask') and self.GAmask is not None:
                    stats_mask = np.logical_or(stats_mask, self.GAmask)

                # --- FIX: ADD THIS BLOCK ---
                if hasattr(self, 'HyperTDmask') and self.HyperTDmask is not None:
                    stats_mask = np.logical_or(stats_mask, self.HyperTDmask)
                # ---------------------------

                # 2. Call cc_comp with exclusion_mask
                # Pass stats_mask so cc_comp ignores HyperTD/GA when calculating percentiles.
                # Pass hyperTD_mask=None so the image is generated "cleanly" (compensated) for Gaussian Blur.
                para_img_comp, para_img_uncomp = cc_comp(theOCTCC,
                                                         theOCTCC,
                                                         self.OCTARetina,
                                                         para1=self.CompPara,
                                                         para2=parameter,
                                                         para3=0.001,
                                                         exclusion_mask=stats_mask,  # <--- Includes GA + HyperTD
                                                         hyperTD_mask=None)  # <--- Keep None for loop

                # ... (Rest of function remains the same) ...

                para_img_comp_float = para_img_comp.astype(float) * 255

                # Apply Gaussian filter
                # Now that the input is uniformly compensated, no bright artifacts will bleed out
                para_img_comp_gaussian = apply_gaussian_filter(para_img_comp.astype(float) * 255)

                # --- APPLY TOTAL EXCLUSION MASK (Set to NaN) ---
                # This removes Vessels, GA, AND HyperTD from stats calc
                # This is where the actual exclusion happens for the math
                if para_img_comp_float.ndim == 3:
                    # Handle 3 channel images if necessary
                    mask_3d = np.stack([total_exclusion_mask] * 3, axis=2)
                    para_img_comp_float[mask_3d] = np.nan
                    para_img_comp_gaussian[mask_3d] = np.nan
                else:
                    para_img_comp_float[total_exclusion_mask] = np.nan
                    para_img_comp_gaussian[total_exclusion_mask] = np.nan

                # Stats Calculation
                original_scaled = np.copy(para_img_comp_float)
                valid_pixels = para_img_comp_float[~np.isnan(para_img_comp_float)]

                if len(valid_pixels) > 0:
                    current_std = np.nanstd(para_img_comp_float, ddof=1)
                    current_mean = np.nanmean(original_scaled)
                    gaussian_std = np.nanstd(para_img_comp_gaussian, ddof=1)

                    # CV
                    cv = np.nanstd(valid_pixels, ddof=1) / current_mean if current_mean != 0 else np.nan
                    # Skewness
                    current_skewness = skew(valid_pixels.flatten(), nan_policy='omit')
                else:
                    current_std, current_mean, gaussian_std, cv, current_skewness = np.nan, np.nan, np.nan, np.nan, np.nan

                # Save debug images
                super_imwrite(os.path.join(destination_rawop, f'0para_comp_gaussian_{parameter:.2f}_img.tiff'),
                              para_img_comp_gaussian)
                # Just for visualization
                ppp = np.copy(para_img_comp.astype(float) * 255)
                super_imwrite(os.path.join(destination_rawop, f'0para_{parameter:.2f}_img.tiff'), ppp)

                # Store
                results.append({
                    'Parameter': f"{parameter:.2f}",
                    'SD': f"{current_std:.6f}",
                    'Gaussian_SD': f"{gaussian_std:.6f}",
                    'Mean': f"{current_mean:.6f}",
                    'CV': f"{cv:.6f}",
                    'Skewness': f"{current_skewness:.6f}"
                })

                parameter_values.append(parameter)
                skewness_values.append(current_skewness)
                std_values.append(current_std)
                gaussian_std_values.append(gaussian_std)

            # =========================================================
            # STEP 3: ANALYZE RESULTS
            # =========================================================
            lowest_std, lowest_std_parameter, skew_value, best_parameter = analyze_skewness_and_std(
                parameter_values, skewness_values, std_values)

            self.send_and_display_the_log(
                f"Original: Lowest SD value: {lowest_std_parameter:.2f}, SD: {lowest_std:.2f}")

            min_gaussian_std_index = np.argmin(gaussian_std_values)
            lowest_gaussian_std = gaussian_std_values[min_gaussian_std_index]
            gaussian_optimal_parameter = parameter_values[min_gaussian_std_index]

            self.send_and_display_the_log(
                f"Gaussian: Lowest SD value: {gaussian_optimal_parameter:.2f}, SD: {lowest_gaussian_std:.2f}")

            if best_parameter == 0.0:
                best_parameter = 1.0

            self.CompPara2 = np.copy(best_parameter)

            # Write CSV
            csv_path = os.path.join(destination_rawop, '0_gamma_analysis_results.csv')
            from Analysis.CCFD_Standalone.CSV_Analyzer import select_and_analyze
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['Parameter', 'SD', 'Gaussian_SD', 'Mean', 'CV', 'Skewness']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
            try:
                select_and_analyze(path=csv_path, include_gaussian_sd=True)
            except:
                select_and_analyze(path=csv_path)

            use_skewness_instead = (abs(gaussian_optimal_parameter - 4.0) < 0.01)

            if method2 == 1:
                if use_skewness_instead:
                    self.send_and_display_the_log(f"Using Skewness (Gaussian Maxed): {best_parameter:.2f}")
                    return best_parameter
                else:
                    self.send_and_display_the_log(f"Using Gaussian SD: {gaussian_optimal_parameter:.2f}")
                    return gaussian_optimal_parameter
            else:
                self.send_and_display_the_log(f"Using Skewness: {best_parameter:.2f}")
                return best_parameter

        if self.checkBox_toggleforSD.isChecked():
            method2 = 1
        else:
            method2 = 0

        # Run the Finder
        self.CompPara2 = find_optimal_gamma(method2)

        # Force override if checkbox is checked
        if self.checkBox_togglegamma.isChecked():
            self.CompPara2 = float(self.lineEdit_forcegamma.text())

        # Generate Final Image using the found parameter
        # This will merge the HyperTD mask (uncompensated) into the result
        final_stats_mask = np.zeros(self.target_shape, dtype=bool)
        if self.GAmask is not None:
            final_stats_mask = np.logical_or(final_stats_mask, self.GAmask)
        # We usually exclude HyperTD from stats here too so background looks consistent
        if self.HyperTDmask is not None:
            final_stats_mask = np.logical_or(final_stats_mask, self.HyperTDmask)

        # Generate Final Image
        # Pass HyperTDmask here so they appear BRIGHT in the final result
        self.img_comp, self.img_uncomp = cc_comp(self.OCTACC, self.OCTCC,
                                                 self.OCTARetina,
                                                 para1=self.CompPara,
                                                 para2=self.CompPara2,
                                                 para3=self.TailPara,
                                                 exclusion_mask=final_stats_mask,
                                                 hyperTD_mask=self.HyperTDmask)

        if self.img_comp.shape != self.target_shape:
            self.img_comp = resize(self.img_comp, self.target_shape)

        self.send_and_display_the_log(f"Final Gamma used: {self.CompPara2:.2f}")

        # VISUALIZATION FIX: Black out GA regions
        display_img = np.uint8(np.clip(self.img_comp * 255, 0, 255))

        if self.GAmask is not None:
            if self.GAmask.shape != display_img.shape:
                disp_mask = resize_mask_nearest(self.GAmask, display_img.shape)
            else:
                disp_mask = self.GAmask
            display_img[disp_mask == True] = 0

        self.scene_ImageComp.update_image(display_img)
        self.graphicsView_ImageComp.fitInView(self.scene_ImageComp.sceneRect(), Qt.KeepAspectRatio)

    def load_img_CC(self):
        """
        preview the CC FD images
        :return:
        """

        try:
            # file_url = 'F:/OCTpy/Test/demo/CCFD.bmp'
            # img = imread(file_url)
            self.send_and_display_the_log('Running CC flow defict algorithm...')
            print(self.url_ccfd)
            # time.sleep(4)
            self.update_parameters()
            if self.checkBox_Fovea.isChecked():
                self.fov_x = np.uint16(self.lineEdit_fovX.text())
                self.fov_y = np.uint16(self.lineEdit_fovY.text())
                fovea_center = (self.fov_x, self.fov_y)
            else:
                fovea_center = None
                self.fov_x = 250
                self.fov_y = 250

            if self.checkBox_UseMask.isChecked():
                # If GAmask exists (even if None), use it. If None, default to blank.
                if self.GAmask is not None:
                    mask = self.GAmask
                else:
                    self.send_and_display_the_log("Warning: 'Use Mask' checked but no mask loaded. Using blank.")
                    mask = np.zeros(self.target_shape, dtype=bool)
            else:
                # If unchecked, explicitly blank
                mask = np.zeros(self.target_shape, dtype=bool)

            if self.checkBox_usePhansalkar.isChecked():
                print('Using Phansalkar thresholding')
                self.CCFD_img, img_color, self.CCFDSize, self.CCFDD, self.usedK, self.CCFD_ori = phansalkar_thresholding(
                    self.OCTARetina, self.img_comp,
                    scansize=self.ScanSize,
                    img_mask=mask,
                    scaleX=float(self.lineEdit_ScaleFactor_X.text()),
                    scaleY=float(self.lineEdit_ScaleFactor_Y.text()))

            else:
                if self.checkBox_Knumber.isChecked():  # use the predefined k value
                    print('Using fuzzy thresholding')
                    if self.checkBox_oldthresh.isChecked():
                        threshold_largevessel = 0.6
                        self.send_and_display_the_log(
                            'Running with old thresholds...')
                    else:
                        threshold_largevessel = 0.68
                        self.send_and_display_the_log(
                            'Running with new thresholds...')

                    self.CCFD_img, img_color, self.CCFDSize, self.CCFDD, self.usedK, self.CCFD_ori, self.vols, self.meansizes, final_total_mask = fuzz_CC_thresholding(
                        self.OCTARetina, self.img_comp, threshold_largevessel,
                        scansize=self.ScanSize,
                        k_val=self.BestK,
                        CCthresAdj=self.CCthres,
                        img_mask=mask,
                        scaleX=float(self.lineEdit_ScaleFactor_X.text()),
                        scaleY=float(self.lineEdit_ScaleFactor_Y.text()), save_path=self.url_ccfd,
                        def_fovea_center=fovea_center)
                    print('use the input best K:', self.BestK)
                else:
                    print('Using fuzzy thresholding')
                    if self.checkBox_oldthresh.isChecked():
                        threshold_largevessel = 0.6
                        self.send_and_display_the_log(
                            'Running with old thresholds...')
                    else:
                        threshold_largevessel = 0.68
                        self.send_and_display_the_log(
                            'Running with new thresholds...')

                    # Interchanged input for compensated Structure CC slab

                    self.CCFD_img, img_color, self.CCFDSize, self.CCFDD, self.usedK, self.CCFD_ori, self.vols, self.meansizes, final_total_mask = fuzz_CC_thresholding(
                        self.OCTARetina, self.img_comp, threshold_largevessel,
                        scansize=self.ScanSize,
                        CCthresAdj=self.CCthres,
                        img_mask=mask,
                        scaleX=float(self.lineEdit_ScaleFactor_X.text()),
                        scaleY=float(self.lineEdit_ScaleFactor_Y.text()),
                        save_path=self.url_ccfd)

            # plt.imshow(img_color)
            # plt.show()

            # update display
            if 'final_total_mask' in locals():
                display_total = np.uint8(final_total_mask) * 255
                self.scene_Mask_ExclusionCombined.update_image(display_total)
                self.graphicsView_Mask_ExclusionCombined.fitInView(
                    self.scene_Mask_ExclusionCombined.sceneRect(), Qt.KeepAspectRatio)
                self.send_and_display_the_log("Updated Final Exclusion Mask (User + Vessels)")
            self.scene_CCflowvoids.update_image(self.CCFD_img)
            self.graphicsView_CCflowvoids.fitInView(
                self.scene_CCflowvoids.sceneRect(), Qt.KeepAspectRatio)

            # display the number
            ccfdd_str = '%0.5f' % self.CCFDD
            fdsize_str = '%0.5f' % self.CCFDSize
            if 'final_total_mask' in locals():
                display_total = np.uint8(final_total_mask) * 255
                self.scene_Mask_ExclusionCombined.update_image(display_total)
                self.graphicsView_Mask_ExclusionCombined.fitInView(
                    self.scene_Mask_ExclusionCombined.sceneRect(), Qt.KeepAspectRatio)

            # Update text stats
            self.lineEdit_FD.setText(ccfdd_str)
            self.lineEdit_MeanSize.setText(fdsize_str)
            self.send_and_display_the_log("CC flow deficits Calculation completed")

        except Exception as e:
            print(e)
            print(traceback.format_exc())

        # %%

    def on_button_clicked_open_batch(self):
        """ Open the folder for batch processing
        :return:
        """
        url = self.url_ccfd
        try:
            url = self.open_folder(url)

            # put the url to the line edit
            url = str(Path(url))
            self.lineEdit_OpenPath.setText(url)
            # self.lineEdit_SavePath.setText(url)

            print('open batch folder = ', url)

            file_df = self.search_files_in_folder(url)
            self.file_df = file_df  # put this to the class
            self.batch_url = url

            # clean previous logs
            self.listWidget_BatchList.clear()

            # display
            self.display_batch_files_to_list(file_df)
        except Exception as e:
            print(e)

    def search_files_in_folder(self, url):
        """ get files by file pattern
        Args:
            url: the url of folders
        Return:
            dataframe
        """
        try:

            file_df = pd.DataFrame(
                columns=['path', 'folder', 'path1', 'path2', 'path3'])

            # file_list = os.listdir(url)
            # list all folders in the url
            for folder in os.listdir(url):
                folder_path = os.path.join(url, folder)
                if fnmatch.fnmatch(folder,
                                   self.lineEdit_FilePattern1.text()) and os.path.isdir(
                    folder_path):
                    print('folder=', folder_path)
                    # search files in the folder
                    B1, B2, B3 = False, False, False
                    for file in os.listdir(folder_path):
                        if fnmatch.fnmatch((file),
                                           self.lineEdit_FilePattern2.text()):
                            B1 = True
                            file1 = file
                        if fnmatch.fnmatch((file),
                                           self.lineEdit_FilePattern3.text()):
                            B2 = True
                            file2 = file
                        if fnmatch.fnmatch((file),
                                           self.lineEdit_FilePattern4.text()):
                            B3 = True
                            file3 = file

                    # find all files
                    if B1 and B2 and B3:
                        file_df = file_df.append(
                            {'path': url, 'folder': folder, 'path1': file1,
                             'path2': file2, 'path3': file3}, ignore_index=True)
            print(file_df)
        except Exception as e:
            print(e)
        return file_df

    def display_batch_files_to_list(self, file_df):
        """ display batch files to list
        """
        for index, row in file_df.iterrows():
            self.listWidget_BatchList.addItem('--- Batch ---  ' + str(index))
            self.listWidget_BatchList.addItem(
                'Folder: ' + os.path.basename(str(row['folder'])))
            self.listWidget_BatchList.addItem('CC flow' + row['path1'])
            self.listWidget_BatchList.addItem('Retina' + row['path2'])
            self.listWidget_BatchList.addItem('CC stru' + row['path3'])

    def on_button_clicked_run_single(self):
        """Run the select files from the list
        :return:
        """
        try:
            # get the file names from list
            row = self.listWidget_BatchList.currentRow()

            row_num = row // 5

            url1 = os.path.join(self.file_df.iloc[row_num, 0],
                                self.file_df.iloc[row_num, 1],
                                self.file_df.iloc[row_num, 2])
            url2 = os.path.join(self.file_df.iloc[row_num, 0],
                                self.file_df.iloc[row_num, 1],
                                self.file_df.iloc[row_num, 3])
            url3 = os.path.join(self.file_df.iloc[row_num, 0],
                                self.file_df.iloc[row_num, 1],
                                self.file_df.iloc[row_num, 4])

            self.process_single_file(url1, url2, url3)
        except Exception as e:
            print(e)

    def on_button_clicked_run_batch(self):
        """Run batch files
        :return:
        """
        try:
            for index, row in self.file_df.iterrows():
                row_num = index

                url1 = os.path.join(self.file_df.iloc[row_num, 0],
                                    self.file_df.iloc[row_num, 1],
                                    self.file_df.iloc[row_num, 2])
                url2 = os.path.join(self.file_df.iloc[row_num, 0],
                                    self.file_df.iloc[row_num, 1],
                                    self.file_df.iloc[row_num, 3])
                url3 = os.path.join(self.file_df.iloc[row_num, 0],
                                    self.file_df.iloc[row_num, 1],
                                    self.file_df.iloc[row_num, 4])

                self.process_single_file(url1, url2, url3)
                # time.sleep(1)
        except Exception as e:
            print(e)

    def process_single_file(self, url1, url2, url3):
        """
        processing single file of the
        :return:
        """

        self.filename = url1
        self.OCTACC = imread(url1)
        self.OCTARetina = imread(url2)
        self.OCTCC = imread(url3)

        try:
            # run the program
            self.load_img_comp()
            self.load_img_CC()

            # save result
            self.save_results()

        except Exception as e:
            print('cannnot run:', e)

    def global_threshold_cc(self):
        """
        Qinqin's golbal threshold method
        :return:
        """

    def clear_all(self):
        """
        clean the display of images
        :return:
        """
        self.scene_Mask.clear()
        self.scene_Mask_HyperTD.clear()
        self.scene_Mask_ExclusionCombined.clear()
        self.scene_CCflowvoids.clear()
        self.scene_ImageComp.clear()
        self.scene_OCTARetina.clear()
        self.scene_OCTACC.clear()
        self.scene_OCTCC.clear()
        self.GAmask = None
        self.update_parameters()
        self.send_and_display_the_log("Inputs cleared")

    def save_results(self):
        """
        Save all results: CCFD maps, Compensated Images, and Mask variations.
        """
        try:
            # Create Output Folder
            # We use a timestamp or unique identifier to avoid overwriting if running multiple times
            base_folder = os.path.dirname(self.fileloc)
            output_folder = os.path.join(base_folder, f'CCFD_Outputs_{self.CompPara2:.2f}')

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            file_url = os.path.join(output_folder, 'CCFD_Final_Clean.png')
            imwrite(file_url, self.CCFD_img)

            # --- 2. Save Original CCFD (Unmasked / No Exclusion) ---
            # self.CCFD_ori is the version before the GA/Vessel mask was applied
            if hasattr(self, 'CCFD_ori') and self.CCFD_ori is not None:
                file_url = os.path.join(output_folder, 'CCFD_Original_Unmasked.png')
                imwrite(file_url, self.CCFD_ori)

            # --- 3. Save Compensated Image (Raw / Without Mask) ---
            # img_comp is Float [0-1], needs conversion to Uint8 [0-255]
            img_comp_uint8 = np.uint8(np.clip(self.img_comp * 255, 0, 255))
            file_url = os.path.join(output_folder, 'Compensated_Flow_Raw.png')
            imwrite(file_url, img_comp_uint8)

            # --- 4. Save Compensated Image (With Exclusion Mask Applied) ---
            if self.GAmask is not None:
                img_comp_masked = np.copy(img_comp_uint8)
                # Ensure mask is resized to match
                if self.GAmask.shape != img_comp_masked.shape:
                    from skimage.transform import resize
                    curr_mask = resize_mask_nearest(self.GAmask, img_comp_masked.shape)
                else:
                    curr_mask = self.GAmask

                img_comp_masked[curr_mask == True] = 0  # Black out excluded areas
                file_url = os.path.join(output_folder, 'Compensated_Flow_ExclusionMasked.png')
                imwrite(file_url, img_comp_masked)

                # Also save the Exclusion Mask itself for reference
                imwrite(os.path.join(output_folder, 'Mask_Exclusion_Used.png'), np.uint8(curr_mask) * 255)

            # --- 5. Save Compensated Image (With HyperTD Mask Applied) ---
            if self.HyperTDmask is not None:
                img_comp_hyper = np.copy(img_comp_uint8)
                if self.HyperTDmask.shape != img_comp_hyper.shape:
                    from skimage.transform import resize
                    curr_hyper = resize_mask_nearest(self.HyperTDmask, img_comp_hyper.shape)
                else:
                    curr_hyper = self.HyperTDmask

                # Highlight HyperTDs (e.g., make them black or white to visualize location)
                # Here we set them to 0 (Black) to see where they are
                img_comp_hyper[curr_hyper == True] = 0
                file_url = os.path.join(output_folder, 'Compensated_Flow_HyperTDMasked.png')
                imwrite(file_url, img_comp_hyper)

                # Save the HyperTD mask itself
                imwrite(os.path.join(output_folder, 'Mask_HyperTD_Used.png'), np.uint8(curr_hyper) * 255)

            self.send_and_display_the_log(f'Images saved successfully to: {output_folder}')

        except Exception as e:
            print('Cannot save images: ', e)
            self.send_and_display_the_log(f'Error saving images: {e}')

        # --- 6. Save CSV Data ---
        # Ensure we have data to save
        if not hasattr(self, 'meansizes') or self.meansizes is None:
            # Handle case where algo didn't run fully
            self.meansizes = [0, 0, 0]
            self.vols = [0, 0, 0]

        result = pd.DataFrame([[
            self.CCFDD,
            self.CCFDSize,
            self.filename,
            self.TailPara,
            self.CompPara2,
            self.ScanSize,
            self.usedK,
            self.fov_x,
            self.fov_y,
            self.vols[0], self.vols[1], self.vols[2],
            self.meansizes[0], self.meansizes[1], self.meansizes[2]
        ]], columns=[
            'ccfdd', 'fdsize', 'filename',
            'TailRemoval', 'CompPara2', 'ScanSize',
            'bestK', 'Fovea X', 'Fovea Y',
            '1mm FD', '3mm FD', '5mm FD',
            '1mm Size', '3mm Size', '5mm Size'
        ])

        try:
            csv_path = os.path.join(output_folder, 'Quantification_Results.csv')
            result.to_csv(csv_path)
            self.send_and_display_the_log('CSV Data saved.')
        except Exception as e:
            print('Cannot save CSV file ', e)

    def send_and_display_the_log(self, text):
        '''Display the logs
        '''
        time = datetime.now().strftime('%m-%d %H:%M:%S -- ')
        str_send = time + text
        self.plainTextEdit_log.appendPlainText(str_send)


if __name__ == "__main__":
    import sys

    # 1. Create the Application
    app = QtWidgets.QApplication(sys.argv)

    # 2. Create an instance of the window
    main_window = CCquanWindow()

    # 3. Show the window
    main_window.show()

    # 4. Start the event loop
    sys.exit(app.exec_())