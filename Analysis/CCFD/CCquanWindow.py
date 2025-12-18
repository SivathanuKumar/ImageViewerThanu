import os
import sys
import csv
import fnmatch
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from cv2 import imread
from imageio import imwrite
from scipy.stats import skew
from skimage.transform import resize
from skimage.morphology import remove_small_objects as pprem

# PyQt5
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

from Analysis.CCFD import QuanCC

# 2. Logic/Algorithm files
from ccquan import cc_comp, phansalkar_thresholding, fuzz_CC_thresholding, scaleMaxMin
from UI.scenes import GraphicsScene2d

class CCquanWindow(QtWidgets.QMainWindow, QuanCC.Ui_MainWindow):
    """
    CC quantification window
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        self.url_ccfd = None
        self.setupUi(self)

        # display
        self.scene_OCTACC = GraphicsScene2d()
        self.graphicsView_OCTACC.setScene(self.scene_OCTACC)

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
        #self.CompPara2 = float(self.lineEdit_CompPara2.text())
        self.ScanSize = int(self.lineEdit_ScanSize.text())
        self.BestK = int(self.lineEdit_BestK.text())
        self.CCthres = float(self.lineEdit_CCthres.text())

    def load_img_OCTACC(self):
        """
        load image 1 OCTA CC to class member
        :return:
        """

        url = self.lineEdit_OpenPath.text()
        file_url = self.openFileNameDialog(url, 'image')
        if os.path.exists(file_url):
            self.filename = file_url
            self.lineEdit_OpenPath.setText(
                os.path.dirname(file_url))  # set the url to the window
            img = imread(file_url)
            self.OCTACC = img
            self.scene_OCTACC.update_image(img)
            self.graphicsView_OCTACC.fitInView(self.scene_OCTACC.sceneRect(),
                                               Qt.KeepAspectRatio)
            self.send_and_display_the_log('Load image: ' + file_url)
        else:
            self.send_and_display_the_log('File does not exist: ' + file_url)

    def load_img_OCTCC(self):
        """
        load image 1 OCT CC to class member
        :return:
        """
        # self.get_current_path()
        url = self.lineEdit_OpenPath.text()
        file_url = self.openFileNameDialog(url, 'image')
        self.file_url_for_ref = file_url
        if os.path.exists(file_url):
            img = imread(file_url)
            self.lineEdit_OpenPath.setText(
                os.path.dirname(file_url))  # set the url to the window

            self.OCTCC = img
            self.scene_OCTCC.update_image(img)
            self.graphicsView_OCTCC.fitInView(self.scene_OCTCC.sceneRect(),
                                              Qt.KeepAspectRatio)
            self.send_and_display_the_log('Load image: ' + file_url)
        else:
            self.send_and_display_the_log('File does not exist: ' + file_url)

    def on_button_clicked_load_mask(self):
        """
        Load the outside mask to exclude
        Returns:
        """
        url = self.lineEdit_OpenPath.text()
        file_url = self.openFileNameDialog(url, 'image')
        if os.path.exists(file_url):
            img = imread(file_url)
            self.lineEdit_OpenPath.setText(
                os.path.dirname(file_url))  # set the url to the window

            # resize and to 8bit
            if img.ndim == 3:
                img = img[:, :, 0]

            img = resize(img, (500, 500))
            self.GAmask = img > 0.7  #

            self.scene_Mask.update_image(img)
            self.graphicsView_Mask.fitInView(self.scene_Mask.sceneRect(),
                                             Qt.KeepAspectRatio)
            self.send_and_display_the_log('Load image: ' + file_url)
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

    def load_img_OCTARetina(self):
        """
        load image 1 OCTA CC to class member
        :return:
        """
        # self.get_current_path()
        url = self.lineEdit_OpenPath.text()
        file_url = self.openFileNameDialog(url, 'image')
        self.fileloc = file_url
        if os.path.exists(file_url):
            img = imread(file_url)
            self.lineEdit_OpenPath.setText(os.path.dirname(file_url))
            self.OCTARetina = img
            self.scene_OCTARetina.update_image(img)
            self.graphicsView_OCTARetina.fitInView(
                self.scene_OCTARetina.sceneRect(), Qt.KeepAspectRatio)
            self.send_and_display_the_log('Load image: ' + file_url)
        else:
            self.send_and_display_the_log('File does not exist: ' + file_url)

    def load_img_comp(self):
        """
        preview the compensation
        :return:
        """
        """
        file_url = 'F:/OCTpy/Test/demo/MaxCC corrected by CCstructureSumAPAR.tiff'
        img = imread(file_url)
        self.send_and_display_the_log('Run image compensation...')
        time.sleep(2)
        """

        self.send_and_display_the_log('Previewing OCTA CC image compensation...')
        self.update_parameters()

        try:
            destination_rawop = os.path.dirname(self.fileloc) + "/CCFD_debug"
            if not os.path.exists(destination_rawop):
                os.makedirs(destination_rawop)

            self.url_ccfd = os.path.dirname(self.fileloc) + '/CCFD_Outputs/'
            if not os.path.exists(self.url_ccfd):
                os.makedirs(self.url_ccfd)
        except Exception as e:
            print('cannot save images: ', e)

        def find_optimal_gamma(method2):
            results = []  # List to store results for CSV
            theOCTCC = np.copy(self.OCTCC)

            def process_retinal_image(enface_image, overlay_image):
                J = cv2.GaussianBlur(enface_image, (3, 3), cv2.BORDER_DEFAULT)
                J = scaleMaxMin(J, J.min(), J.max())
                # change the binary value
                binary = J > 0.63
                from skimage.morphology import remove_small_objects as pprem
                cleaned_mask = pprem(binary, 200)
                imwrite(
                    os.path.join(destination_rawop, f'cleaned_mask_img.png'),
                    np.uint8(cleaned_mask) * 255)

                # Convert overlay image to float for NaN compatibility
                result = overlay_image.astype(float)

                # Replace vessel regions with NaN
                result[cleaned_mask] = np.nan
                return result, cleaned_mask

            def apply_gaussian_filter(image):
                """Apply a Gaussian filter with size [13,13] and sigma=15"""
                # Convert to float for processing
                img_float = image.astype(float)
                # Apply Gaussian blur
                gaussian_img = cv2.GaussianBlur(img_float, (13, 13), 15)
                return gaussian_img

            def analyze_skewness_and_std(parameters, skewness_values, std_values):
                # Find lowest std value and its parameter
                min_std_index = np.argmin(std_values)
                lowest_std = std_values[min_std_index]
                lowest_std_parameter = parameters[min_std_index]

                # First try to find crossover point
                crossover_parameter = None
                for i in range(1, len(skewness_values)):
                    if skewness_values[i - 1] < 0 and skewness_values[i] >= 0:
                        # Linear interpolation to find more precise crossover point
                        x1, x2 = parameters[i - 1], parameters[i]
                        y1, y2 = skewness_values[i - 1], skewness_values[i]
                        crossover_parameter = 0.1 + x1 + (0 - y1) * (x2 - x1) / (
                                y2 - y1)
                        crossover_skewness = 0  # At crossover point, skewness is 0
                        break

                # If no crossover found, use lowest skewness value
                if crossover_parameter is None:
                    min_skew_index = np.argmin(skewness_values)
                    crossover_skewness = skewness_values[min_skew_index]
                    crossover_parameter = parameters[min_skew_index] + 0.2

                return lowest_std, lowest_std_parameter, crossover_skewness, crossover_parameter

            parameter_values = []
            skewness_values = []
            std_values = []
            gaussian_std_values = []  # Track standard deviations after Gaussian filtering

            for parameter in np.arange(0.2, 4.01, 0.2):
                mask_para, dmask = process_retinal_image(self.OCTARetina, theOCTCC)
                if hasattr(self, 'GAmask') and self.GAmask is not None and self.GAmask.size > 0:
                    mask_para[self.GAmask] = np.nan
                mask_nan = np.isnan(mask_para)

                # Process the image with the current parameter
                para_img_comp, para_img_uncomp = cc_comp(theOCTCC,
                                                         theOCTCC,
                                                         self.OCTARetina,
                                                         para1=self.CompPara,
                                                         para2=parameter,
                                                         para3=0.001)
                ppp = np.copy(para_img_comp.astype(float)*255)
                para_img_comp_float = para_img_comp.astype(float)*255
                para_img_comp_float[mask_nan] = np.nan

                # Apply Gaussian filter to the processed image BEFORE applying the mask
                para_img_comp_gaussian = apply_gaussian_filter(para_img_comp.astype(float) * 255)
                # Apply the mask after Gaussian filtering
                para_img_comp_gaussian[mask_nan] = np.nan

                mask_para_skew = para_img_comp
                dmask_skew = np.zeros([500, 500])
                original_scaled = np.copy(para_img_comp_float)
                valid_pixels = para_img_comp_float[~np.isnan(para_img_comp_float)]

                if len(valid_pixels) > 0:
                    # Calculate metrics for original image
                    current_std = np.nanstd(para_img_comp_float, ddof=1)
                    current_mean = np.nanmean(original_scaled)

                    # Calculate metrics for Gaussian filtered image
                    gaussian_std = np.nanstd(para_img_comp_gaussian, ddof=1)

                    # Calculate CV using non-NaN values
                    valid_pixels_original = original_scaled[
                        ~np.isnan(mask_para_skew)]
                    cv = np.nanstd(valid_pixels_original,
                                   ddof=1) / current_mean if current_mean != 0 else np.nan

                    # Calculate skewness only on non-NaN values
                    current_skewness = skew(valid_pixels_original.flatten(),
                                            nan_policy='omit')
                else:
                    current_std = np.nan
                    current_mean = np.nan
                    gaussian_std = np.nan
                    cv = np.nan
                    current_skewness = np.nan

                def super_imwrite(filename, img):
                    """
                    Wrapper around cv2.imwrite that ensures proper image saving
                    """
                    # Ensure the image is in the correct format for saving
                    if img.dtype != np.uint8:
                        img = np.uint8(np.clip(img, 0, 255))

                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(filename), exist_ok=True)

                    # Save the image
                    cv2.imwrite(filename, img)

                def save_processed_images(para_img_comp, mask_para, dmask,
                                          parameter, destination_rawop):
                    # For the original comparison image
                    super_imwrite(os.path.join(destination_rawop,
                                               f'0para_comp_{parameter:.2f}_img.tiff'),
                                  (para_img_comp))

                # Save original processed image
                save_processed_images(para_img_comp_float, mask_para_skew, dmask_skew,
                                      parameter, destination_rawop)

                # Save Gaussian filtered image
                super_imwrite(os.path.join(destination_rawop,
                                           f'0para_comp_gaussian_{parameter:.2f}_img.tiff'),
                              np.uint8(np.clip(para_img_comp_gaussian, 0, 255)))
                super_imwrite(os.path.join(destination_rawop,
                                           f'0para_{parameter:.2f}_img.tiff'),
                              np.uint8(np.clip(ppp, 0, 255)))

                # Store results for CSV
                results.append({
                    'Parameter': f"{parameter:.2f}",
                    'SD': f"{current_std:.6f}",
                    'Gaussian_SD': f"{gaussian_std:.6f}",  # Add Gaussian SD to results
                    'Mean': f"{current_mean:.6f}",
                    'CV': f"{cv:.6f}",
                    'Skewness': f"{current_skewness:.6f}"
                })

                parameter_values.append(parameter)
                skewness_values.append(current_skewness)
                std_values.append(current_std)
                gaussian_std_values.append(gaussian_std)  # Store Gaussian SD values

            # Find optimal parameter based on original SD
            lowest_std, lowest_std_parameter, skew_value, best_parameter = analyze_skewness_and_std(parameter_values,
                                                                                                    skewness_values,
                                                                                                    std_values)
            self.send_and_display_the_log(
                f"Original: Lowest SD value: {lowest_std_parameter:.2f}, with a SD of: {lowest_std:.2f}")

            # Find optimal parameter based on Gaussian filtered SD
            min_gaussian_std_index = np.argmin(gaussian_std_values)
            lowest_gaussian_std = gaussian_std_values[min_gaussian_std_index]
            gaussian_optimal_parameter = parameter_values[min_gaussian_std_index]
            self.send_and_display_the_log(
                f"Gaussian: Lowest SD value: {gaussian_optimal_parameter:.2f}, with a SD of: {lowest_gaussian_std:.2f}")

            if best_parameter == 0.0:
                best_parameter = 1.0
                self.send_and_display_the_log(
                    "No transition from negative to positive skewness found.")
            else:
                self.send_and_display_the_log(
                    f"Optimal Skewness para: {best_parameter:.2f}")

            self.CompPara2 = np.copy(best_parameter)

            # Write results to CSV
            csv_path = os.path.join(destination_rawop,
                                    '0_gamma_analysis_results.csv')
            from CSV_Analyzer import select_and_analyze
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['Parameter', 'SD', 'Gaussian_SD', 'Mean', 'CV', 'Skewness']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    writer.writerow(row)

            # Modify the select_and_analyze function to include Gaussian SD plot
            # We'll use try-except to maintain compatibility if the function doesn't support additional plots
            try:
                select_and_analyze(path=csv_path, include_gaussian_sd=True)
            except:
                select_and_analyze(path=csv_path)

            # Check if the optimal Gaussian SD value is 4.0 (maximum in our range)
            use_skewness_instead = (abs(gaussian_optimal_parameter - 4.0) < 0.01)

            if method2 == 1:
                if use_skewness_instead:
                    self.send_and_display_the_log(
                        f"Gaussian+SD value is at maximum (4.0), using Skewness value instead: {best_parameter:.2f}")
                    return best_parameter
                else:
                    self.send_and_display_the_log(
                        f"Optimal Gaussian+SD based Gamma metric is: {gaussian_optimal_parameter:.2f}")
                    return gaussian_optimal_parameter
            else:
                self.send_and_display_the_log(
                    f"Optimal Skewness based Gamma metric is: {best_parameter:.2f}")
                return best_parameter

        if self.checkBox_toggleforSD.isChecked():
            method2 = 1
        else:
            method2 = 0

        self.CompPara2 = find_optimal_gamma(method2)

        if self.checkBox_togglegamma.isChecked():

            self.CompPara2 = float(self.lineEdit_forcegamma.text())


        self.img_comp, self.img_uncomp = cc_comp(self.OCTACC, self.OCTCC,
                                                 self.OCTARetina,
                                                 para1=self.CompPara,
                                                 para2=self.CompPara2,
                                                 para3=self.TailPara)
        self.img_comp = resize(self.img_comp, (500,500))
        self.send_and_display_the_log(
            f"The final Gamma used for processing is: {self.CompPara2:.2f} for compensation")
        self.scene_ImageComp.update_image(self.img_comp)
        self.graphicsView_ImageComp.fitInView(self.scene_ImageComp.sceneRect(),
                                              Qt.KeepAspectRatio)



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
                print('Using masks...')
                try:

                    mask = self.GAmask
                except Exception as e:
                    print('Load mask first')
            else:
                self.GAmask = np.zeros_like(self.img_comp)
                mask = self.GAmask

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

                    self.CCFD_img, img_color, self.CCFDSize, self.CCFDD, self.usedK, self.CCFD_ori, self.vols, self.meansizes = fuzz_CC_thresholding(
                        self.OCTARetina, self.img_comp, threshold_largevessel,
                        scansize=self.ScanSize,
                        k_val=self.BestK,
                        CCthresAdj=self.CCthres,
                        img_mask=mask,
                        scaleX=float(self.lineEdit_ScaleFactor_X.text()),
                        scaleY=float(self.lineEdit_ScaleFactor_Y.text()), save_path=self.url_ccfd, def_fovea_center=fovea_center)
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

                    self.CCFD_img, img_color, self.CCFDSize, self.CCFDD, self.usedK, self.CCFD_ori, self.vols, self.meansizes = fuzz_CC_thresholding(
                        self.OCTARetina, self.img_comp, threshold_largevessel,
                        scansize=self.ScanSize,
                        CCthresAdj=self.CCthres,
                        img_mask=mask,
                        scaleX=float(self.lineEdit_ScaleFactor_X.text()),
                        scaleY=float(self.lineEdit_ScaleFactor_Y.text()),
                    save_path= self.url_ccfd)

            # plt.imshow(img_color)
            # plt.show()

            # update display
            self.scene_CCflowvoids.update_image(self.CCFD_img)
            self.graphicsView_CCflowvoids.fitInView(
                self.scene_CCflowvoids.sceneRect(), Qt.KeepAspectRatio)

            # display the number
            ccfdd_str = '%0.5f' % self.CCFDD
            fdsize_str = '%0.5f' % self.CCFDSize
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
        save the results to local folder
        :return:
        """
        # save the images to the drive
        try:
            url = os.path.dirname(self.fileloc) + f'\\CCFD_Outputs\\{self.CompPara2:.2f}'
            if not os.path.exists(url):
                os.makedirs(url)

            file_url = url + '_CCFD.png'
            imwrite(file_url, self.CCFD_img)

            file_url = url + '_CCFD_original.png'
            imwrite(file_url, self.CCFD_ori)

            file_url = url + '_img_comp.png'
            imwrite(file_url, np.uint8(self.img_comp*255))

            self.img_comp_with_mask = np.copy(self.img_comp)
            self.img_comp_with_mask[self.GAmask == True] = 0
            file_url = url + '_img_comp_with_mask.png'
            imwrite(file_url, np.uint8(self.img_comp_with_mask*255))

        except Exception as e:
            print('cannot save images: ', e)

        # save the values and parameters
        result = pd.DataFrame([[self.CCFDD, self.CCFDSize, self.filename,
                                self.TailPara, self.CompPara2, self.ScanSize,
                                self.usedK, self.fov_x, self.fov_y, self.vols[0], self.vols[1], self.vols[2], self.meansizes[0],self.meansizes[1],self.meansizes[2] ]],
                              columns=['ccfdd', 'fdsize', 'filename',
                                       'TailRemoval', 'CompPara2', 'ScanSize',
                                       'bestK', 'Fovea X', 'Fovea Y', '1mm FD', '3mm FD', '5mm FD', '1mm Size', '3mm Size', '5mm Size'])
        try:
            result.to_csv(url + '_cc_result.csv')
            self.send_and_display_the_log('File saved')
        except Exception as e:
            print('cannot save csv file ', e)

    def send_and_display_the_log(self, text):
        '''Display the logs
        '''
        time = datetime.now().strftime('%m-%d %H:%M:%S -- ')
        str_send = time + text
        self.plainTextEdit_log.appendPlainText(str_send)

