import os
import sys
import csv
import io as ioio
from datetime import datetime
from pathlib import Path

# Data & Math
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# Image Processing
import tifffile
from imageio import imwrite
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# PyQt5
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from Utils.data_class import state
from UI.scenes import GraphicsScene2d

from Analysis.Registration_Codes import SITK_Registration, ORLdifferenceUI


class ORLdifference(QtWidgets.QMainWindow, ORLdifferenceUI.Ui_MainWindow):
    """
    ORL difference window test
    """

    def __init__(self):
        super(self.__class__, self).__init__()

        #actualthicknessvalues
        self.actbasethick = None
        self.actfollowthick = None
        self.carryover = None
        self.saveloc = None
        self.difsave = None
        self.set_thickness_data_frame()


        self.differenceORL = None
        self.pil_image_Baseline = None
        self.pil_image_FollowUp = None
        self.pil_image_Baseline_thick = None
        self.pil_image_FollowUp_thick = None
        self.filename_FollowUp = None
        self.filename_FollowUp_thickness = None
        self.filename_Baseline = None
        self.filename_Baseline_thickness = None
        self.nptrans = None
        self.npthick = None
        self.fovea_center = [250,250]
        self.actual_values = {'1mm': 160.143, '3mm': 155.154, '5mm': 150.305,
                         'complete': 147.746}
        self.setupUi(self)

        # log
        self.plainTextEdit.setReadOnly(True)

        # display Baseline Scan
        self.scene_ORLBaseline = GraphicsScene2d()
        self.graphicsView_ORLBaseline.setScene(self.scene_ORLBaseline)
        self.scene_ORLBaseline_2 = GraphicsScene2d()
        self.graphicsView_ORLBaseline_2.setScene(self.scene_ORLBaseline_2)

        # display Scan to register
        self.scene_ORLDifference = GraphicsScene2d()
        self.graphicsView_ORLDifference.setScene(self.scene_ORLDifference)
        self.scene_ORLDifference_2 = GraphicsScene2d()
        self.graphicsView_ORLDifference_2.setScene(self.scene_ORLDifference_2)

        # display Scan registered
        self.scene_ORLFollowUp = GraphicsScene2d()
        self.graphicsView_ORLFollowUp.setScene(self.scene_ORLFollowUp)
        self.scene_ORLFollowUp_2 = GraphicsScene2d()
        self.graphicsView_ORLFollowUp_2.setScene(self.scene_ORLFollowUp_2)

        #Display difference Scan
        self.scene_Diff = GraphicsScene2d()
        self.graphicsView_ORLDifference_3.setScene(self.scene_Diff)

        # process
        self.pushButton_Calculate_Registered_Thickness.clicked.connect(
            self.on_button_clicked_Registered_Thickness)
        self.pushButton_calc_thick.clicked.connect(
            self.on_button_clicked_calc_thickness)
        #self.pushButton_futhickness.clicked.connect(
        #    self.on_button_clicked_futhickness)
        self.pushButton_SaveResults_ORL.clicked.connect(
            self.on_button_clicked_save_results_ORL)

        self.pushButton_load_Baseline.clicked.connect(
            self.on_button_clicked_load_Baseline)
        self.pushButton_load_FollowUp.clicked.connect(
            self.on_button_clicked_load_FollowUp)
        self.radioButton_align1.toggled.connect(self.align1)
        self.radioButton_align2.toggled.connect(self.align2)

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

    def send_and_display_the_log(self, text):
        '''Display the logs
        '''
        time = datetime.now().strftime('%m-%d %H:%M:%S -- ')
        str_send = time + text
        self.plainTextEdit.appendPlainText(str_send)
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

    def align2(self):
        self.pil_image_FollowUp = self.pil_image_FollowUp.rotate(-90)
        self.pil_image_FollowUp = self.pil_image_FollowUp.transpose(
            Image.FLIP_LEFT_RIGHT)
        qimage = QImage(self.pil_image_FollowUp.tobytes(),
                        self.pil_image_FollowUp.width,
                        self.pil_image_FollowUp.height,
                        QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        scene4 = QGraphicsScene(self)
        scene4.addPixmap(pixmap)
        self.graphicsView_ORLFollowUp.setScene(scene4)
        self.graphicsView_ORLFollowUp.fitInView(scene4.sceneRect(),
                                                Qt.KeepAspectRatio)

    def align1(self):
        self.pil_image_Baseline = self.pil_image_Baseline.rotate(-90)
        self.pil_image_Baseline = self.pil_image_Baseline.transpose(
            Image.FLIP_LEFT_RIGHT)
        qimage = QImage(self.pil_image_Baseline.tobytes(),
                        self.pil_image_Baseline.width,
                        self.pil_image_Baseline.height,
                        QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        scene2 = QGraphicsScene(self)
        scene2.addPixmap(pixmap)
        self.graphicsView_ORLBaseline.setScene(scene2)
        self.graphicsView_ORLBaseline.fitInView(scene2.sceneRect(),
                                                Qt.KeepAspectRatio)



    def on_button_clicked_load_FollowUp(self):
        #url = os.getcwd()
        url = self.carryover
        self.send_and_display_the_log("Select follow-up scan")
        file_url_FollowUp = self.openFileNameDialog(url)
        self.send_and_display_the_log("Select follow up scan thickness map")
        file_url_FollowUp_thick = self.openFileNameDialog(file_url_FollowUp)
        self.saveloc = file_url_FollowUp

        if os.path.exists(file_url_FollowUp):
            self.filename_FollowUp = file_url_FollowUp
            self.pil_image_FollowUp = Image.open(file_url_FollowUp)
            qimage = QImage(self.pil_image_FollowUp.tobytes(), self.pil_image_FollowUp.width, self.pil_image_FollowUp.height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)

            scene4 = QGraphicsScene(self)
            scene4.addPixmap(pixmap)
            self.graphicsView_ORLFollowUp.setScene(scene4)
            self.graphicsView_ORLFollowUp.fitInView(scene4.sceneRect(),
                                                      Qt.KeepAspectRatio)
            if os.path.exists(url):
                self.filename_FollowUp_thickness = file_url_FollowUp_thick
                if file_url_FollowUp_thick.lower().endswith(('.tif', '.tiff')):
                    self.actfollowthick = tifffile.imread(
                        file_url_FollowUp_thick)
                    self.pil_image_FollowUp_thick = Image.open(
                        file_url_FollowUp_thick)
                else:
                    self.pil_image_FollowUp_thick = Image.open(
                        file_url_FollowUp_thick)
                    self.actfollowthick = np.array(
                        self.pil_image_FollowUp_thick)


                # Convert to RGB if it's not already
                if self.pil_image_FollowUp_thick.mode != 'RGB':
                    self.pil_image_FollowUp_thick = self.pil_image_FollowUp_thick.convert(
                        'RGB')

                # Convert PIL image to numpy array
                image_array = np.array(self.pil_image_FollowUp_thick)

                height, width, channel = image_array.shape
                bytes_per_line = 3 * width
                qimage_thick = QImage(image_array.data, width, height,
                                      bytes_per_line, QImage.Format_RGB888)

                pixmap_thick = QPixmap.fromImage(qimage_thick)


                scene5 = QGraphicsScene(self)
                scene5.addPixmap(pixmap_thick)
                self.graphicsView_ORLFollowUp_2.setScene(scene5)
                self.graphicsView_ORLFollowUp_2.fitInView(scene5.sceneRect(),
                                                          Qt.KeepAspectRatio)
        else:
            self.send_and_display_the_log('load failed')

    def on_button_clicked_load_Baseline(self):
        url = os.getcwd()
        self.send_and_display_the_log("Select baseline scan")
        file_url = self.openFileNameDialog(url)
        self.carryover = file_url
        self.send_and_display_the_log("Select baseline scan thickness map")
        file_url_thick = self.openFileNameDialog(file_url)
        if os.path.exists(file_url):
            self.filename_Baseline = file_url
            self.pil_image_Baseline = Image.open(file_url)
            qimage = QImage(self.pil_image_Baseline.tobytes(), self.pil_image_Baseline.width, self.pil_image_Baseline.height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)

            scene2 = QGraphicsScene(self)
            scene2.addPixmap(pixmap)
            #self.graphicsView_ORLBaseline.scale(0.7, 0.7)
            self.graphicsView_ORLBaseline.setScene(scene2)
            self.graphicsView_ORLBaseline.fitInView(scene2.sceneRect(),
                                                      Qt.KeepAspectRatio)
            if os.path.exists(url):
                self.filename_Baseline_thickness = file_url_thick
                if file_url_thick.lower().endswith(('.tif', '.tiff')):
                    self.actbasethick = tifffile.imread(
                        file_url_thick)
                    self.pil_image_Baseline_thick = Image.open(
                        file_url_thick)
                else:
                    self.pil_image_Baseline_thick = Image.open(
                        file_url_thick)
                    self.actbasethick = np.array(
                        self.pil_image_Baseline_thick)

                #this is where the input thickness maps are


                # Convert to RGB if it's not already
                if self.pil_image_Baseline_thick.mode != 'RGB':
                    self.pil_image_Baseline_thick = self.pil_image_Baseline_thick.convert(
                        'RGB')

                # Convert PIL image to numpy array
                image_array = np.array(self.pil_image_Baseline_thick)

                height, width, channel = image_array.shape
                bytes_per_line = 3 * width
                qimage_thick = QImage(image_array.data, width, height,
                                      bytes_per_line, QImage.Format_RGB888)

                pixmap_thick= QPixmap.fromImage(qimage_thick)


                scene3 = QGraphicsScene(self)
                scene3.addPixmap(pixmap_thick)
                self.graphicsView_ORLBaseline_2.setScene(scene3)
                self.graphicsView_ORLBaseline_2.fitInView(scene3.sceneRect(),
                                                          Qt.KeepAspectRatio)
        else:
            print('load failed')

    def set_thickness_data_frame(self):
        self.thickness_data = pd.DataFrame(
            columns=['All', '1mm', '3mm', '5mm'],
            index=['With Mask', 'W/O Mask', 'Fovea (X,Y)'])
        # print(self.thickness_data)
    def on_button_clicked_futhickness(self):
        url = os.getcwd()
        self.send_and_display_the_log("Select follw-up scan thickness file (.csv)")
        file_url_csv = self.openFileNameDialog(url)

        if os.path.exists(file_url_csv):
            df = pd.read_csv(file_url_csv)
            self.actual_values = {
                '1mm': df.iloc[1, 2],  # 3C
                '3mm': df.iloc[1, 3],  # 3D
                '5mm': df.iloc[1, 4], # 3E
                'complete': df.iloc[1, 1]  # 3B
            }
            print("hello")
            print(self.actual_values)
        else:
            raise ValueError(
                "Unsuitable format used. Default thickness values applied")

    def on_button_clicked_calc_thickness(self):


        def calculate_thickness_from_map(img_thickness):


            def create_circular_mask_old(h, w, center=None, radius=None):

                x = self.spinBox_Fovea_X.value()
                y = w - self.spinBox_Fovea_Y.value()
                self.fovea_center = (x, y)
                if center is None:  # use the middle of the image
                    center = self.fovea_center if self.fovea_center else (
                        int(w / 2), int(h / 2))
                    #center = (int(w / 2), int(h / 2))
                if radius is None:  # use the smallest distance between the center and image walls
                    radius = min(center[0], center[1], w - center[0],
                                 h - center[1])

                x = np.linspace(0, h - 1, h)
                y = np.linspace(0, w - 1, w)
                X, Y = np.meshgrid(x, y)
                dist_from_center = np.sqrt(
                    (X - center[0]) ** 2 + (Y - center[1]) ** 2)

                mask = dist_from_center <= radius
                return mask
            rownames = self.thickness_data.index
            # print(self.thickness_data)

            # get parameters as input
            for row in range(0, self.thickness_data.shape[0]-1):
                h, w = img_thickness.shape[0], img_thickness.shape[1]
                mask1 = create_circular_mask_old(h, w, radius=42)
                mask2 = create_circular_mask_old(h, w, radius=42 * 3)
                mask3 = create_circular_mask_old(h, w, radius=42 * 5)
                masked_1mm = img_thickness.copy()
                masked_1mm[~mask1] = 0
                masked_3mm = img_thickness.copy()
                masked_3mm[~mask2] = 0
                masked_5mm = img_thickness.copy()
                masked_5mm[~mask3] = 0
                masked_1mm = masked_1mm.astype(float)
                masked_3mm = masked_3mm.astype(float)
                masked_5mm = masked_5mm.astype(float)
                img_thickness = img_thickness.astype(float)
                masked_1mm[masked_1mm == 0] = np.nan
                masked_3mm[masked_3mm == 0] = np.nan
                masked_5mm[masked_5mm == 0] = np.nan
                img_thickness[img_thickness == 0.0] = np.nan
                mean0 = np.nanmean(img_thickness)
                mean1 = np.nanmean(masked_1mm)
                mean2 = np.nanmean(masked_3mm)
                mean3 = np.nanmean(masked_5mm)
                m0 = [0, mean0]
                m1 = [0, mean1]
                m2 = [0, mean2]
                m3 = [0, mean3]

                # Overwrite dataframe with computed thickness numbers
                self.thickness_data.loc[rownames[row]] = float(
                    "{:.6f}".format(m0[row])), float(
                    "{:.6f}".format(m1[row])), float(
                    "{:.6f}".format(m2[row])), float("{:.6f}".format(m3[row]))
                # Apply the scaling factors
                results = {
                    '1mm': mean1,
                    '3mm': mean2,
                    '5mm': mean3,
                    'complete': mean0
                }


            self.send_and_display_the_log("Thickness Computation Complete")
            return results


        try:
            if self.npthick is not None:
                # Ensure dimensions match by adding row/column if needed
                if self.npthick.shape[0] != self.npthick.shape[1]:
                    max_dim = max(self.npthick.shape[0], self.npthick.shape[1])
                    if self.npthick.shape[0] < max_dim:
                        # Add row(s) by duplicating the last row
                        rows_to_add = max_dim - self.npthick.shape[0]
                        self.npthick = np.vstack([self.npthick,
                                                  np.tile(self.npthick[-1:],
                                                          (rows_to_add, 1))])
                    else:
                        # Add column(s) by duplicating the last column
                        cols_to_add = max_dim - self.npthick.shape[1]
                        self.npthick = np.hstack([self.npthick,
                                                  np.tile(self.npthick[:, -1:],
                                                          (1, cols_to_add))])

                #update fovea
                self.thickness_values = calculate_thickness_from_map(self.npthick)
                print(self.thickness_values)
                m11 = round(self.thickness_values['1mm'], 3)
                m31 = round(self.thickness_values['3mm'], 3)
                m51 = round(self.thickness_values['5mm'], 3)
                m71 = round(self.thickness_values['complete'], 3)

                self.lineEdit_1mm.setText(str(m11))
                self.lineEdit_3mm.setText(str(m31))
                self.lineEdit_5mm.setText(str(m51))
                self.lineEdit_All.setText(str(m71))

                #Display the difference maps
                colored_diff, change_diff = diffbetween(self.npthick, self.filename_Baseline_thickness)
                image = Image.fromarray(colored_diff)

                #image = image.filter(ImageFilter.GaussianBlur(radius=1.0))

                # Enhance contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.5)
                self.difsave = image

                # Convert PIL Image to QPixmap via QImage
                #qimg = QImage(image.tobytes(), image.width, image.height,
                #              3 * image.width, QImage.Format_RGB888)

                def create_difference_map_with_colorbar(npthick,
                                                        baseline_thickness):
                    # Calculate the difference
                    colored_dif, actual_dif = diffbetween(npthick, baseline_thickness)

                    # Create a figure with the main image and colorbar
                    fig, ax = plt.subplots(figsize=(6, 5))

                    # Display the difference map
                    reversed_jet = plt.cm.get_cmap('jet').reversed()
                    actual_dif = np.where((actual_dif >= -12) & (actual_dif <= 12), 0, actual_dif)
                    actual_dif[np.isnan(actual_dif)] = 0.0
                    actual_dif[actual_dif > 150] = 0.0

                    im = ax.imshow(-actual_dif, cmap='RdBu', vmin=-200, vmax=200)

                    # Add colorbar
                    plt.colorbar(im, ax=ax, label='Thickness Difference (μm)')

                    # Remove axes for cleaner look
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Save the matplotlib figure to a buffer
                    buf = ioio.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    plt.close()

                    # Convert buffer to PIL Image
                    image = Image.open(buf)

                    # Apply your existing image enhancements
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(2.5)

                    # Store the image for later use
                    self.difsave = image
                    image = image.convert('RGB')

                    # Convert to QPixmap for display
                    qimg = QImage(image.tobytes(), image.width, image.height,
                                  3*image.width, QImage.Format_RGB888)
                    pixmap_dif = QPixmap.fromImage(qimg)

                    return pixmap_dif

                pixmap_dif = create_difference_map_with_colorbar(self.npthick,
                                                                 self.filename_Baseline_thickness)

                scene8 = QGraphicsScene(self)
                scene8.addPixmap(pixmap_dif)
                self.graphicsView_ORLDifference_3.setScene(scene8)
                self.graphicsView_ORLDifference_3.fitInView(scene8.sceneRect(),
                                                            Qt.KeepAspectRatio)
            else:
                raise ValueError(
                    "Thickness Maps do not exist. Please input thickness maps for calculation.")
        except NameError:
            self.send_and_display_the_log(
                'Error in processing thickness differences')

    def on_button_clicked_Registered_Thickness(self):
        if self.radioButton_align1.isChecked() == True:
            al1 = 1
        else:
            al1 = 0
        if self.radioButton_align2.isChecked() == True:
            al2 = 1
        else:
            al2 = 0

        self.nptrans, self.npthick = SITK_Registration(self.filename_Baseline, self.filename_FollowUp, self.filename_Baseline_thickness, self.filename_FollowUp_thickness, al1, al2)

        height, width = self.nptrans.shape[:2]  # Get the first two dimensions
        self.nptrans = self.nptrans.astype(np.uint8)
        bytes_per_line = width
        qimage = QImage(self.nptrans.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        shapy = self.npthick.shape
        channels = 1
        if len(shapy) == 2:
            height1, width1 = shapy
        else:
            height1, width1, channels = shapy

        comb_thick = np.array(self.npthick)
        comb_thick_low = np.uint8(comb_thick)
        if channels == 1:  # If it's actually grayscale

            bytes_per_line1 = width1
            qimage_thick = QImage(comb_thick_low.data, width1, height1,
                                  bytes_per_line1, QImage.Format_Grayscale8)


        elif channels == 3:  # If it's RGB
            bytes_per_line1 = channels * width1
            qimage_thick = QImage(self.npthick.data, width1, height1,
                                  bytes_per_line1, QImage.Format_RGB888)
        elif channels == 4:  # If it's RGBA
            bytes_per_line1 = channels * width1
            qimage_thick = QImage(self.npthick.data, width1, height1,
                                  bytes_per_line1, QImage.Format_RGBA8888)

        else:
            self.send_and_display_the_log(self.npthick.shape)
            raise ValueError(f"Unsupported datatype: {self.npthick.dtype}")

        pixmap_res = QPixmap.fromImage(qimage)
        pixmap_res_thick = QPixmap.fromImage(qimage_thick)

        scene6 = QGraphicsScene(self)
        scene6.addPixmap(pixmap_res)
        self.graphicsView_ORLDifference.setScene(scene6)
        self.graphicsView_ORLDifference.fitInView(scene6.sceneRect(),
                                                  Qt.KeepAspectRatio)

        scene7 = QGraphicsScene(self)
        scene7.addPixmap(pixmap_res_thick)
        self.graphicsView_ORLDifference_2.setScene(scene7)
        self.graphicsView_ORLDifference_2.fitInView(scene7.sceneRect(),
                                                  Qt.KeepAspectRatio)


    def on_button_clicked_save_results_ORL(self):
        """
        save the figrues to the drive
        Returns:
        """
        url = os.path.split(self.saveloc)[0]
        # save images
        imwrite(os.path.join(url, 'Registered_image.png'), (self.nptrans).astype(np.uint8))
        tifffile.imwrite(os.path.join(url, 'Registered_thickness.tif'), self.npthick)
        imwrite(os.path.join(url, 'Difference_thickness.tif'), self.difsave)
        filename = os.path.join(url,'Registration_thickness.csv')
        self.send_and_display_the_log(f"Files saved at location: {url}")
        # save numbers
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(['Measurement', 'Value'])
            # Write data
            for key, value in self.thickness_values.items():
                writer.writerow([key, value])

