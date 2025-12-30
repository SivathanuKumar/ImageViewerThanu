# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:10:24 2019

@author: BAILGPU
"""
import sys
import os
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pandas as pd
from pathlib import Path
import qimage2ndarray
import matplotlib.pyplot as plt
from datetime import datetime
import imageio_ffmpeg as io
#Setup Python Path to find subfolders
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#Import Global State
from Utils.data_class import state
img_obj = state.img_obj
from UI import design_smooth
from Analysis.SM.layer_seg_parallel import seg_video_parallel, OAC_calculation, seg_video_SD_parallel
from Analysis.SM.AlternateSegmentation import quick_process_video
from Analysis.SM.ORL_Segmentation_UNet_SDOCT import ORL_segmentation_UNet_SDOCT
from Analysis.SM.ORL_Segmentation_UNet_reserve2 import ORL_segmentation_UNet_ch
from Analysis.SM.IPL_Segmentation_UNet import IPL_segmentation_UNet_ch
from Analysis.Layers_Wizard.SegWizard import SegWizardWindow
import RemoveTail
import GAsel

#Individual running codes, to call from here.
from Analysis.ORLThickness.ORLThickness import ORLThicknessWindow
from Analysis.CCFD.CCquanWindow import CCquanWindow
from Analysis.Segmentation_Correction.Manual_Correction import SegWindow
from Analysis.Registration_Codes.ORLRegistration_old import ORLdifference

# Environment Settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)


class MainWindow(QtWidgets.QMainWindow, design_smooth.Ui_MainWindow):
    """
    main window
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.img_obj = state.img_obj


        # load data
        self.pushButton_LoadStru.clicked.connect(
            self.on_button_clicked_load_stru)
        self.pushButton_LoadFlow.clicked.connect(
            self.on_button_clicked_load_flow)
        self.pushButton_LoadSeg.clicked.connect(self.on_button_clicked_load_seg)

        self.spinBox_StartLayer.valueChanged.connect(self.get_strat_proj_layer)

        # display button
        self.radioButton_DisplayStru.toggled.connect(self.radio_clicked_stru)
        self.radioButton_DisplayFlow.toggled.connect(self.radio_clicked_flow)
        self.radioButton_DisplayFlattenStru.toggled.connect(
            self.radio_clicked_flatten_stru)
        self.radioButton_DisplayFlattenFlow.toggled.connect(
            self.radio_clicked_flatten_flow)

        # segmentation
        #self.radioButton_Skin.toggled.connect(self.radio_seg_mode_set_range)
        #self.radioButton_ORL.toggled.connect(self.radio_seg_mode_set_range)
        #self.radioButton_Eye.toggled.connect(self.radio_seg_mode_set_range)
        #self.pushButton_AutoSeg.clicked.connect(self.on_button_auto_seg)
        #self.pushButton_QuickSeg.clicked.connect(self.on_button_quick_seg)
        #self.pushButton_SubPixel.clicked.connect(self.on_button_do_subpixel)

        # projeciton
        self.pushButton_AddTable.clicked.connect(
            self.add_porjection_information_to_table)
        self.pushButton_DelTable.clicked.connect(self.delete_row_from_table)
        self.pushButton_Proj.clicked.connect(self.on_button_clicked_proj)

        # import projection
        self.pushButton_ImportSetting.clicked.connect(
            self.on_button_clicked_import_projection_setting)
        self.pushButton_ExportSetting.clicked.connect(
            self.on_button_clicked_export_projection_setting)

        # flatten
        self.pushButton_Flatten.clicked.connect(self.on_button_clicked_flatten)

        # thickness
        self.pushButton_Thickness.clicked.connect(self.display_thickness_map)

        # update
        self.pushButton_UpdateSeg.clicked.connect(self.spinbox_range_set)

        # save data
        self.pushButton_OpenSaveFolder.clicked.connect(
            self.on_button_clicked_open_save_folder)
        self.pushButton_SaveResult.clicked.connect(
            self.on_button_clicked_save_results)

        # log
        self.plainTextEdit_log.setReadOnly(True)

        # scale and alignment
        self.global_scale = 1.0
        # self.horizontalSlider_ScaleSlice.valueChanged.connect
        # (self.on_button_clicked_scale_slice)
        self.pushButton_SliceZoomIn.clicked.connect(
            self.on_button_clicked_scale_slice_zoom_in)
        self.pushButton_SliceZoomOut.clicked.connect(
            self.on_button_clicked_scale_slice_zoom_out)
        self.pushButton_ZoomToFit.clicked.connect(
            self.on_button_clicked_scale_to_fit)
        self.pushButton_volclock.clicked.connect(self.on_button_clicked_volclock)
        self.pushButton_hori.clicked.connect(self.on_button_clicked_horizontal)
        self.pushButton_Vert.clicked.connect(self.on_button_clicked_vertical)

        # proj display
        #self.pushButton_ProjAll.clicked.connect(self.loop_project)
        self.spinBox_ProjGroups.valueChanged.connect(
            self.plot_projs_on_the_scene)

        # flatten
        self.img_flatten_stru = None
        self.img_flatten_flow = None

        # save
        self.img_projs_stru = None
        self.img_projs_flow = None

        # display slab
        self.connect_slab_display_setting()

        self.get_current_path()

        # projeciton
        self.set_data_frame()

        # 1. Initialize Device Types
        device_types = [
            "ZEISS PLEX Elite 9000",
            "ZEISS Cirrus 6000",
            "Intalight DREAM OCT",
            "SD-OCT",
            "SS-OCT"
        ]
        self.comboBox_type.addItems(device_types)

        # 2. Initialize Scan Sizes
        scan_sizes = [
            "6x6 mm",
            "12x12mm",
            "3x3 mm",
            "N/A"
        ]
        self.comboBox_size.addItems(scan_sizes)

        # 3. Connect the Wizard Button
        # Assuming your button in .ui file is named 'pushButton_SegWizard'
        if hasattr(self, 'pushButton_SegWizard'):
            self.pushButton_SegWizard.clicked.connect(self.launch_segmentation_wizard)
        else:
            print("Warning: pushButton_SegWizard not found in UI file")

        # batch
        self.pushButton_BatchOpen.clicked.connect(
            self.on_button_clicked_open_batch)
        self.pushButton_UpdateRule.clicked.connect(
            self.on_button_clicked_open_batch)
        self.pushButton_RunBatch.clicked.connect(
            self.on_button_clicked_run_batch)
        self.pushButton_RunSingle.clicked.connect(
            self.on_button_clicked_run_single)

        # status control
        self.plot_lines = False
        self.display = 'none'

        # other windows
        self.pushButton_ManualCorrect.clicked.connect(
            self.on_button_clicked_manual_segmentation)
        #self.pushButton_Registration.clicked.connect(self.on_button_clicked_registration)
        #self.pushButton_TailRemoval.clicked.connect(self.on_button_clicked_tail)
        self.pushButton_CCquan.clicked.connect(self.on_button_clicked_CCquan)
        self.pushButton_Choroid.clicked.connect(self.on_button_clicked_Choroid)
        #self.pushButton_SegPreview.clicked.connect(self.on_button_clicked_SegPreview)
        self.pushButton_ORL.clicked.connect(
            self.on_button_clicked_ORLThicknessWindow)
        self.pushButton_Difference.clicked.connect(
            self.on_button_clicked_ORLdifference)

    def launch_segmentation_wizard(self):
        """
        Launches the Wizard Window
        """
        # Check if data exists first
        if not self.img_obj.exist_stru:
            self.pop_up_alert("Please load a Structural (Stru) file first.")
            return

        # Get current selection from Main Window Combo Boxes
        selected_device = self.comboBox_type.currentText()
        selected_size = self.comboBox_size.currentText()

        # Create and Show the Wizard
        # We pass 'self' as parent to keep the window on top of Main
        wizard = SegWizardWindow(selected_device, selected_size, parent=self)

        if wizard.exec_() == QtWidgets.QDialog.Accepted:
            # If the user clicked "Run" and it finished successfully:
            self.send_and_display_the_log(f"Wizard Segmentation completed using {selected_device}")
            self.spinbox_range_set()  # Update spinboxes for new layer counts
            self.scroll_bar_reset()  # Refresh view
            self.radioButton_DisplayStru.setChecked(True)  # Force refresh display
            self.plot_lines = True  # Turn on lines

            # Force a refresh of the current slice view
            self.slider_changed(self.horizontalScrollBar_FastScan, 'fast', self.img_obj.stru3d)
    def get_current_path(self):
        """ Get the current path of file, forward it to the lineEdit_OpenPath
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
        elif type == 'seg':
            fileName, _ = QFileDialog.getOpenFileName(None, "Open file", url,
                                                      "Seg Files (*.txt *.mat *.npy)",
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

    def on_button_clicked_load_stru(self):
        '''Load the stru file
        '''
        print('Start loading')
        url = self.lineEdit_OpenPath.text()
        files = self.openFileNameDialog(url, 'video')
        if not files:
            print('cannot load')
            self.pop_up_alert('Load failed')
        else:
            img_obj.read_stru_data(files)
            # img_obj = img_obj.stabilize()
            print(files, ' has been loaded')
            self.send_and_display_the_log(
                'Load structural file: ' + files)  # send the message
            self.pop_up_alert('Load success')

            # --- FIX STARTS HERE ---
            self.radioButton_DisplayStru.setEnabled(True)
            self.radioButton_DisplayStru.setChecked(True)  # Automatically check the box
            self.display = 'stru'  # Explicitly set the mode
            # --- FIX ENDS HERE ---

            self.filename = self.get_file_name(files)
            img_obj.save_path = files

            cwd = str(Path(os.path.dirname(files)))

            self.scroll_bar_reset()
            # rewrite the cwd
            self.lineEdit_OpenPath.setText(cwd)
            self.lineEdit_SavePath.setText(cwd)

    def on_button_clicked_load_flow(self):
        print('Start loading')
        url = self.lineEdit_OpenPath.text()
        files = self.openFileNameDialog(url, 'video')
        if not files:
            print('cannot load')
            self.pop_up_alert('Load failed')
        else:
            img_obj.read_flow_data(files)
            print(files, ' has been loaded')

            self.scroll_bar_reset()
            self.send_and_display_the_log('Load flow file: ' + files)
            self.pop_up_alert('Load success')
            self.radioButton_DisplayFlow.setEnabled(True)
            self.radioButton_DisplayFlow.setChecked(True)
            self.display = 'flow'

            # if filename does not exist
            self.filename = self.get_file_name(files)
            img_obj.save_path = files

    def on_button_clicked_load_seg(self):
        print('Start loading')
        url = self.lineEdit_OpenPath.text()
        files = self.openFileNameDialog(url, 'seg')
        if not files:
            print('cannot load')
            self.pop_up_alert('Load failed')
        else:
            img_obj.read_seg_layers(files)
            print(files, ' has been loaded')
            self.send_and_display_the_log('Load seg file: ' + files)
            # self.display_thickness_map() # disable here
            self.pop_up_alert('Load success')
            self.spinbox_range_set()

    def get_file_name(self, files):
        '''Get the file name without extension and the type 'stru'

        :param file:
        :return:
        '''
        base = os.path.basename(files)
        return os.path.splitext(base)[0]

    def pop_up_alert(self, message='Load success'):
        '''Pop up the message box
        '''
        alert = QMessageBox()
        alert.setText(message)
        alert.exec_()

    def radio_clicked_stru(self, enabled):
        '''Display the stru
        '''
        if enabled:
            if img_obj.exist_stru:
                self.display = 'stru'
                if img_obj.exist_seg:
                    self.plot_lines = True
                self.send_and_display_the_log('Display structural image')
                self.scroll_bar_reset()

    def radio_clicked_flow(self, enabled):
        '''Display the flow
        '''
        if enabled:
            if img_obj.exist_flow:
                self.display = 'flow'
                if img_obj.exist_seg:
                    self.plot_lines = True
                self.send_and_display_the_log('Display flow image')
                self.scroll_bar_reset()

    def radio_clicked_flatten_stru(self, enabled):
        '''Display the flatten stru
        '''
        if enabled:
            try:
                self.img_flatten_stru
                self.display = 'flat_stru'
                self.plot_lines = False
                self.send_and_display_the_log('Display flatten stru image')
                self.scroll_bar_reset()
            except NameError:
                self.send_and_display_the_log(
                    'Flatten stru image does not exist')

    def radio_clicked_flatten_flow(self, enabled):
        '''Display the flatten flow
        '''
        if enabled:
            try:
                self.img_flatten_flow
                self.display = 'flat_flow'
                self.plot_lines = False
                self.send_and_display_the_log('Display flatten flow image')
                self.scroll_bar_reset()
            except NameError:
                self.send_and_display_the_log(
                    'Flatten flow image does not exist')

    def radio_seg_mode_set_range(self):
        """Set the range the of the
        :return:
        """
        if self.radioButton_Skin.isChecked():
            self.spinBox_RetinaUpperOffset.setProperty("value", 50)
            self.spinBox_RetinaLowerOffset.setProperty("value", 1)
        elif self.radioButton_Eye.isChecked():
            self.spinBox_RetinaUpperOffset.setProperty("value", 200)
            self.spinBox_RetinaLowerOffset.setProperty("value", 50)

    def on_button_do_subpixel(self):
        """
        Do the subpixel registration and replace the old volumes
        Returns:
        """
        if img_obj.exist_flow:
            img_obj.stru3d, img_obj.flow3d, errors = img_obj.sub_pixel_regi(
                img_obj.stru3d, img_obj.flow3d)
        else:
            img_obj.stru3d, errors = img_obj.sub_pixel_regi(img_obj.stru3d)

    def on_button_quick_seg(self):
        '''Perform the quick segmentaion on the structural data
        '''
        try:
            if img_obj.exist_stru:
                img_obj.layers = quick_process_video(img_obj)
                img_obj.layer_num = img_obj.layers.shape[2]
                img_obj.exist_seg = True
                print('Quick Segmentation done')
                self.send_and_display_the_log('Quick Segmentation done')
                self.spinbox_range_set()
        except Exception as e:
            self.send_and_display_the_log('Quick Segmentation Error')
            print(e)

    '''
    def on_button_auto_seg(self):
        
        try:
            upper_offset = self.spinBox_RetinaUpperOffset.value()
            lower_offset = self.spinBox_RetinaLowerOffset.value()
            depth_start, depth_end = img_obj._get_auto_range()

            paras = {'retina_lower_offset': lower_offset,
                     'retina_upper_offset': upper_offset,
                     'upper_bound': depth_start, 'lower_bound': depth_end}
            
            if self.radioButton_ORL.isChecked():
                Y_orl, X_orl, Z_orl = img_obj.stru3d.shape
                layers_final = np.zeros((X_orl, Z_orl, 4))

                self.send_and_display_the_log(
                    'ORL Segmentation processing. Will take some time')

                if self.checkBox_SDOCTORL.isChecked():
                    orl = ORL_segmentation_UNet_SDOCT(img_obj, paras)
                    layers_final[:, :, 0] = orl[:, :, 0]
                    layers_final[:, :, 1] = orl[:, :, 1]
                    img_obj.layers = orl
                    img_obj.layer_num = img_obj.layers.shape[2]
                    for i in range(0, img_obj.layers.shape[1] - 1):
                        img_obj.layers[:, i, 0] = img_obj.layers[:, i, 0] + 80
                        img_obj.layers[:, i, 1] = img_obj.layers[:, i, 1] + 90

                    print("Segmentation done for SDOCT")
                    img_obj.exist_seg = True
                else:

                    orl = ORL_segmentation_UNet_ch(img_obj, paras)

                    # sending to segmentation
                    if self.checkBox_IPL.isChecked():
                        layers_up = IPL_segmentation_UNet_ch(img_obj, paras)
                        img_obj.layers = layers_up
                        layers_final[:, :, 0] = orl[:, :, 0]
                        layers_final[:, :, 1] = orl[:, :, 1]
                        layers_final[:, :, 2] = layers_up[:, :, 0]
                        layers_final[:, :, 3] = layers_up[:, :, 1]
                        img_obj.layers = layers_final
                        img_obj.layer_num = img_obj.layers.shape[2]
                        print("IPL segmented")
                    else:
                        img_obj.layers = orl
                        img_obj.layer_num = img_obj.layers.shape[2]

                    img_obj.exist_seg = True

            elif self.radioButton_Skin.isChecked():
                # img_obj.auto_seg_stru(auto_range=True, retina_lower_offset=lower_offset,
                #                       retina_upper_offset=upper_offset, mode='skin')
                img_obj.auto_seg_stru(auto_range=True,
                                      retina_lower_offset=lower_offset,
                                      retina_upper_offset=upper_offset,
                                      mode='eye', better_BM=True)
            elif self.radioButton_SDOCT.isChecked():
                img_obj.layers = seg_video_SD_parallel(img_obj,
                                                       retina_th=lower_offset)
                img_obj.layer_num = img_obj.layers.shape[2]
                img_obj.exist_seg = True

            print('segmentation done')
            self.send_and_display_the_log('Segmentation done')
            self.spinbox_range_set()
        except Exception as e:
            self.send_and_display_the_log('Segmentation Error')
            print(e)
            

    '''
    def spinbox_range_set(self):
        '''Set the range of spinbox to fit the image
        '''
        if img_obj.exist_seg:
            layer_num = img_obj.layers.shape[2] - 1
            self.spinBox_StartLayer.setMaximum(layer_num)
            self.spinBox_EndLayer.setMaximum(layer_num)
            self.spinBox_FlattenLayer.setMaximum(layer_num)
            self.spinBox_EndLayer.setMaximum(layer_num)
            self.spinBox_ThicknessStart.setMaximum(layer_num)
            self.spinBox_ThicknessEnd.setMaximum(layer_num)

    def scroll_bar_reset(self):
        '''Reset the maximum range and the first and
        '''
        try:
            imgs = None  # Initialize as None to prevent "UnboundLocalError"

            # Determine which data to show based on self.display state
            if self.display == 'stru':
                if img_obj.exist_stru:
                    imgs = img_obj.stru3d
            elif self.display == 'flow':
                if img_obj.exist_flow:
                    imgs = img_obj.flow3d
            elif self.display == 'flat_stru':
                imgs = self.img_flatten_stru
            elif self.display == 'flat_flow':
                imgs = self.img_flatten_flow

            # AUTO-RECOVERY:
            # If display is 'none' but we have data, force a default view
            if imgs is None:
                if img_obj.exist_stru:
                    self.display = 'stru'
                    imgs = img_obj.stru3d
                    self.radioButton_DisplayStru.setChecked(True)
                elif img_obj.exist_flow:
                    self.display = 'flow'
                    imgs = img_obj.flow3d
                    self.radioButton_DisplayFlow.setChecked(True)

            # GUARD CLAUSE: If we still have no images, stop here.
            if imgs is None:
                return

            # Proceed only if we have valid images
            self.scroll_bar_moved(self.horizontalScrollBar_FastScan, 'fast',
                                  imgs)
            self.scroll_bar_moved(self.horizontalScrollBar_SlowScan, 'slow',
                                  imgs)
            self.scroll_bar_moved(self.horizontalScrollBar_DepthScan, 'depth',
                                  imgs)

            self.scroll_bar_set_range(imgs)
        except Exception as e:
            print(f"Error in scroll_bar_reset: {e}")

    def scroll_bar_set_range(self, imgs):
        '''set the range of scroll bar based on images
        '''
        self.horizontalScrollBar_FastScan.setMaximum(imgs.shape[2] - 1)
        self.horizontalScrollBar_SlowScan.setMaximum(imgs.shape[1] - 1)
        self.horizontalScrollBar_DepthScan.setMaximum(imgs.shape[0] - 1)

    def scroll_bar_moved(self, scrollbar, view, imgs):
        # scrollbar.sliderMoved.connect(self.slider_changed)
        # scrollbar.valueChanged.connect(self.slider_changed)
        # self.slider_changed(scrollbar, view, imgs)  # instance display
        # connect the signal
        self.slider_changed(scrollbar, view, imgs)
        # scrollbar.sliderMoved.connect(lambda: self.slider_changed(scrollbar, view, imgs))

        # disconnect previous signal
        if scrollbar.receivers(scrollbar.valueChanged) > 0:
            scrollbar.valueChanged.disconnect()
        scrollbar.valueChanged.connect(
            lambda: self.slider_changed(scrollbar, view, imgs))

    def slider_changed(self, scrollbar, view, imgs):
        # frame = self.horizontalScrollBar_FastScan.value()
        frame = scrollbar.value()
        if img_obj.exist_stru:
            if view == 'fast':
                img = imgs[:, :, frame]

                if self.plot_lines:
                    lines = np.squeeze(img_obj.layers[:, frame, :])
                    self.plot_on_the_scene(self.graphicsView_FastScan, img,
                                           add_line=True, add_slab_lines=True,
                                           lines=lines)
                else:
                    self.plot_on_the_scene(self.graphicsView_FastScan, img)
                self.lcdNumber_FastScan.display(frame)
            elif view == 'slow':
                img = imgs[:, frame, :]
                if self.plot_lines:
                    lines = np.squeeze(img_obj.layers[frame, :, :])
                    self.plot_on_the_scene(self.graphicsView_SlowScan, img,
                                           add_line=True, lines=lines)
                else:
                    self.plot_on_the_scene(self.graphicsView_SlowScan, img)
                self.lcdNumber_SlowScan.display(frame)
            elif view == 'depth':
                img = imgs[frame, :, :]
                self.plot_on_the_scene(self.graphicsView_DepthScan, img)
                self.lcdNumber_DepthScan.display(frame)

    def get_status_of_projection_setting(self):
        """ Get the parameters of projection from the GUI input
        Return:
            one group of projection parameters
        """
        start_layer = self.get_strat_proj_layer()
        end_layer = self.get_end_proj_layer()
        start_offset = self.get_strat_proj_offset()
        end_offset = self.get_end_proj_offset()
        proj_method = self.comboBox_ProjMethod.currentText()
        is_proj_stru = self.checkBox_ProjStru.isChecked()
        is_proj_flow = self.checkBox_ProjFlow.isChecked()
        return start_layer, end_layer, start_offset, end_offset, proj_method, is_proj_stru, is_proj_flow

    def get_strat_proj_layer(self):
        """ get the start layer
        """
        return self.spinBox_StartLayer.value()

    def get_end_proj_layer(self):
        """ Get the end layer
        """
        return self.spinBox_EndLayer.value()

    def get_strat_proj_offset(self):
        '''Get the offsets of the starting layer
        '''
        return self.spinBox_StartOffset.value()

    def get_end_proj_offset(self):
        '''Get the offsets of the starting layer
        '''
        return self.spinBox_EndOffset.value()

    def connect_slab_display_setting(self):
        """Display the line the when value change
        """
        self.spinBox_StartLayer.valueChanged.connect(self.scroll_bar_reset)
        self.spinBox_EndLayer.valueChanged.connect(self.scroll_bar_reset)
        self.spinBox_StartOffset.valueChanged.connect(self.scroll_bar_reset)
        self.spinBox_EndOffset.valueChanged.connect(self.scroll_bar_reset)

    def set_data_frame(self):
        self.proj_data = pd.DataFrame(
            columns=['start_layer', 'end_layer', 'start_offset', 'end_offset',
                     'proj_method', 'is_proj_stru', 'is_proj_flow'])

    def add_porjection_information_to_table(self):
        '''
        '''
        table = self.tableWidget
        row = table.rowCount()

        # get parameters as input
        start_layer, end_layer, start_offset, end_offset, proj_method, is_proj_stru, is_proj_flow = self.get_status_of_projection_setting()

        # display on the table widget
        self.add_row_on_table(table, text='Proj ' + str(row))
        self.set_item_on_table(table, row, 0, str(start_layer))
        self.set_item_on_table(table, row, 1, str(end_layer))
        self.set_item_on_table(table, row, 2, str(start_offset))
        self.set_item_on_table(table, row, 3, str(end_offset))
        self.set_item_on_table(table, row, 4, proj_method)
        self.set_item_on_table(table, row, 5, str(is_proj_stru))
        self.set_item_on_table(table, row, 6, str(is_proj_flow))

        # add the information to the dataframe\
        self.proj_data = self.proj_data._append({'start_layer': start_layer,
                                                'end_layer': end_layer,
                                                'start_offset': start_offset,
                                                'end_offset': end_offset,
                                                'proj_method': proj_method,
                                                'is_proj_stru': is_proj_stru,
                                                'is_proj_flow': is_proj_flow},
                                               ignore_index=True)

    def on_button_clicked_import_projection_setting(self):
        """ Load projection setting from outside files
        """
        # open folder
        url = self.lineEdit_OpenPath.text()
        try:
            url = self.open_folder(url)
        except Exception as e:
            print(e)

        # read the projection settings
        try:
            file_name = os.path.join(url, 'proj_settings.csv')
            self.proj_data = pd.read_csv(file_name)
            self.proj_data.drop(['Unnamed: 0'], inplace=True, axis=1,
                                errors='ignore')
            self.display_projection_to_table()
        except Exception as e:
            print(e)
            self.send_and_display_the_log('Cannot open the projection settings')

    def on_button_clicked_export_projection_setting(self):
        """ Export the projection to the folder
        """
        # save the setting to current folder
        url = self.lineEdit_OpenPath.text()
        try:
            url = self.open_folder(url)
            self.proj_data.to_csv(os.path.join(url, 'proj_settings.csv'))
        except Exception as e:
            print(e)
            self.send_and_display_the_log(
                'Cannot export the projection settings')

    def display_projection_to_table(self):

        table = self.tableWidget
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(0)

        # get parameters as input
        for row in range(0, self.proj_data.shape[0]):
            start_layer, end_layer, start_offset, end_offset, proj_method, is_proj_stru, is_proj_flow = self.get_porjection_information_from_table(
                row)

            # display on the table widget
            self.add_row_on_table(table, text='Proj ' + str(row))
            self.set_item_on_table(table, row, 0, str(start_layer))
            self.set_item_on_table(table, row, 1, str(end_layer))
            self.set_item_on_table(table, row, 2, str(start_offset))
            self.set_item_on_table(table, row, 3, str(end_offset))
            self.set_item_on_table(table, row, 4, proj_method)
            self.set_item_on_table(table, row, 5, str(is_proj_stru))
            self.set_item_on_table(table, row, 6, str(is_proj_flow))

    def get_porjection_information_from_table(self, row):
        '''Get the information from dataframe
        '''
        return self.proj_data.iloc[row, 0], self.proj_data.iloc[row, 1], \
        self.proj_data.iloc[row, 2], \
            self.proj_data.iloc[row, 3], self.proj_data.iloc[row, 4], \
        self.proj_data.iloc[row, 5], \
            self.proj_data.iloc[row, 6]

    def add_row_on_table(self, table, text='ProjNew'):
        '''Add a row below current table widget
        '''
        rowPosition = table.rowCount()
        table.insertRow(rowPosition)
        table.setVerticalHeaderItem(rowPosition, QTableWidgetItem(text))

    def delete_row_from_table(self):
        """ delete the selected row from the tablewidget and the self.proj_data
        """
        try:
            table = self.tableWidget
            if table.rowCount() > 0:
                table.removeRow(table.currentRow())

                # also del a line from dataframe
                self.proj_data = self.proj_data.drop(
                    self.proj_data.index[table.currentRow()])

        except Exception as e:
            print(e)

    def set_item_on_table(self, table, row, column, text):
        '''set a item on table widget
        '''
        table.setItem(row, column, QTableWidgetItem(text))

    def get_projections(self, row_num):
        ''' Get the projection for saving
        '''

        start_layer, end_layer, start_offset, end_offset, proj_method, is_proj_stru, is_proj_flow = self.get_porjection_information_from_table(
            row_num)

        # for eye image, rot90 and flip
        if self.radioButton_Skin.isChecked():
            rot_status = False
        elif self.radioButton_SDOCT.isChecked():
            rot_status = True
        elif self.radioButton_ORL.isChecked():
            rot_status = False
        elif self.radioButton_Eye.isChecked():
            rot_status = True
        else:
            rot_status = False

        # plot the image
        if is_proj_stru:
            img_proj_stru = img_obj.plot_proj(start_layer, end_layer, 'stru',
                                              proj_method,
                                              start_offset=start_offset,
                                              end_offset=end_offset,
                                              display=False, enhance=False,
                                              rotate=rot_status)
            img_proj_stru = self.transfer_image_range_to_uint8(img_proj_stru)
        else:
            img_proj_stru = None

        if is_proj_flow:
            img_proj_flow = img_obj.plot_proj(start_layer, end_layer, 'flow',
                                              proj_method,
                                              start_offset=start_offset,
                                              end_offset=end_offset,
                                              display=False, enhance=False,
                                              rotate=rot_status)
            img_proj_flow = self.transfer_image_range_to_uint8(img_proj_flow)
        else:
            img_proj_flow = None

        # except Exception as e:
        #    self.send_and_display_the_log('Projeciton error on: ' + str(row_num))
        #   print(e)

        return img_proj_stru, img_proj_flow

    def on_button_clicked_proj(self):
        '''preview the enface projection when click the preview button
        '''
        # get parameters
        try:
            # push to the table and dataframe

            start_layer, end_layer, start_offset, end_offset, proj_method, is_proj_stru, is_proj_flow = self.get_status_of_projection_setting()
            print('projeciton method: ', proj_method)

            # set parameters
            if self.radioButton_Skin.isChecked():
                rot_status = False
            elif self.radioButton_SDOCT.isChecked():
                rot_status = True
            elif self.radioButton_Eye.isChecked():
                rot_status = True
            elif self.radioButton_ORL.isChecked():
                rot_status = False
            else:
                rot_status = False

            # plot the image
            if is_proj_stru:
                self.img_proj_stru = img_obj.plot_proj(start_layer, end_layer,
                                                       'stru', proj_method,
                                                       start_offset=start_offset,
                                                       end_offset=end_offset,
                                                       display=False,
                                                       rotate=rot_status)
                self.img_proj_stru = self.transfer_image_range_to_uint8(
                    self.img_proj_stru)
                self.plot_on_the_scene(self.graphicsView_ProjectionStru,
                                       self.img_proj_stru)
                self.send_and_display_the_log('Plot the stru projeciton')

            if is_proj_flow:
                self.img_proj_flow = img_obj.plot_proj(start_layer, end_layer,
                                                       'flow', proj_method,
                                                       start_offset=start_offset,
                                                       end_offset=end_offset,
                                                       display=False,
                                                       rotate=rot_status)
                self.img_proj_flow = self.transfer_image_range_to_uint8(
                    self.img_proj_flow)
                self.plot_on_the_scene(self.graphicsView_ProjectionFlow,
                                       self.img_proj_flow)
                self.send_and_display_the_log('Plot the flow projeciton')

            print('Plot the enface projection')

            # plot the slab
            # self.plot_slab()
        except Exception as e:
            self.send_and_display_the_log('Projeciton error!')
            print(e)

        # send the paramters to the table

        # plot the thickness map
        try:
            self.img_thickness = img_obj.thickness_map(
                self.spinBox_ThicknessStart.value(),
                self.spinBox_ThicknessEnd.value(), smooth=True)
            self.plot_on_the_scene(self.graphicsView_Thickness,
                                   self.img_thickness)
            print('Plot the thickness map')
        except Exception as e:
            print(e)

    def loop_project(self):
        '''Project all the enface layers, save them to self.projs
        '''
        # print(self.proj_data.head())
        # print(self.proj_data.shape)
        # self.proj_data.to_csv('./Dicom/proj.csv')
        # try:
        # init

        if self.radioButton_Eye.isChecked() or self.radioButton_SDOCT.isChecked():
            self.projs_stru = np.zeros((img_obj.img_framenum, img_obj.img_width,
                                        self.proj_data.shape[0]))
            self.projs_flow = np.zeros((img_obj.img_framenum, img_obj.img_width,
                                        self.proj_data.shape[0]))
        else:
            self.projs_stru = np.zeros((img_obj.img_width, img_obj.img_framenum,
                                        self.proj_data.shape[0]))
            self.projs_flow = np.zeros((img_obj.img_width, img_obj.img_framenum,
                                        self.proj_data.shape[0]))

        # loop projection
        for i in range(0, self.proj_data.shape[0]):
            img_proj_stru, img_proj_flow = self.get_projections(i)
            self.projs_stru[:, :, i] = np.copy(img_proj_stru)
            self.projs_flow[:, :, i] = np.copy(img_proj_flow)
            self.send_and_display_the_log('Projeciton group num: ' + str(i))
            # except Exception as e:
        #    print(e)
        self.spinBox_ProjGroups.setMaximum(self.proj_data.shape[0] - 1)

    def plot_projs_on_the_scene(self):
        """ Plot the preview en-face projection to the scene when change the value of spinbox
        """
        try:
            ind = self.spinBox_ProjGroups.value()
            self.plot_on_the_scene(self.graphicsView_ProjectionStru,
                                   self.transfer_image_range_to_uint8(
                                       self.projs_stru[:, :, ind]))
            self.plot_on_the_scene(self.graphicsView_ProjectionFlow,
                                   self.transfer_image_range_to_uint8(
                                       self.projs_flow[:, :, ind]))
            self.send_and_display_the_log(
                'Display the projection group: ' + str(ind))
        except Exception as e:
            print(e)
            self.send_and_display_the_log(
                'Cannot display the projection group: ' + str(ind))

    def transfer_image_range_to_uint8(self, img):
        """ Transfer the image range from uint 16 or double to uint8[0-255] for display
        Args:
            img: numpy array
        Return:
             img_out: numpy array as type uint8 within range [0, 255]
        """
        img_out = img / img.max()
        img_out = np.uint8(img_out * 255)
        img_out[img_out < 0] = 0
        img_out[img_out == -np.inf] = 0
        img_out[img_out > 255] = 255
        return img_out

        # if np.issubdtype(type(img), np.uint8):
        #    img_out = img
        # if np.issubdtype(type(img), np.uint16):
        #    img_out = np.unit8(img/256)

    def plot_on_the_scene(self, graphicsView, img, add_line=False,
                          add_slab_lines=False, lines=None):
        '''Plot the image to the graphicesView
        Args:
            graphicsView:
            img: prefer 8 bit file, will transfer uint16 data to uint8
            add_line: if add the segmentation line to the scene
            add_slab_lines: if add the slab lines to the scene
            lines: segmentation lines
        '''
        # if the data is uint 16 type
        if np.issubdtype(img.dtype, np.uint16):
            img = np.uint8(img / 256.0)

        # draw a new scene
        scene = QGraphicsScene()
        img_pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(img))
        scene.addPixmap(img_pixmap)
        if add_line:
            scene = self.plot_add_lines(scene, lines)
        if add_slab_lines:
            lines = self.get_slab_lines()
            scene = self.plot_add_lines(scene, lines, QPen(Qt.green, 1))

        # put the scene to the graphicview
        graphicsView.setScene(scene)

        # set scale to fit the view
        graphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
        graphicsView.scale(self.global_scale, self.global_scale)
        # scale = graphicsView.size().width()/ img_pixmap.size().width()
        # graphicsView.scale(scale, scale)

    def on_button_clicked_volclock(self):
        '''Get the flatten images
        '''

        # stru
        if (img_obj.exist_stru):
            strurot = np.rot90(img_obj.stru3d, k=-1, axes=(0, 1))
            img_obj.stru3d = strurot
            tempo = img_obj.img_width
            img_obj.img_width = img_obj.img_depth
            img_obj.img_depth = tempo
            self.send_and_display_the_log('Struct Clockwise Rotated')
            self.display = 'stru'
            if img_obj.exist_seg:
                self.plot_lines = True
            self.send_and_display_the_log('Updated structural image')
            self.scroll_bar_reset()

        if (img_obj.exist_flow):
            flowrot = np.rot90(img_obj.flow3d, k=-1, axes=(0, 1))
            img_obj.flow3d = flowrot
            self.send_and_display_the_log('Flow Clockwise Rotated')
            self.display = 'flow'
            if img_obj.exist_seg:
                self.plot_lines = True
            self.send_and_display_the_log('Updated flow image')
            self.scroll_bar_reset()

    def on_button_clicked_horizontal(self):
        '''Get the flatten images
        '''

        # stru
        if (img_obj.exist_stru):
            strurot = np.fliplr(img_obj.stru3d)
            img_obj.stru3d = strurot
            self.send_and_display_the_log('Struct Clockwise Rotated')
            self.display = 'stru'
            if img_obj.exist_seg:
                self.plot_lines = True
            self.send_and_display_the_log('Updated structural image')
            self.scroll_bar_reset()

        if (img_obj.exist_flow):
            flowrot = np.fliplr(img_obj.flow3d)
            img_obj.flow3d = flowrot
            self.send_and_display_the_log('Flow Clockwise Rotated')
            self.display = 'flow'
            if img_obj.exist_seg:
                self.plot_lines = True
            self.send_and_display_the_log('Updated flow image')
            self.scroll_bar_reset()

    def on_button_clicked_vertical(self):
        # stru
        if (img_obj.exist_stru):
            strurot = np.flipud(img_obj.stru3d)
            img_obj.stru3d = strurot
            self.send_and_display_the_log('Struct Clockwise Rotated')
            self.display = 'stru'
            if img_obj.exist_seg:
                self.plot_lines = True
            self.send_and_display_the_log('Updated structural image')
            self.scroll_bar_reset()

        if (img_obj.exist_flow):
            flowrot = np.flipud(img_obj.flow3d)
            img_obj.flow3d = flowrot
            self.send_and_display_the_log('Flow Clockwise Rotated')
            self.display = 'flow'
            if img_obj.exist_seg:
                self.plot_lines = True
            self.send_and_display_the_log('Updated flow image')
            self.scroll_bar_reset()

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
        scale = 1 / self.global_scale
        self.global_scale = 1
        self.update_scale_of_view(scale)

    def update_scale_of_view(self, scale):
        ''' Updates all graphics view
        '''
        self.graphicsView_FastScan.scale(scale, scale)
        self.graphicsView_SlowScan.scale(scale, scale)
        self.graphicsView_DepthScan.scale(scale, scale)
        # self.graphicsView_SlabSelection.scale(scale, scale)
        self.graphicsView_Thickness.scale(scale, scale)
        self.graphicsView_ProjectionFlow.scale(scale, scale)
        self.graphicsView_ProjectionStru.scale(scale, scale)

    def get_slab_lines(self):
        """ Get the lines of current slab
        """
        # get parameters
        frame = self.horizontalScrollBar_FastScan.value()
        start_layer = self.get_strat_proj_layer()
        end_layer = self.get_end_proj_layer()
        start_offset = self.get_strat_proj_offset()
        end_offset = self.get_end_proj_offset()

        # Get the lines
        line_up = img_obj.layers[:, frame, start_layer].astype(int) + start_offset
        line_low = img_obj.layers[:, frame, end_layer].astype(int) + end_offset
        lines = np.array([line_up, line_low])
        return lines.T

    def plot_add_lines(self, scene, lines, pen=QPen(Qt.red, 1)):
        '''Add lines on the scene
        Args:
            lines: [img_width, number_of_lines]
        '''

        img_w = lines.shape[0]
        for i in range(0, lines.shape[1]):
            # define the pen here
            # pen = QPen(Qt.red, 1)
            for j in range(0, img_w - 1):
                scene.addLine(j, lines[j, i], j + 1, lines[j + 1, i], pen)

        return scene

        # painter = QPainter(self)
        # pixmap = QPixmap("myPic.png")
        # painter.drawPixmap(self.rect(), pixmap)
        # pen = QPen(Qt.red, 3)
        # painter.setPen(pen)
        # painter.drawLine(10, 10, self.rect().width() -10 , 10)

    def plot_slab(self, frame=10):
        '''Plot the slab on the scene (not use)
        '''
        # get parameters
        start_layer = self.get_strat_proj_layer()
        end_layer = self.get_end_proj_layer()
        start_offset = self.get_strat_proj_offset()
        end_offset = self.get_end_proj_offset()

        # plot the lines on the canvas
        line_up = img_obj.layers[:, frame, start_layer] + start_offset
        line_low = img_obj.layers[:, frame, end_layer] + end_offset
        fig = plt.figure()
        plt.imshow(img_obj.stru3d[:, :, frame],
                   cmap='gray')  # should change here
        plt.plot(line_up, '--', linewidth=2)
        plt.plot(line_low, '--', linewidth=2)
        img = self.canvas_to_array(plt.gcf())
        plt.close(fig)
        # self.plot_on_the_scene(self.graphicsView_SlabSelection, img)

    def canvas_to_array(self, fig):
        '''Convert the plt canvas to array
        '''
        # fig = plt.gcf()
        fig.canvas.draw()
        figData = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        figHandleSize = fig.canvas.get_width_height()[::-1] + (3,)
        fData = figData.reshape(figHandleSize)
        return fData

    def on_button_clicked_flatten(self):
        '''Get the flatten images
        '''
        ref_layer = self.spinBox_FlattenLayer.value()

        new_loc = img_obj.flatten_loc
        is_processed = False
        # stru
        if (img_obj.exist_stru):
            self.img_flatten_stru = img_obj.save_flatten_video(
                video_type='stru', ref_layer_num=ref_layer, new_loc=new_loc,
                saved=False)
            self.send_and_display_the_log(
                'Flatten the stru image')  # send the message
            self.radioButton_DisplayFlattenStru.setEnabled(True)
            is_processed = True

        # flow
        if (img_obj.exist_flow):
            self.img_flatten_flow = img_obj.save_flatten_video(
                video_type='flow', ref_layer_num=ref_layer, new_loc=new_loc,
                saved=False)
            self.send_and_display_the_log(
                'Flatten the flow image')  # send the message
            self.radioButton_DisplayFlattenFlow.setEnabled(True)
            is_processed = True
        if not is_processed:
            self.send_and_display_the_log('Error: Neither Stru nor Flow data was found.')
            #

    def display_thickness_map(self):
        '''Display the thicknessmap
        '''
        start = self.spinBox_ThicknessStart.value()
        end = self.spinBox_ThicknessEnd.value()
        if end >= start:
            thicknessmap = img_obj.thickness_map(start, end)
            # plot
            fig = plt.figure()
            # ax = plt.Axes(fig, [0., 0., 1., 1.])
            # ax.set_axis_off()
            # fig.add_axes(ax)
            # ax.imshow(thicknessmap, aspect='auto')
            plt.imshow(thicknessmap, cmap='jet')
            plt.colorbar()
            img = self.canvas_to_array(plt.gcf())
            plt.close(fig)
            self.send_and_display_the_log(
                'Display the thickness map')  # send the message
            self.plot_on_the_scene(self.graphicsView_Thickness, img)

        else:
            self.send_and_display_the_log('Please input correct number')

        mean_thickness = img_obj.get_mean_thickness(start, end)
        thick_msg = 'Mean Thickness = ' + str(
            '%0.4f' % mean_thickness) + ' pixels'
        self.lineEdit_MeanThickness.setText(thick_msg)

    def send_and_display_the_log(self, text):
        '''Display the logs
        '''
        time = datetime.now().strftime('%m-%d %H:%M:%S -- ')
        str_send = time + text
        self.plainTextEdit_log.appendPlainText(str_send)

    # %% saving

    def on_button_clicked_open_save_folder(self):
        '''open the folder for saving
        '''
        url = self.open_folder()
        self.lineEdit_SavePath.setText(url)
        print(url)

    def on_button_clicked_save_results(self):
        '''Save the results
        '''
        # get the path
        url = self.lineEdit_SavePath.text()

        try:
            url = os.path.join(url, self.filename)
        except Exception as e:
            print(e)
        print('save the result: ', url)

        if not os.path.exists(url):
            os.makedirs(url)

        try:
            if self.checkBox_SaveFlatten.isChecked():
                self.save_flatten(url)
                print('Flatten video saved')
                self.send_and_display_the_log(
                    'Flatten video saved')  # send the message

            if self.checkBox_SaveOriVideo.isChecked():
                self.save_ori_video_dcm(url)
                print('Original video saved')
                self.send_and_display_the_log(
                    'original video saved')  # send the message

            if self.checkBox_SaveProjections.isChecked():
                self.save_projections(url)
                print('projection images saved')
                self.send_and_display_the_log(
                    'projection images saved')  # send the message

            if self.checkBox_SaveSeg.isChecked():
                self.save_seg(url)
                print('Segmentation files saved')
                self.send_and_display_the_log(
                    'Segmentation files saved')  # send the message

            if self.checkBox_SaveSegVideo.isChecked():
                self.save_seg_video(url)
                print('Segmentation video saved')
                self.send_and_display_the_log(
                    'Segmentation video saved')  # send the message

            if self.checkBox_LayersVolume.isChecked():
                self.save_layer_volumes(url)
                print('Segmentation volumes saved')
                self.send_and_display_the_log(
                    'Segmentation volumes saved')  # send the message

            if self.checkBox_AviVideo.isChecked():
                self.save_ori_video(url)
                self.send_and_display_the_log(
                    'Original video saved')  # send the message


            if self.checkBox_SaveOAC.isChecked():
                self.save_OAC_video(url)
                self.send_and_display_the_log(
                    'OAC volumes saved')  # send the message

        except Exception as e:
            self.send_and_display_the_log('Cannot save files')
            print(e)

    def save_flatten(self, url):
        '''save the flatten
        '''
        ref_layer = self.spinBox_FlattenLayer.value()
        new_loc = img_obj.flatten_loc

        if ((self.checkBox_FlattenStru.isChecked()) & (img_obj.exist_stru)):
            file_name = os.path.join(url,
                                     'stru_flatten.dcm')  # send the message
            self.img_flatten_stru = img_obj.save_flatten_video(
                file_name=file_name, video_type='stru', new_loc=new_loc,
                ref_layer_num=ref_layer, saved=True)
            self.send_and_display_the_log('Flatten the stru image')
            self.radioButton_DisplayFlattenStru.setEnabled(True)

        if ((self.checkBox_FlattenFlow.isChecked()) & (img_obj.exist_flow)):
            file_name = os.path.join(url, 'flow_flatten.dcm')

            self.img_flatten_flow = img_obj.save_flatten_video(
                file_name=file_name, video_type='flow', new_loc=new_loc,
                ref_layer_num=ref_layer, saved=True)
            self.send_and_display_the_log(
                'Flatten the flow image')  # send the message
            self.radioButton_DisplayFlattenFlow.setEnabled(True)

    def save_ori_video_dcm(self, url):
        """
        :param url:
        :return:
        """
        if img_obj.exist_stru:
            file_name = os.path.join(url, 'ori_video_stru.dcm')
            img_obj.save_video(img_obj.stru3d, file_name)

        if img_obj.exist_flow:
            file_name = os.path.join(url, 'ori_video_flow.dcm')
            img_obj.save_video(img_obj.flow3d, file_name)



    def save_projections(self, url):
        '''save the projections
        '''
        '''
        if self.checkBox_ProjStru.isChecked():
            file_name = os.path.join(url, 'projs_stru.png')
            imwrite(file_name, self.img_proj_stru)
        if self.checkBox_ProjFlow.isChecked():
            file_name = os.path.join(url, 'projs_flow.png')
            imwrite(file_name, self.img_proj_flow)
        '''
        # loop project

        try:
            self.loop_project()

            def norm_c(image):
                image = np.nan_to_num(image)
                image_norm = (image - np.min(image)) / (
                        np.max(image) - np.min(image))
                return np.uint8(image_norm * 255)

            for i in range(0, self.proj_data.shape[0]):
                file_name = os.path.join(url, 'projs_stru_' + str(i) + '.png')
                if not (np.sum(~np.isnan(self.projs_stru[:, :, i])) == 0):
                    io.imsave(file_name, (norm_c(self.projs_stru[:, :, i].T)).T)
                file_name = os.path.join(url, 'projs_flow_' + str(i) + '.png')
                if not (np.sum(~np.isnan(self.projs_flow[:, :, i])) == 0):
                    io.imsave(file_name, (norm_c(self.projs_flow[:, :, i].T)).T)

            self.proj_data.to_csv(os.path.join(url, 'proj_settings.csv'))
        except Exception as e:
            print('Projection failed: ', e)

    def save_seg(self, url):
        '''save the segmentation files
        '''
        file_name_mat = os.path.join(url, 'layers.mat')
        file_name_npy = os.path.join(url, 'layers.npy')
        img_obj.save_layers(file_name_mat)
        img_obj.save_layers(file_name_npy)

    def save_seg_video(self, url):
        '''save the segmentation video
        '''
        file_name = os.path.join(url, 'lines.avi')
        img_obj.save_seg_video(step=10, file_name=file_name)

    def save_layer_volumes(self, url):
        '''save the segmentation video
        '''
        # save all slabs
        for row in range(0, self.proj_data.shape[0]):
            start_layer, end_layer, start_offset, end_offset, proj_method, is_proj_stru, is_proj_flow = self.get_porjection_information_from_table(
                row)
            if is_proj_stru:
                img_proj_stru = img_obj.save_single_layer_volume(img_obj.stru3d,
                                                                 start_layer,
                                                                 end_layer,
                                                                 start_offset,
                                                                 end_offset)
                file_name = os.path.join(url, 'layer_stru_' + str(row) + '.avi')
                # print('salb saved:', file_name)
                img_obj.save_video(img_proj_stru.filled(0), file_name)

            if is_proj_flow:
                img_proj_flow = img_obj.save_single_layer_volume(img_obj.flow3d,
                                                                 start_layer,
                                                                 end_layer,
                                                                 start_offset,
                                                                 end_offset)
                file_name = os.path.join(url, 'layer_flow_' + str(row) + '.avi')
                img_obj.save_video(img_proj_flow.filled(0), file_name)

            self.send_and_display_the_log('Save slab num: ' + str(row))

    def save_ori_video(self, url):
        """
        save the original video to avi
        :param url:
        :return:
        """
        if img_obj.exist_stru:
            file_name = os.path.join(url, 'ori_struct.avi')
            img_obj.save_video(img_obj.stru3d, file_name)

        if img_obj.exist_flow:
            file_name = os.path.join(url, 'ori_flow.avi')
            img_obj.save_video(img_obj.flow3d, file_name)


    def save_OAC_video(self, url):
        """Calculate and save the OAC volume from the structural data
        """
        try:
            self.oac = OAC_calculation(img_obj.stru3d)
            max_val = np.nanpercentile(self.oac, 99)
            print('99%max val in OAC=', max_val)
            self.oac[self.oac > max_val] = max_val
            self.oac = self.oac / max_val * 255  # normalize to uint8
            file_name = os.path.join(url, 'OAC_struct.dcm')
            img_obj.save_video(self.oac, file_name)
        except Exception as e:
            print(e)

    # %% batch processing

    def on_button_clicked_open_batch(self):
        """ Open the folder for batch processing

        :return:
        """
        url = self.lineEdit_OpenPath.text()
        try:
            url = self.open_folder(url)

            # put the url to the line edit
            url = str(Path(url))
            self.lineEdit_OpenPath.setText(url)
            self.lineEdit_SavePath.setText(url)

            print('open batch folder  =', url)

            file_df = self.search_files_in_folder(url)
            self.file_df = file_df
            self.batch_url = url

            # clean previous logs
            self.listWidget_BatchList.clear()

            # display
            self.display_batch_files_to_list(file_df)
        except Exception as e:
            print(e)

    def search_files_in_folder(self, url):
        """ get files by file pattern
        Return:
            dataframe
        """
        '''
        stru_pattern = 'stru.dcm'
        flow_pattern = 'flow.dcm'
        seg_pattern = 'seg.npy'
        url = '../data'
        '''
        try:
            stru_pattern = self.lineEdit_StruPattern.text()
            flow_pattern = self.lineEdit_FlowPattern.text()
            seg_pattern = self.lineEdit_SegPattern.text()
            file_df = pd.DataFrame(columns=['stru', 'flow', 'seg'])

            file_list = os.listdir(url)

            for file in file_list:
                # print(file)
                loc = file.find(stru_pattern)
                # loc = file.find(flow_pattern)
                # file is the stru
                if loc != -1:
                    file_df = file_df.append({'stru': os.path.join(url, file)},
                                             ignore_index=True)

                    # get the general file name
                    file_name = file[0: loc]
                    # file_format = file.split('.')[-1]
                    flow_name = file_name + flow_pattern
                    # seg_name = file_name + seg_pattern
                    flow_name = os.path.join(url, flow_name)
                    seg_name = os.path.join(url, self.get_file_name(file),
                                            seg_pattern)
                    print(flow_name, '  ', seg_name)

                    row_index = file_df.shape[0] - 1

                    if os.path.exists(flow_name):
                        file_df.iloc[row_index, 1] = flow_name
                    if os.path.exists(seg_name):
                        file_df.iloc[row_index, 2] = seg_name

            print(file_df)
        except Exception as e:
            print(e)
        return file_df

    def display_batch_files_to_list(self, file_df):
        """

        :return:
        """
        for index, row in file_df.iterrows():
            self.listWidget_BatchList.addItem('Batch ' + str(index))
            self.listWidget_BatchList.addItem(
                'Stru: ' + os.path.basename(str(row['stru'])))
            self.listWidget_BatchList.addItem(
                'Flow: ' + os.path.basename(str(row['flow'])))
            self.listWidget_BatchList.addItem(
                'Seg: ' + os.path.basename(str(row['seg'])))

    def on_button_clicked_run_single(self):
        """Run the select files from the list
        :return:
        """
        try:
            # get the file names from list
            row = self.listWidget_BatchList.currentRow()

            row_num = row // 4
            stru_file = self.file_df.iloc[row_num, 0]
            flow_file = self.file_df.iloc[row_num, 1]
            seg_file = self.file_df.iloc[row_num, 2]

            self.process_single_file(stru_file, flow_file, seg_file)
        except Exception as e:
            print(e)

    def on_button_clicked_run_batch(self):
        """Run batch files
        :return:
        """
        try:
            for index, row in self.file_df.iterrows():
                row_num = index
                stru_file = self.file_df.iloc[row_num, 0]
                flow_file = self.file_df.iloc[row_num, 1]
                seg_file = self.file_df.iloc[row_num, 2]

                self.process_single_file(stru_file, flow_file, seg_file)
        except Exception as e:
            print(e)

    def process_single_file(self, stru_file, flow_file, seg_file):
        """ Run single fils
        :return:
        """
        print('processing the file: ', stru_file)
        self.send_and_display_the_log('processing the file: ' + stru_file)

        # img_obj = Oct3Dimage()
        img_obj.read_stru_data(stru_file)
        self.radioButton_DisplayStru.setEnabled(True)
        self.filename = self.get_file_name(stru_file)
        img_obj.save_path = stru_file

        # if the flow not nan
        if not self.isnan(flow_file):
            img_obj.read_flow_data(flow_file)  # read the flow data
            self.radioButton_DisplayFlow.setEnabled(True)

        # if not exist seg files, do auto segmentation
        if self.isnan(seg_file):
            self.on_button_auto_seg()

            upper_offset = self.spinBox_RetinaUpperOffset.value()
            lower_offset = self.spinBox_RetinaLowerOffset.value()

            if self.radioButton_Eye.isChecked()== True:
                if img_obj.exist_flow:
                    img_obj.auto_seg_stru(auto_range=True, retina_lower_offset=lower_offset,
                                          retina_upper_offset=upper_offset, mode='eye', better_BM=True)
                else:
                    img_obj.auto_seg_stru(auto_range=True, retina_lower_offset=lower_offset,
                                          retina_upper_offset=upper_offset, mode='eye', better_BM=False)

            elif self.radioButton_Skin.isChecked()== True:
                img_obj.auto_seg_stru(auto_range=True, retina_lower_offset=lower_offset,
                                      retina_upper_offset=upper_offset, mode='skin', better_BM=False)

        else:
            img_obj.read_seg_layers(seg_file)  # read the segmentation file

        # load segmentation setting
        proj_url = os.path.join(self.batch_url, 'proj_settings.csv')
        if os.path.exists(proj_url):
            self.proj_data = pd.read_csv(proj_url)
            self.proj_data.drop(['Unnamed: 0'], inplace=True, axis=1,
                                errors='ignore')

            # display the setting in the table
            self.display_projection_to_table()

        # save files
        self.on_button_clicked_save_results()
        self.send_and_display_the_log('result saved: ' + stru_file)

    def isnan(self, value):
        try:
            status = np.isnan(value)
            return status
        except Exception as e:
            return False

    def on_button_clicked_manual_segmentation(self):
        """ Open a new window for manual segmentation
        """
        try:
            self.SW = SegWindow()
            self.SW.show()
        except Exception as e:
            print(e)


    def on_button_clicked_CCquan(self):
        """
        Open the new window for CC quantification
        """
        try:
            self.CCquan = CCquanWindow()
            self.CCquan.show()
        except Exception as e:
            print('Cannot open CC window: ', e)

    def on_button_clicked_Choroid(self):
        """
        Open the new window for choroid quantification
        """
        try:
            print('TODO: Yet to implement Choroidal Thickness software here ')
        except Exception as e:
            print('Cannot open choroid window: ', e)

    def on_button_clicked_ORLdifference(self):
        """
        Open the new window for ORL difference
        """
        try:
            self.Difference = ORLdifference()
            self.Difference.show()
        except Exception as e:
            print('Cannot open Difference window: ', e)

    def on_button_clicked_ORLThicknessWindow(self):
        """
            Open the new window for ORL Thickness window
            """
        try:
            self.ORLThickness = ORLThicknessWindow()
            self.ORLThickness.show()
        except Exception as e:
            print('Cannot open ORL Thickness window: ', e)


def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))  # or 'Windows' 'GTK+' 'Fusion'

    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    form = MainWindow()
    # form.showMaximized()
    form.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
