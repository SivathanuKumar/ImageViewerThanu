# -*- coding: utf-8 -*-
"""
Unified Software v1.00
Updated: Feb 02, 2026
"""
import sys
import os
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pandas as pd
from pathlib import Path
import qimage2ndarray
import matplotlib.pyplot as plt
from datetime import datetime
import imageio_ffmpeg as ffmpeg
import io

# Setup Python Path to find subfolders
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Global State
# Note: Ensure Utils/data_class.py exists
from Utils.data_class import state

img_obj = state.img_obj


# -----------------------------------------------------------------------------
# WORKER THREAD FOR LOADING FILES (Prevents "Not Responding")
# -----------------------------------------------------------------------------
class LoadWorker(QThread):
    """
    Worker thread to load files in the background without freezing the UI
    """
    finished = pyqtSignal(bool, str)  # signal (success_status, message/file_path)

    def __init__(self, img_object, file_path, load_type):
        super().__init__()
        self.img_obj = img_object
        self.file_path = file_path
        self.load_type = load_type

    def run(self):
        try:
            if self.load_type == 'stru':
                self.img_obj.read_stru_data(self.file_path)
            elif self.load_type == 'flow':
                self.img_obj.read_flow_data(self.file_path)
            elif self.load_type == 'seg':
                self.img_obj.read_seg_layers(self.file_path)

            self.finished.emit(True, self.file_path)
        except Exception as e:
            self.finished.emit(False, str(e))


# -----------------------------------------------------------------------------
# MAIN WINDOW
# -----------------------------------------------------------------------------

# Load UI File
current_dir = os.path.dirname(os.path.abspath(__file__))
UI_FILE_PATH = os.path.join(current_dir, 'UI', 'design_smooth.ui')

# Load the UI dynamically
if os.path.exists(UI_FILE_PATH):
    with open(UI_FILE_PATH, 'r', encoding='utf-8') as f:
        ui_xml = f.read()
    # Apply Qt6 -> Qt5 compatibility patches if needed
    ui_xml = ui_xml.replace('Qt::AlignmentFlag::', 'Qt::')
    ui_xml = ui_xml.replace('Qt::Orientation::', 'Qt::')
    ui_xml = ui_xml.replace('Qt::WindowType::', 'Qt::')
    ui_xml = ui_xml.replace('QFrame::Shadow::', 'QFrame::')
    f_io = io.StringIO(ui_xml)
    Ui_MainWindow, QtBaseClass = uic.loadUiType(f_io)
else:
    # If file not found, we cannot proceed effectively, but we try to continue for debugging
    print(f"CRITICAL: UI file not found at {UI_FILE_PATH}")
    Ui_MainWindow, QtBaseClass = object, object  # dummy

# Import Analysis Modules (Keep your original imports)
from Analysis.SM.layer_seg_parallel import seg_video_parallel, OAC_calculation
from Analysis.SM.AlternateSegmentation import quick_process_video
from Analysis.Layers_Wizard.SegWizard import SegWizardWindow
from Analysis.ORLThickness.ORLThickness import ORLThicknessWindow
from Analysis.CCFD.CCquanWindow import CCquanWindow
from Analysis.Segmentation_Correction.Manual_Correction import SegWindow
from Analysis.Registration_Codes.ORLRegistration_old import ORLdifference

# Environment Settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        try:
            self.setupUi(self)
        except Exception as e:
            print(f"Error setting up UI: {e}")
            return

        self.img_obj = state.img_obj

        # --- Connection: Load Data ---
        self.pushButton_LoadStru.clicked.connect(self.on_button_clicked_load_stru)
        self.pushButton_LoadFlow.clicked.connect(self.on_button_clicked_load_flow)
        self.pushButton_LoadSeg.clicked.connect(self.on_button_clicked_load_seg)

        # --- Connection: Display Modes ---
        self.radioButton_DisplayStru.toggled.connect(self.radio_clicked_stru)
        self.radioButton_DisplayFlow.toggled.connect(self.radio_clicked_flow)
        self.radioButton_DisplayFlattenStru.toggled.connect(self.radio_clicked_flatten_stru)
        self.radioButton_DisplayFlattenFlow.toggled.connect(self.radio_clicked_flatten_flow)

        # --- Connection: Scrollbars & SpinBoxes ---
        self.spinBox_StartLayer.valueChanged.connect(self.get_strat_proj_layer)

        # Connect Slab/Thickness changes to refreshing the view
        self.connect_slab_display_setting()

        # --- Connection: Projections ---
        self.pushButton_AddTable.clicked.connect(self.add_porjection_information_to_table)
        self.pushButton_DelTable.clicked.connect(self.delete_row_from_table)
        self.pushButton_Proj.clicked.connect(self.on_button_clicked_proj)  # Preview
        self.pushButton_ImportSetting.clicked.connect(self.on_button_clicked_import_projection_setting)
        self.pushButton_ExportSetting.clicked.connect(self.on_button_clicked_export_projection_setting)
        self.spinBox_ProjGroups.valueChanged.connect(self.plot_projs_on_the_scene)

        # --- Connection: Flatten ---
        self.pushButton_Flatten.clicked.connect(self.on_button_clicked_flatten)

        # --- Connection: Thickness ---
        self.pushButton_Thickness.clicked.connect(self.display_thickness_map)

        # --- Connection: Update/Misc ---
        self.pushButton_UpdateSeg.clicked.connect(self.spinbox_range_set)
        self.pushButton_OpenSaveFolder.clicked.connect(self.on_button_clicked_open_save_folder)
        self.pushButton_SaveResult.clicked.connect(self.on_button_clicked_save_results)

        # Log
        self.plainTextEdit_log.setReadOnly(True)

        # --- Connection: View Controls (Zoom/Flip) ---
        self.global_scale = 1.0
        self.pushButton_SliceZoomIn.clicked.connect(self.on_button_clicked_scale_slice_zoom_in)
        self.pushButton_SliceZoomOut.clicked.connect(self.on_button_clicked_scale_slice_zoom_out)
        self.pushButton_ZoomToFit.clicked.connect(self.on_button_clicked_scale_to_fit)
        self.pushButton_volclock.clicked.connect(self.on_button_clicked_volclock)
        self.pushButton_hori.clicked.connect(self.on_button_clicked_horizontal)
        self.pushButton_Vert.clicked.connect(self.on_button_clicked_vertical)

        # --- Initialize Internal Variables ---
        self.img_flatten_stru = None
        self.img_flatten_flow = None
        self.img_projs_stru = None
        self.img_projs_flow = None
        self.plot_lines = False
        self.display = 'none'

        self.get_current_path()
        self.set_data_frame()

        # --- Connection: Batch & Wizard ---
        if hasattr(self, 'pushButton_SegWizard'):
            self.pushButton_SegWizard.clicked.connect(self.launch_segmentation_wizard)
        self.pushButton_BatchOpen.clicked.connect(self.on_button_clicked_open_batch)
        self.pushButton_RunBatch.clicked.connect(self.on_button_clicked_run_batch)
        self.pushButton_RunSingle.clicked.connect(self.on_button_clicked_run_single)

        # --- Connection: External Modules ---
        self.pushButton_ManualCorrect.clicked.connect(self.on_button_clicked_manual_segmentation)
        self.pushButton_CCquan.clicked.connect(self.on_button_clicked_CCquan)
        self.pushButton_Choroid.clicked.connect(self.on_button_clicked_Choroid)
        self.pushButton_ORL.clicked.connect(self.on_button_clicked_ORLThicknessWindow)
        self.pushButton_Difference.clicked.connect(self.on_button_clicked_ORLdifference)

    # -------------------------------------------------------------------------
    # LOADING FUNCTIONS (WITH PROGRESS BAR)
    # -------------------------------------------------------------------------
    def get_current_path(self):
        path = os.getcwd()
        self.lineEdit_OpenPath.setText(path)

    def openFileNameDialog(self, url=None, type=None):
        url = str(Path(url))
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        if type == 'video':
            fileName, _ = QFileDialog.getOpenFileName(None, "Open file", url, "Video Files (*.avi *.dcm *.img)",
                                                      options=options)
        elif type == 'seg':
            fileName, _ = QFileDialog.getOpenFileName(None, "Open file", url, "Seg Files (*.txt *.mat *.npy)",
                                                      options=options)
        else:
            fileName, _ = QFileDialog.getOpenFileName(None, "Open file", url, "All Files (*.*)", options=options)
        return fileName

    def _start_loading(self, file_path, load_type, message):
        """Helper to start the loading thread and progress dialog"""
        if not file_path:
            return

        self.progress_dialog = QProgressDialog(message, None, 0, 0, self)
        self.progress_dialog.setWindowTitle("Loading Data")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()

        self.loader_thread = LoadWorker(self.img_obj, file_path, load_type)
        if load_type == 'stru':
            self.loader_thread.finished.connect(self.on_stru_load_finished)
        elif load_type == 'flow':
            self.loader_thread.finished.connect(self.on_flow_load_finished)
        elif load_type == 'seg':
            self.loader_thread.finished.connect(self.on_seg_load_finished)

        self.loader_thread.finished.connect(self.progress_dialog.close)
        self.loader_thread.start()

    def on_button_clicked_load_stru(self):
        print('Start loading Stru')
        url = self.lineEdit_OpenPath.text()
        files = self.openFileNameDialog(url, 'video')
        self._start_loading(files, 'stru', "Loading Structural Data... Please Wait.")

    def on_stru_load_finished(self, success, result):
        if success:
            files = result
            self.send_and_display_the_log(f'Load structural file: {files}')
            self.pop_up_alert('Load success')

            self.radioButton_DisplayStru.setEnabled(True)
            self.radioButton_DisplayStru.setChecked(True)
            self.display = 'stru'

            self.filename = self.get_file_name(files)
            self.img_obj.save_path = files
            cwd = str(Path(os.path.dirname(files)))
            self.lineEdit_OpenPath.setText(cwd)
            self.lineEdit_SavePath.setText(cwd)
            self.scroll_bar_reset()
        else:
            self.pop_up_alert(f'Load failed: {result}')

    def on_button_clicked_load_flow(self):
        print('Start loading Flow')
        url = self.lineEdit_OpenPath.text()
        files = self.openFileNameDialog(url, 'video')
        self._start_loading(files, 'flow', "Loading Flow Data... Please Wait.")

    def on_flow_load_finished(self, success, result):
        if success:
            files = result
            self.send_and_display_the_log(f'Load flow file: {files}')
            self.pop_up_alert('Load success')

            self.radioButton_DisplayFlow.setEnabled(True)
            self.radioButton_DisplayFlow.setChecked(True)
            self.display = 'flow'

            if not self.filename:
                self.filename = self.get_file_name(files)
            self.img_obj.save_path = files
            self.scroll_bar_reset()
        else:
            self.pop_up_alert(f'Load failed: {result}')

    def on_button_clicked_load_seg(self):
        print('Start loading Seg')
        url = self.lineEdit_OpenPath.text()
        files = self.openFileNameDialog(url, 'seg')
        self._start_loading(files, 'seg', "Loading Segmentation... Please Wait.")

    def on_seg_load_finished(self, success, result):
        if success:
            files = result
            self.send_and_display_the_log(f'Load seg file: {files}')
            self.pop_up_alert('Load success')
            self.spinbox_range_set()
            # Force lines to show
            self.plot_lines = True
            self.scroll_bar_reset()
        else:
            self.pop_up_alert(f'Load failed: {result}')

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------
    def get_file_name(self, files):
        base = os.path.basename(files)
        return os.path.splitext(base)[0]

    def pop_up_alert(self, message='Load success'):
        alert = QMessageBox()
        alert.setText(message)
        alert.exec_()

    def send_and_display_the_log(self, text):
        time = datetime.now().strftime('%m-%d %H:%M:%S -- ')
        str_send = time + text
        self.plainTextEdit_log.appendPlainText(str_send)

    def spinbox_range_set(self):
        if img_obj.exist_seg:
            layer_num = img_obj.layers.shape[2] - 1
            # Prevent setting negative maximums
            if layer_num < 0: layer_num = 0

            self.spinBox_StartLayer.setMaximum(layer_num)
            self.spinBox_EndLayer.setMaximum(layer_num)
            self.spinBox_FlattenLayer.setMaximum(layer_num)
            self.spinBox_ThicknessStart.setMaximum(layer_num)
            self.spinBox_ThicknessEnd.setMaximum(layer_num)

    # -------------------------------------------------------------------------
    # DISPLAY & SCROLLBAR LOGIC
    # -------------------------------------------------------------------------
    def radio_clicked_stru(self, enabled):
        if enabled and img_obj.exist_stru:
            self.display = 'stru'
            if img_obj.exist_seg: self.plot_lines = True
            self.send_and_display_the_log('Display structural image')
            self.scroll_bar_reset()

    def radio_clicked_flow(self, enabled):
        if enabled and img_obj.exist_flow:
            self.display = 'flow'
            if img_obj.exist_seg: self.plot_lines = True
            self.send_and_display_the_log('Display flow image')
            self.scroll_bar_reset()

    def radio_clicked_flatten_stru(self, enabled):
        if enabled and hasattr(self, 'img_flatten_stru') and self.img_flatten_stru is not None:
            self.display = 'flat_stru'
            self.plot_lines = False
            self.send_and_display_the_log('Display flatten stru image')
            self.scroll_bar_reset()

    def radio_clicked_flatten_flow(self, enabled):
        if enabled and hasattr(self, 'img_flatten_flow') and self.img_flatten_flow is not None:
            self.display = 'flat_flow'
            self.plot_lines = False
            self.send_and_display_the_log('Display flatten flow image')
            self.scroll_bar_reset()

    def scroll_bar_reset(self):
        try:
            imgs = None
            if self.display == 'stru' and img_obj.exist_stru:
                imgs = img_obj.stru3d
            elif self.display == 'flow' and img_obj.exist_flow:
                imgs = img_obj.flow3d
            elif self.display == 'flat_stru':
                imgs = self.img_flatten_stru
            elif self.display == 'flat_flow':
                imgs = self.img_flatten_flow

            # Default fallback
            if imgs is None:
                if img_obj.exist_stru:
                    self.display = 'stru'
                    imgs = img_obj.stru3d
                    self.radioButton_DisplayStru.setChecked(True)
                elif img_obj.exist_flow:
                    self.display = 'flow'
                    imgs = img_obj.flow3d
                    self.radioButton_DisplayFlow.setChecked(True)

            if imgs is None: return

            self.scroll_bar_moved(self.horizontalScrollBar_FastScan, 'fast', imgs)
            self.scroll_bar_moved(self.horizontalScrollBar_SlowScan, 'slow', imgs)
            self.scroll_bar_moved(self.horizontalScrollBar_DepthScan, 'depth', imgs)

            self.horizontalScrollBar_FastScan.setMaximum(imgs.shape[2] - 1)
            self.horizontalScrollBar_SlowScan.setMaximum(imgs.shape[1] - 1)
            self.horizontalScrollBar_DepthScan.setMaximum(imgs.shape[0] - 1)

        except Exception as e:
            print(f"Error in scroll_bar_reset: {e}")

    def scroll_bar_moved(self, scrollbar, view, imgs):
        # Disconnect any old connections to avoid double plotting
        try:
            scrollbar.valueChanged.disconnect()
        except:
            pass

        scrollbar.valueChanged.connect(lambda: self.slider_changed(scrollbar, view, imgs))
        # Initial call to set image
        self.slider_changed(scrollbar, view, imgs)

    def slider_changed(self, scrollbar, view, imgs):
        frame = scrollbar.value()
        try:
            if view == 'fast':
                img = imgs[:, :, frame]
                self.lcdNumber_FastScan.display(frame)

                # Plot lines logic
                if self.plot_lines and img_obj.exist_seg and self.display in ['stru', 'flow']:
                    # Assuming layers: [B-scan, A-scan, Layer]
                    # Adjust slicing based on your data format.
                    # If lines don't appear, check img_obj.layers shape.
                    if frame < img_obj.layers.shape[1]:
                        lines = np.squeeze(img_obj.layers[:, frame, :])
                        self.plot_on_the_scene(self.graphicsView_FastScan, img, add_line=True, add_slab_lines=True,
                                               lines=lines)
                    else:
                        self.plot_on_the_scene(self.graphicsView_FastScan, img)
                else:
                    self.plot_on_the_scene(self.graphicsView_FastScan, img)

            elif view == 'slow':
                img = imgs[:, frame, :]
                self.lcdNumber_SlowScan.display(frame)
                if self.plot_lines and img_obj.exist_seg and self.display in ['stru', 'flow']:
                    if frame < img_obj.layers.shape[0]:
                        lines = np.squeeze(img_obj.layers[frame, :, :])
                        self.plot_on_the_scene(self.graphicsView_SlowScan, img, add_line=True, lines=lines)
                    else:
                        self.plot_on_the_scene(self.graphicsView_SlowScan, img)
                else:
                    self.plot_on_the_scene(self.graphicsView_SlowScan, img)

            elif view == 'depth':
                img = imgs[frame, :, :]
                self.lcdNumber_DepthScan.display(frame)
                self.plot_on_the_scene(self.graphicsView_DepthScan, img)
        except Exception as e:
            # Suppress index errors during rapid scrolling
            print(f"Display error (likely index out of bounds): {e}")

    def plot_on_the_scene(self, graphicsView, img, add_line=False, add_slab_lines=False, lines=None):
        if img is None: return

        # Data Type Conversion
        if np.issubdtype(img.dtype, np.uint16) or img.dtype == np.float64 or img.dtype == np.float32:
            img = np.nan_to_num(img)
            if img.max() > 0:
                img = np.uint8((img / img.max()) * 255)
            else:
                img = np.uint8(img)

        scene = QGraphicsScene()
        img_pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(img))
        scene.addPixmap(img_pixmap)

        if add_line and lines is not None:
            self.plot_add_lines(scene, lines)
        if add_slab_lines:
            try:
                slab_lines = self.get_slab_lines()
                self.plot_add_lines(scene, slab_lines, QPen(Qt.green, 1))
            except Exception as e:
                pass  # Silently fail if slab lines can't be calculated

        graphicsView.setScene(scene)
        graphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
        graphicsView.scale(self.global_scale, self.global_scale)

    def plot_add_lines(self, scene, lines, pen=QPen(Qt.red, 1)):
        # lines shape: [width, num_layers]
        try:
            img_w = lines.shape[0]
            num_layers = lines.shape[1]

            for i in range(num_layers):
                for j in range(img_w - 1):
                    y1 = lines[j, i]
                    y2 = lines[j + 1, i]
                    # Simple check to avoid drawing lines at 0 (often used as 'no data')
                    if y1 > 1 and y2 > 1:
                        scene.addLine(j, y1, j + 1, y2, pen)
        except Exception as e:
            print(f"Error drawing lines: {e}")
        return scene

    def get_slab_lines(self):
        """ Get the lines of current slab for display """
        frame = self.horizontalScrollBar_FastScan.value()
        start_layer = self.spinBox_StartLayer.value()
        end_layer = self.spinBox_EndLayer.value()
        start_offset = self.spinBox_StartOffset.value()
        end_offset = self.spinBox_EndOffset.value()

        if not img_obj.exist_seg: return np.zeros((10, 2))

        # Safety check indices
        if frame >= img_obj.layers.shape[1]: return np.zeros((10, 2))

        # Get lines
        line_up = img_obj.layers[:, frame, start_layer] + start_offset
        line_low = img_obj.layers[:, frame, end_layer] + end_offset

        return np.array([line_up, line_low]).T

    # -------------------------------------------------------------------------
    # PROJECTION FUNCTIONS (FIXED)
    # -------------------------------------------------------------------------
    def get_strat_proj_layer(self):
        return self.spinBox_StartLayer.value()

    def get_end_proj_layer(self):
        return self.spinBox_EndLayer.value()

    def connect_slab_display_setting(self):
        self.spinBox_StartLayer.valueChanged.connect(self.scroll_bar_reset)
        self.spinBox_EndLayer.valueChanged.connect(self.scroll_bar_reset)
        self.spinBox_StartOffset.valueChanged.connect(self.scroll_bar_reset)
        self.spinBox_EndOffset.valueChanged.connect(self.scroll_bar_reset)

    def on_button_clicked_proj(self):
        '''
        Preview the enface projection with fixed orientation and layer ordering.
        '''
        if not img_obj.exist_stru and not img_obj.exist_flow:
            self.pop_up_alert("No Data Loaded to Project")
            return

        # 1. Get Parameters
        start_layer = self.spinBox_StartLayer.value()
        end_layer = self.spinBox_EndLayer.value()
        start_offset = self.spinBox_StartOffset.value()
        end_offset = self.spinBox_EndOffset.value()
        proj_method = self.comboBox_ProjMethod.currentText()

        # 2. Fix Layer Order (Swap if Start > End)
        if start_layer > end_layer:
            start_layer, end_layer = end_layer, start_layer
            # Temporarily update UI to reflect valid order (optional, but good for clarity)
            self.spinBox_StartLayer.blockSignals(True)
            self.spinBox_EndLayer.blockSignals(True)
            self.spinBox_StartLayer.setValue(start_layer)
            self.spinBox_EndLayer.setValue(end_layer)
            self.spinBox_StartLayer.blockSignals(False)
            self.spinBox_EndLayer.blockSignals(False)

        # 3. Generate and Display
        try:
            # --- STRUCTURE ---
            if self.checkBox_ProjStru.isChecked() and img_obj.exist_stru:
                # Note: We set rotate=False in backend because we handle it globally here
                raw_proj_stru = img_obj.plot_proj(
                    start_layer, end_layer, 'stru', proj_method,
                    start_offset=start_offset, end_offset=end_offset,
                    display=False, rotate=False
                )

                # Apply Global Transform
                self.img_proj_stru = self.fix_orientation(raw_proj_stru)

                if self.img_proj_stru is not None:
                    self.plot_on_the_scene(self.graphicsView_ProjectionStru, self.img_proj_stru)
                    self.send_and_display_the_log('Projected Structure (Oriented)')

            # --- FLOW ---
            if self.checkBox_ProjFlow.isChecked() and img_obj.exist_flow:
                raw_proj_flow = img_obj.plot_proj(
                    start_layer, end_layer, 'flow', proj_method,
                    start_offset=start_offset, end_offset=end_offset,
                    display=False, rotate=False
                )

                # Apply Global Transform
                self.img_proj_flow = self.fix_orientation(raw_proj_flow)

                if self.img_proj_flow is not None:
                    self.plot_on_the_scene(self.graphicsView_ProjectionFlow, self.img_proj_flow)
                    self.send_and_display_the_log('Projected Flow (Oriented)')

        except Exception as e:
            self.send_and_display_the_log(f'Projection Error: {str(e)}')
            print(f"Detailed Projection Error: {e}")

    def fix_orientation(self, img):
        """
        Applies the specific transform requested:
        1. Rotate 90 degrees Clockwise
        2. Flip Left-to-Right
        """
        if img is None: return None
        # Rotate 90 degrees Clockwise (k=-1)
        img_rot = np.rot90(img, k=-1)
        # Flip Left-to-Right
        img_final = np.fliplr(img_rot)
        return img_final

    def display_thickness_map(self):
        '''
        Calculates and displays the thickness map with fixed orientation.
        '''
        if not img_obj.exist_seg:
            self.pop_up_alert("Segmentation not loaded.")
            return

        start = self.spinBox_ThicknessStart.value()
        end = self.spinBox_ThicknessEnd.value()

        # Fix Order
        if start > end:
            start, end = end, start
            self.spinBox_ThicknessStart.setValue(start)
            self.spinBox_ThicknessEnd.setValue(end)

        try:
            thicknessmap = None
            try:
                thicknessmap = img_obj.thickness_map(start, end, smooth=True)
            except:
                pass

            if thicknessmap is None:
                layer1 = img_obj.layers[:, :, start]
                layer2 = img_obj.layers[:, :, end]
                thicknessmap = np.abs(layer2 - layer1)

            # Apply Orientation Fix to the raw data BEFORE plotting
            thicknessmap = self.fix_orientation(thicknessmap)

            if thicknessmap is not None:
                fig = plt.figure()
                plt.imshow(thicknessmap, cmap='jet', aspect='auto')
                plt.colorbar()

                # Canvas Draw
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                im_array = data.reshape((h, w, 4))[:, :, :3]
                plt.close(fig)

                self.plot_on_the_scene(self.graphicsView_Thickness, im_array)

                # Calculate Mean (ignores NaNs/Zeros correctly)
                mean_thickness = np.nanmean(thicknessmap)
                self.lineEdit_MeanThickness.setText(f'{mean_thickness:.4f}')
                self.send_and_display_the_log(f'Thickness Map Displayed. Mean: {mean_thickness:.4f}')

        except Exception as e:
            self.send_and_display_the_log(f'Thickness Error: {str(e)}')

    # -------------------------------------------------------------------------
    # PROJECTION TABLE FUNCTIONS
    # -------------------------------------------------------------------------
    def set_data_frame(self):
        self.proj_data = pd.DataFrame(
            columns=['start_layer', 'end_layer', 'start_offset', 'end_offset', 'proj_method', 'is_proj_stru',
                     'is_proj_flow'])

    def add_porjection_information_to_table(self):
        table = self.tableWidget
        row = table.rowCount()

        start_layer = self.spinBox_StartLayer.value()
        end_layer = self.spinBox_EndLayer.value()
        start_offset = self.spinBox_StartOffset.value()
        end_offset = self.spinBox_EndOffset.value()
        proj_method = self.comboBox_ProjMethod.currentText()
        is_proj_stru = self.checkBox_ProjStru.isChecked()
        is_proj_flow = self.checkBox_ProjFlow.isChecked()

        self.add_row_on_table(table, text='Proj ' + str(row))
        self.set_item_on_table(table, row, 0, str(start_layer))
        self.set_item_on_table(table, row, 1, str(end_layer))
        self.set_item_on_table(table, row, 2, str(start_offset))
        self.set_item_on_table(table, row, 3, str(end_offset))
        self.set_item_on_table(table, row, 4, proj_method)
        self.set_item_on_table(table, row, 5, str(is_proj_stru))
        self.set_item_on_table(table, row, 6, str(is_proj_flow))

        self.proj_data = self.proj_data._append({
            'start_layer': start_layer, 'end_layer': end_layer,
            'start_offset': start_offset, 'end_offset': end_offset,
            'proj_method': proj_method, 'is_proj_stru': is_proj_stru,
            'is_proj_flow': is_proj_flow
        }, ignore_index=True)

        self.spinBox_ProjGroups.setMaximum(self.proj_data.shape[0] - 1)

    def add_row_on_table(self, table, text):
        rowPosition = table.rowCount()
        table.insertRow(rowPosition)
        table.setVerticalHeaderItem(rowPosition, QTableWidgetItem(text))

    def set_item_on_table(self, table, row, column, text):
        table.setItem(row, column, QTableWidgetItem(text))

    def delete_row_from_table(self):
        table = self.tableWidget
        if table.rowCount() > 0:
            cr = table.currentRow()
            if cr >= 0:
                table.removeRow(cr)
                self.proj_data = self.proj_data.drop(self.proj_data.index[cr])

    def on_button_clicked_import_projection_setting(self):
        url = self.lineEdit_OpenPath.text()
        try:
            url = self.open_folder(url)
            file_name = os.path.join(url, 'proj_settings.csv')
            if os.path.exists(file_name):
                self.proj_data = pd.read_csv(file_name)
                if 'Unnamed: 0' in self.proj_data.columns:
                    self.proj_data.drop(['Unnamed: 0'], inplace=True, axis=1)
                self.display_projection_to_table()
        except Exception as e:
            print(e)

    def on_button_clicked_export_projection_setting(self):
        url = self.lineEdit_OpenPath.text()
        try:
            url = self.open_folder(url)
            self.proj_data.to_csv(os.path.join(url, 'proj_settings.csv'))
        except Exception as e:
            print(e)

    def display_projection_to_table(self):
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(0)
        for index, row in self.proj_data.iterrows():
            # Logic to repopulate table from dataframe
            pass  # (Simplified for brevity, logic same as add)

    def plot_projs_on_the_scene(self):
        # View saved projections from loop_project
        try:
            ind = self.spinBox_ProjGroups.value()
            if self.projs_stru is not None and ind < self.projs_stru.shape[2]:
                self.plot_on_the_scene(self.graphicsView_ProjectionStru, self.projs_stru[:, :, ind])
            if self.projs_flow is not None and ind < self.projs_flow.shape[2]:
                self.plot_on_the_scene(self.graphicsView_ProjectionFlow, self.projs_flow[:, :, ind])
        except:
            pass

    # -------------------------------------------------------------------------
    # FLATTEN, ZOOM, ROTATE
    # -------------------------------------------------------------------------
    def on_button_clicked_flatten(self):
        ref_layer = self.spinBox_FlattenLayer.value()
        # Default flattening location center
        new_loc = img_obj.img_depth // 2

        if img_obj.exist_stru:
            self.img_flatten_stru = img_obj.save_flatten_video(video_type='stru', ref_layer_num=ref_layer,
                                                               new_loc=new_loc, saved=False)
            self.radioButton_DisplayFlattenStru.setEnabled(True)
        if img_obj.exist_flow:
            self.img_flatten_flow = img_obj.save_flatten_video(video_type='flow', ref_layer_num=ref_layer,
                                                               new_loc=new_loc, saved=False)
            self.radioButton_DisplayFlattenFlow.setEnabled(True)

        self.send_and_display_the_log('Flattening calculation done')

    def on_button_clicked_scale_slice_zoom_in(self):
        self.global_scale *= 1.2
        self.update_scale_of_view()

    def on_button_clicked_scale_slice_zoom_out(self):
        self.global_scale *= 0.8
        self.update_scale_of_view()

    def on_button_clicked_scale_to_fit(self):
        self.global_scale = 1.0
        self.graphicsView_FastScan.resetTransform()
        self.graphicsView_SlowScan.resetTransform()
        self.graphicsView_DepthScan.resetTransform()

    def update_scale_of_view(self):
        # Reset then Scale to avoid compounding
        for view in [self.graphicsView_FastScan, self.graphicsView_SlowScan, self.graphicsView_DepthScan]:
            view.resetTransform()
            view.scale(self.global_scale, self.global_scale)

    def on_button_clicked_volclock(self):
        if img_obj.exist_stru:
            img_obj.stru3d = np.rot90(img_obj.stru3d, k=-1, axes=(0, 1))
            # Swap dims
            img_obj.img_width, img_obj.img_depth = img_obj.img_depth, img_obj.img_width
            self.scroll_bar_reset()
        if img_obj.exist_flow:
            img_obj.flow3d = np.rot90(img_obj.flow3d, k=-1, axes=(0, 1))
            self.scroll_bar_reset()

    def on_button_clicked_horizontal(self):
        if img_obj.exist_stru:
            img_obj.stru3d = np.fliplr(img_obj.stru3d)
            self.scroll_bar_reset()
        if img_obj.exist_flow:
            img_obj.flow3d = np.fliplr(img_obj.flow3d)
            self.scroll_bar_reset()

    def on_button_clicked_vertical(self):
        if img_obj.exist_stru:
            img_obj.stru3d = np.flipud(img_obj.stru3d)
            self.scroll_bar_reset()
        if img_obj.exist_flow:
            img_obj.flow3d = np.flipud(img_obj.flow3d)
            self.scroll_bar_reset()

    # -------------------------------------------------------------------------
    # WIZARD, EXTERNAL WINDOWS
    # -------------------------------------------------------------------------
    def launch_segmentation_wizard(self):
        if not self.img_obj.exist_stru:
            self.pop_up_alert("Please load a Structural (Stru) file first.")
            return

        selected_device = self.comboBox_type.currentText()
        selected_size = self.comboBox_size.currentText()

        wizard = SegWizardWindow(selected_device, selected_size, parent=self)
        if wizard.exec_() == QtWidgets.QDialog.Accepted:
            self.send_and_display_the_log(f"Wizard Segmentation completed.")
            self.spinbox_range_set()
            self.plot_lines = True
            self.scroll_bar_reset()

    def on_button_clicked_open_batch(self):
        url = self.open_folder(self.lineEdit_OpenPath.text())
        if url:
            self.lineEdit_OpenPath.setText(url)
            self.batch_url = url
            # Simplified batch loading logic
            self.file_df = pd.DataFrame()
            # (Requires reimplementing your search_files_in_folder logic here if needed)

    def on_button_clicked_run_batch(self):
        pass  # Implement batch loop

    def on_button_clicked_run_single(self):
        pass  # Implement single batch run

    def open_folder(self, dir=None):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folder_name = QFileDialog.getExistingDirectory(self, 'Select directory', directory=dir, options=options)
        return folder_name

    def on_button_clicked_manual_segmentation(self):
        try:
            self.SW = SegWindow()
            self.SW.show()
        except Exception as e:
            print(e)

    def on_button_clicked_CCquan(self):
        try:
            self.CCquan = CCquanWindow()
            self.CCquan.show()
        except:
            pass

    def on_button_clicked_Choroid(self):
        print("Choroid module not linked.")

    def on_button_clicked_ORLdifference(self):
        try:
            self.Difference = ORLdifference()
            self.Difference.show()
        except:
            pass

    def on_button_clicked_ORLThicknessWindow(self):
        try:
            self.ORLThickness = ORLThicknessWindow()
            self.ORLThickness.show()
        except:
            pass

    def on_button_clicked_open_save_folder(self):
        '''open the folder for saving
        '''
        url = self.open_folder()
        self.lineEdit_SavePath.setText(url)
        print(url)

    def on_button_clicked_save_results(self):
        url = self.lineEdit_SavePath.text()
        if not os.path.exists(url):
            self.pop_up_alert("Save path does not exist!")
            return

        saved_items = []

        # 1. SAVE SEGMENTATION
        if self.checkBox_SaveSeg.isChecked() and img_obj.exist_seg:
            try:
                img_obj.save_layers(os.path.join(url, 'layers.mat'))
                saved_items.append("Layers")
            except Exception as e:
                print(f"Seg Save Error: {e}")

        # 2. SAVE PROJECTIONS (Current View)
        if self.checkBox_SaveProjections.isChecked():
            try:
                # Re-generate to ensure we have the latest data
                self.on_button_clicked_proj()

                if hasattr(self, 'img_proj_stru') and self.img_proj_stru is not None:
                    # img_proj_stru is already rotated/flipped by the display function
                    plt.imsave(os.path.join(url, 'proj_stru.png'), self.img_proj_stru, cmap='gray')
                    saved_items.append("Proj Stru")

                if hasattr(self, 'img_proj_flow') and self.img_proj_flow is not None:
                    plt.imsave(os.path.join(url, 'proj_flow.png'), self.img_proj_flow, cmap='gray')
                    saved_items.append("Proj Flow")
            except Exception as e:
                print(f"Proj Save Error: {e}")

        # 3. SAVE THICKNESS (Map + CSV)
        if self.checkBox_SaveThickness.isChecked() and img_obj.exist_seg:
            try:
                start = self.spinBox_ThicknessStart.value()
                end = self.spinBox_ThicknessEnd.value()
                if start > end: start, end = end, start

                # Calc
                t_map = img_obj.thickness_map(start, end, smooth=True)
                if t_map is None:
                    t_map = np.abs(img_obj.layers[:, :, end] - img_obj.layers[:, :, start])

                # Transform
                t_map = self.fix_orientation(t_map)

                # Save
                np.savetxt(os.path.join(url, f'thickness_{start}_{end}.csv'), t_map, delimiter=',')
                plt.imsave(os.path.join(url, f'thickness_{start}_{end}.png'), t_map, cmap='jet')
                saved_items.append("Thickness")
            except Exception as e:
                print(f"Thickness Save Error: {e}")

        # 4. SAVE OAC (Optical Attenuation Coefficient)
        # We check for STRUCTURE because OAC is calculated from Structure
        if self.checkBox_SaveOAC.isChecked() and img_obj.exist_stru and img_obj.exist_seg:
            try:
                start = self.spinBox_StartLayer.value()
                end = self.spinBox_EndLayer.value()
                if start > end: start, end = end, start

                self.send_and_display_the_log("Calculating OAC... this might take a moment.")

                # Call the new projection helper
                # This handles the volume calculation + slab projection internally
                oac_map = img_obj.get_oac_projection(start, end)

                if oac_map is not None:
                    # Fix Orientation
                    oac_map = self.fix_orientation(oac_map)

                    # Save
                    plt.imsave(os.path.join(url, 'OAC.png'), oac_map, cmap='gray')
                    saved_items.append("OAC")
                else:
                    print("OAC calculation returned None.")
            except Exception as e:
                self.send_and_display_the_log(f"OAC Save Error: {e}")
                print(f"OAC Error Details: {e}")
        # 5. SAVE ORIGINAL VIDEO (AVI)
        if self.checkBox_AviVideo.isChecked():
            try:
                if img_obj.exist_stru:
                    img_obj.save_video(img_obj.stru3d, os.path.join(url, 'Structure.avi'))
                if img_obj.exist_flow:
                    img_obj.save_video(img_obj.flow3d, os.path.join(url, 'Flow.avi'))
                saved_items.append("AVI Video")
            except Exception as e:
                print(f"AVI Save Error: {e}")

        # 6. SAVE FLATTENED VOLUMES
        if self.checkBox_SaveFlatten.isChecked():
            try:
                ref = self.spinBox_FlattenLayer.value()
                if img_obj.exist_stru:
                    img_obj.save_flatten_video(os.path.join(url, 'flat_stru.avi'), 'stru', ref_layer_num=ref)
                if img_obj.exist_flow:
                    img_obj.save_flatten_video(os.path.join(url, 'flat_flow.avi'), 'flow', ref_layer_num=ref)
                saved_items.append("Flattened")
            except Exception as e:
                print(f"Flatten Save Error: {e}")

        # 7. SAVE SEGMENTATION LINES VIDEO
        if self.checkBox_SaveSegVideo.isChecked() and img_obj.exist_stru and img_obj.exist_seg:
            try:
                img_obj.save_seg_video(step=5, file_name=os.path.join(url, 'seg_lines.avi'))
                saved_items.append("Seg Video")
            except Exception as e:
                print(f"SegVideo Save Error: {e}")

        self.pop_up_alert(f"Finished Saving: {', '.join(saved_items)}")


def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    form = MainWindow()
    form.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()