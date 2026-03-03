import numpy as np
from Analysis.Seg_Correct_better.SegWindowUI import Ui_MainWindow
from superqt.sliders import QRangeSlider
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Utils.OCTpy import *
from matplotlib.cm import get_cmap
import qimage2ndarray
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import interp1d
from scipy import interpolate
import os
import psutil
import gc
# --- MODIFICATION START: Added imports for new features ---
from datetime import datetime
import cv2

# --- MODIFICATION END ---

# --- HELPER/UTILITY FUNCTIONS START ---

# A constant for batch processing size to manage memory usage across different functions.
# This limits how many frames are processed in memory at one time.
# Adjust this value based on your system's RAM. Lower is safer.
MEMORY_FRIENDLY_BATCH_SIZE = 25


def check_memory_usage():
    """Check current memory usage"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.1f} MB")
    return memory_mb


def get_pix_map_util(img, normalize=False):
    """
    Transfers a numpy array to a QPixmap, handling type conversion and optional normalization.
    """
    if np.issubdtype(img.dtype, np.uint16):
        img = np.uint8(img / 256.0)
    img_pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(img, normalize=normalize))
    return img_pixmap


# --- HELPER/UTILITY FUNCTIONS END ---

img_obj = Oct3Dimage()


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.is_distorted = False
        return cls._instance


class ZoomerWidget(QWidget):
    """A small widget that shows a magnified view of an image patch for Point & Click mode."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(128, 128)
        self.label = QLabel(self)
        self.label.setFixedSize(128, 128)
        self.label.setStyleSheet("border: 2px solid #00aaff;")

    def set_image(self, pixmap):
        self.label.setPixmap(pixmap)


class SegWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    '''
    Manual segementaiton mode
    '''
    save_triggered = pyqtSignal()

    # --- CHANGE HERE: Add img_obj_in as an argument ---
    def __init__(self, data_in=None):
        import faulthandler
        faulthandler.enable()

        # Update the global variable
        global img_obj
        if data_in is not None:
            img_obj = data_in

        # -------------------------------------------------------------------

        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.is_flattened_mode = False
        if not (img_obj.exist_seg):
            img_obj.layers = np.zeros((img_obj.img_width, img_obj.img_framenum))

        self.check_and_correct_layer_crossings()
        self.set_init_range()

        # Manages the state for non-cumulative line shifting without copying the whole dataset.
        self._original_slice_for_shift = None
        self._last_shift_params = None

        self.lineEdit_LayerNumber.setText(str(img_obj.layer_num))

        # Init the main scene
        self.scene = GraphicsScene(self)  # Pass self as parent
        self.graphicsView_ManualEdit.setScene(self.scene)
        scale = self.graphicsView_ManualEdit.size().width() / (img_obj.img_width + 15)
        self.graphicsView_ManualEdit.scale(scale, scale)

        # Init the thickness map scene
        self.scene_thickness = GraphicsSceneColormap()
        self.graphicsView_Thickness.setScene(self.scene_thickness)
        self.graphicsView_Thickness.scale(
            self.graphicsView_Thickness.size().width() / (img_obj.img_width + 10),
            self.graphicsView_Thickness.size().height() / (img_obj.img_framenum + 10))

        # Init the Z-slice scene
        self.scene_Zslice = GraphicsSceneZslice()
        self.graphicsView_Zslice.setScene(self.scene_Zslice)
        self.graphicsView_Zslice.scale(
            self.graphicsView_Zslice.size().width() / (img_obj.img_width + 10),
            self.graphicsView_Zslice.size().height() / (img_obj.img_framenum + 10))

        # 1. Configure the range slider's properties
        self.contrastRangeSlider.setOrientation(Qt.Horizontal)
        self.contrastRangeSlider.setRange(0, 255)
        self.contrastRangeSlider.setValue((0, 255))  # Default: full range

        # 2. Connect its valueChanged signal to the handler
        self.contrastRangeSlider.valueChanged.connect(self._on_contrast_changed)

        # 3. Call once to set the initial label text
        self._on_contrast_changed(self.contrastRangeSlider.value())

        # 4. Connect the CLAHE checkbox (this remains the same)
        self.checkBox_applyCLAHE.stateChanged.connect(lambda: self.scene.update_display())


        # --- DUAL MODE SETUP START ---
        # To not set "Point & Click" as the default mode on startup
        self.radioButton_DrawMode.setChecked(True)
        self.scene.set_correction_mode('draw')

        # Connect the radio buttons to the function that switches modes
        self.radioButton_PointMode.toggled.connect(self.on_correction_mode_changed)
        self.radioButton_DrawMode.toggled.connect(self.on_correction_mode_changed)
        # --- DUAL MODE SETUP END ---

        # Connect signals for other UI elements
        self.spinBox_CopyLine.valueChanged.connect(lambda: self.scene.set_values(self.spinBox_CopyLine.value()))
        self.connect_to_scene()
        self.pushButton_AddNewLine.clicked.connect(self.add_new_line)
        self.pushButton_DeleteLine.clicked.connect(self.delete_a_line)
        self.pushButton_Save.clicked.connect(self.save_current_result)
        self.pushButton_Smooth.clicked.connect(self.on_button_clicked_smooth)
        self.pushButton_line_shift.clicked.connect(self.on_button_clicked_line_shift)
        # self.checkBox_DisplayLines.stateChanged.connect(self.update_display_line)
        self.pushButton_EZseg.clicked.connect(self.update_scene)
        self.keypress_count = 0
        # self.pushButton_Undo = QtWidgets.QPushButton(self.groupBox_ManualEdit)
        # self.pushButton_Undo.setText("Undo (Ctrl+Z)")
        # self.gridLayout_7.addWidget(self.pushButton_Undo, 5, 2, 1, 1)  # Adjust grid pos as needed
        self.pushButton_Undo.clicked.connect(self.scene.undo_last_action)
        self.scene_thickness.drawCurrentLine(frame=self.spinBox_FrameNumber.value(),
                                             current_layer=self.spinBox_CurrentLayer.value())
        self.scene_Zslice.drawCurrentLine(frame=self.spinBox_FrameNumber.value(),
                                          current_layer=self.spinBox_CurrentLayer.value())

    def _on_contrast_changed(self, value_tuple):
        """
        Updates the contrast label and triggers a scene redraw.
        The 'value_tuple' will be like (min, max).
        """
        min_val, max_val = value_tuple
        self.contrastValueLabel.setText(f"Display Range: {min_val} - {max_val}")

        if hasattr(self, 'scene'):
            self.scene.update_display()
    def on_correction_mode_changed(self):
        """
        Updates the correction mode in the GraphicsScene based on which radio button is selected.
        """
        if self.radioButton_PointMode.isChecked():
            self.scene.set_correction_mode('point')
            print("Switched to Point & Click correction mode.")
        elif self.radioButton_DrawMode.isChecked():
            self.scene.set_correction_mode('draw')
            print("Switched to Freehand Draw correction mode.")

    def check_and_correct_layer_crossings(self):
        """Iterates through all layers and ensures they do not cross using memory-safe batches."""
        if img_obj.layer_num <= 1: return

        # --- Confirmation Dialog ---
        reply = QMessageBox.question(self, 'Confirm Correction',
                                     "Layer crossings were detected. Do you want to automatically correct them?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.No:
            print("User chose not to correct layer crossings.")
            return
        # --- End Confirmation Dialog ---

        print("Performing initial check for layer crossings...")
        corrections_made = 0
        for frame_start in range(0, img_obj.img_framenum, MEMORY_FRIENDLY_BATCH_SIZE):
            frame_end = min(frame_start + MEMORY_FRIENDLY_BATCH_SIZE, img_obj.img_framenum)
            layer_chunk = img_obj.layers[:, frame_start:frame_end, :].copy()
            for frame_idx_in_chunk in range(layer_chunk.shape[1]):
                for x_pos in range(img_obj.img_width):
                    for layer_idx in range(img_obj.layer_num - 1):
                        y1 = layer_chunk[x_pos, frame_idx_in_chunk, layer_idx]
                        y2 = layer_chunk[x_pos, frame_idx_in_chunk, layer_idx + 1]
                        if y2 <= y1:
                            layer_chunk[x_pos, frame_idx_in_chunk, layer_idx + 1] = y1 + 1
                            corrections_made += 1
            img_obj.layers[:, frame_start:frame_end, :] = layer_chunk
            del layer_chunk
            gc.collect()
        if corrections_made > 0:
            print(f"INFO: Found and corrected {corrections_made} layer crossing points.")

    def update_scene(self):
        self.graphicsView_ManualEdit.resetTransform()
        config = Config()
        viewport_width = self.graphicsView_ManualEdit.viewport().size().width()
        scene_width = self.scene.sceneRect().width()

        if self.pushButton_EZseg.isChecked():
            config.is_distorted = True
            kern = float(self.lineEdit_Kernel_EZ.text())
            config.dist_fact = kern
            new_width = scene_width * kern
            scale_w = viewport_width / new_width
            self.graphicsView_ManualEdit.scale(scale_w * kern, scale_w / kern)
        else:
            config.is_distorted = False
            scale_w = viewport_width / scene_width
            self.graphicsView_ManualEdit.scale(scale_w, scale_w)
        self.scene.update()

    def set_init_range(self):
        """Set the range for widgets, spinbox, scrollbar."""
        self.spinBox_FrameNumber.setMaximum(img_obj.img_framenum - 1)
        self.spinBox_CurrentLayer.setMaximum(img_obj.layer_num - 1)
        self.horizontalScrollBar_ManualEdit.setMaximum(img_obj.img_framenum - 1)
        self.spinBox_CopyLine.setMaximum(img_obj.img_framenum - 1)

    # def update_display_line(self):
    #    """ Display the lines."""
    #    inds = list(range(0, img_obj.layer_num))
    #    self.connect_to_scene(disconnect=True)
    #    self.scene.drawCurrentLine(img_obj.layers, inds=inds,
    #                               frame=self.spinBox_FrameNumber.value(),
    #                               current_layer=self.spinBox_CurrentLayer.value(),
    #                               is_draw_line=self.checkBox_DisplayLines.isChecked())

    def add_new_line(self):
        """Add a new layer, copied from the currently selected layer."""
        self._last_shift_params = None  # Invalidate cache
        current_layer_data = img_obj.layers[:, :, self.spinBox_CurrentLayer.value()].copy()
        img_obj.layers = np.dstack((img_obj.layers, current_layer_data))
        img_obj.layer_num += 1
        self.lineEdit_LayerNumber.setText(str(img_obj.layer_num))
        self.spinBox_CurrentLayer.setMaximum(img_obj.layer_num - 1)
        self.connect_to_scene(disconnect=True)

    def delete_a_line(self):
        """Delete the currently selected layer."""
        self._last_shift_params = None  # Invalidate cache
        if img_obj.layers.shape[2] > 1:
            img_obj.layers = np.delete(img_obj.layers, self.spinBox_CurrentLayer.value(), axis=2)
            img_obj.layer_num = img_obj.layers.shape[2]
            self.disconnect_signals()
            self.lineEdit_LayerNumber.setText(str(img_obj.layer_num))
            self.spinBox_CurrentLayer.setValue(img_obj.layer_num - 1)
            self.spinBox_CurrentLayer.setMaximum(img_obj.layer_num - 1)
            self.connect_to_scene(disconnect=False)

    def connect_to_scene(self, disconnect=False):
        """
        Connect and disconnect signals. This version is updated to use the new,
        more robust `scene.update_display()` method, which prevents crashes and
        simplifies the code.
        """
        if disconnect:
            self.disconnect_signals()

        # --- Connections for the Main GraphicsScene (Refactored) ---
        # These now call the new central update method, passing only the changed value.
        # This is safer and prevents crashes from unconfirmed points when changing frames.
        self.spinBox_FrameNumber.valueChanged.connect(
            lambda val: self.scene.update_display(frame=val)
        )
        self.spinBox_CurrentLayer.valueChanged.connect(
            lambda val: self.scene.update_display(layer=val)
        )
        self.horizontalScrollBar_ManualEdit.valueChanged.connect(
            lambda val: self.scene.update_display(frame=val)
        )
        self.checkBox_DisplayLines.stateChanged.connect(
            lambda state: self.scene.update_display(is_draw_line=(state == Qt.Checked))
        )

        # --- Connections for Thickness and Z-slice Scenes (Unchanged) ---
        # These scenes have their own simple drawing logic, so they remain the same.
        self.spinBox_CurrentLayer.valueChanged.connect(
            lambda: self.scene_thickness.drawCurrentLine(frame=self.spinBox_FrameNumber.value(),
                                                         current_layer=self.spinBox_CurrentLayer.value())
        )
        self.horizontalScrollBar_ManualEdit.valueChanged.connect(
            lambda: self.scene_thickness.drawCurrentLine(frame=self.horizontalScrollBar_ManualEdit.value(),
                                                         current_layer=self.spinBox_CurrentLayer.value())
        )
        self.spinBox_CurrentLayer.valueChanged.connect(
            lambda: self.scene_Zslice.drawCurrentLine(frame=self.spinBox_FrameNumber.value(),
                                                      current_layer=self.spinBox_CurrentLayer.value())
        )
        self.horizontalScrollBar_ManualEdit.valueChanged.connect(
            lambda: self.scene_Zslice.drawCurrentLine(frame=self.horizontalScrollBar_ManualEdit.value(),
                                                      current_layer=self.spinBox_CurrentLayer.value())
        )

        # --- Connections for Widget Synchronization (Unchanged) ---
        self.horizontalScrollBar_ManualEdit.valueChanged.connect(self.spinBox_FrameNumber.setValue)
        self.spinBox_FrameNumber.valueChanged.connect(self.horizontalScrollBar_ManualEdit.setValue)

        # --- Connections for Scene Configuration (Unchanged) ---
        # These configure the scene's behavior rather than redrawing it.
        self.checkBox_LineFit.stateChanged.connect(
            lambda: self.scene.set_line_fit_status(self.checkBox_LineFit.isChecked())
        )
        self.spinBox_FitBound.valueChanged.connect(
            lambda: self.scene.set_bounds(self.spinBox_FitBound.value())
        )
        self.radioButton_FitRPE.toggled.connect(
            lambda: self.scene.set_fit_mode(self.radioButton_FitRPE.isChecked(), self.radioButton_FitBM.isChecked())
        )
        self.radioButton_FitBM.toggled.connect(
            lambda: self.scene.set_fit_mode(self.radioButton_FitRPE.isChecked(), self.radioButton_FitBM.isChecked())
        )

    def disconnect_signals(self):
        """Disconnect signals to prevent multiple triggers."""
        try:
            self.spinBox_FrameNumber.valueChanged.disconnect()
            self.spinBox_CurrentLayer.valueChanged.disconnect()
            self.horizontalScrollBar_ManualEdit.valueChanged.disconnect()
        except TypeError:
            pass  # Ignore if already disconnected

    # --- MODIFICATION START: Updated keyPressEvent to exclude navigation from auto-save count ---
    def keyPressEvent(self, event):
        """Set keyboard shortcuts for navigation and editing."""
        key = event.key()
        # --- Navigation keys (A, D, Q, E) ---
        # These no longer trigger the auto-save counter.
        if key == Qt.Key_A:
            self.spinBox_FrameNumber.setValue(self.spinBox_FrameNumber.value() - 1)
        elif key == Qt.Key_D:
            self.spinBox_FrameNumber.setValue(self.spinBox_FrameNumber.value() + 1)
        elif key == Qt.Key_Q:
            self.spinBox_FrameNumber.setValue(self.spinBox_FrameNumber.value() - 10)
        elif key == Qt.Key_E:
            self.spinBox_FrameNumber.setValue(self.spinBox_FrameNumber.value() + 10)

        # --- Editing/Selection keys (W, S, Z, C) ---
        # These actions might imply a future modification, so they can increment the counter.
        elif key == Qt.Key_W:
            self.spinBox_CurrentLayer.setValue(self.spinBox_CurrentLayer.value() - 1)
            self.key_press_count_check()
        elif key == Qt.Key_S:
            self.spinBox_CurrentLayer.setValue(self.spinBox_CurrentLayer.value() + 1)
            self.key_press_count_check()
        elif key == Qt.Key_Z:
            self.spinBox_CopyLine.setValue(self.spinBox_CopyLine.value() - 1)
            self.key_press_count_check()
        elif key == Qt.Key_C:
            self.spinBox_CopyLine.setValue(self.spinBox_CopyLine.value() + 1)
            self.key_press_count_check()

        # --- Other shortcuts ---
        elif key == Qt.Key_Shift:
            self.checkBox_DisplayLines.nextCheckState()
    # --- MODIFICATION END ---


    def on_button_clicked_smooth(self):
        """Smoothing the current layer across all frames with improved stability."""
        self._last_shift_params = None  # Invalidate cache

        try:
            # Safely get kernel size with validation
            kernel_size = float(self.lineEdit_Kernel.text())
            if kernel_size <= 0:
                print("Warning: Kernel size must be positive.")
                return
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for the kernel size.")
            return

        layer_num = self.spinBox_CurrentLayer.value()

        # Preserve precision by converting to float for calculation and then rounding back
        original_dtype = img_obj.layers.dtype
        layer_data = img_obj.layers[:, :, layer_num].astype(float)

        smoothed_data = gaussian_filter(layer_data, sigma=(kernel_size, 1))

        # Clip to ensure values are within valid image bounds before converting back
        smoothed_data.clip(0, img_obj.img_depth - 1, out=smoothed_data)

        img_obj.layers[:, :, layer_num] = np.round(smoothed_data).astype(original_dtype)

        # Use the modern, safer method to update the scene
        self.scene.update_display()
        print(f"Layer {layer_num} smoothed with kernel size {kernel_size}.")

    def on_button_clicked_line_shift(self):
        """Memory-safe line shifting for the current layer."""
        layer_num = self.spinBox_CurrentLayer.value()
        line_shift_amt = self.spinBox_line_shift_amt.value()
        propagate_frames = self.spinBox_CopyLine.value()
        start_frame = self.spinBox_FrameNumber.value()

        # --- Add Input Validation ---
        if not hasattr(img_obj, 'layers') or img_obj.layers is None or img_obj.layers.shape[2] == 0:
            print("ERROR: No layers available to shift.")
            return
        if not (0 <= layer_num < img_obj.layer_num):
            print(f"ERROR: Invalid layer number {layer_num}. Must be between 0 and {img_obj.layer_num - 1}.")
            return
        if img_obj.img_framenum == 0:
            print("ERROR: No frames available to shift.")
            return
        if not (0 <= start_frame < img_obj.img_framenum):
            print(f"ERROR: Start frame {start_frame} is out of bounds for {img_obj.img_framenum} frames.")
            return
        # --- End Input Validation ---

        end_frame = min(start_frame + propagate_frames, img_obj.img_framenum)

        # If no frames are in the calculated range, just update the display and return.
        if start_frame >= end_frame:
            print(
                f"INFO: No frames to shift between {start_frame} and {end_frame}. (propagate_frames might be 0 or too small)")
            self.scene.update_display(frame=start_frame,
                                      layer=layer_num,
                                      is_draw_line=self.checkBox_DisplayLines.isChecked())
            return

        current_op_params = (layer_num, start_frame, end_frame)
        if self._last_shift_params != current_op_params:
            try:
                self._original_slice_for_shift = img_obj.layers[:, start_frame:end_frame, layer_num].copy()
                self._last_shift_params = current_op_params
            except IndexError as e:
                print(f"ERROR: Failed to retrieve original slice for shift (IndexError: {e}). "
                      f"Layer: {layer_num}, Frames: {start_frame}:{end_frame}, img_obj.layers shape: {img_obj.layers.shape}")
                return
            except Exception as e:  # Catch other potential issues
                print(f"ERROR: Unexpected error when retrieving original slice for shift: {e}")
                return

        if self._original_slice_for_shift is not None:
            shifted_slice = self._original_slice_for_shift + line_shift_amt
            shifted_slice.clip(0, img_obj.img_depth - 1, out=shifted_slice)

            try:
                img_obj.layers[:, start_frame:end_frame, layer_num] = shifted_slice
            except ValueError as e:
                print(f"ERROR: Shape mismatch during assignment after shift (ValueError: {e}). "
                      f"Original slice shape: {self._original_slice_for_shift.shape}, "
                      f"Shifted slice shape: {shifted_slice.shape}. Target slice shape in img_obj.layers "
                      f"would be ({img_obj.img_width}, {end_frame - start_frame}).")
                return
            except Exception as e:
                print(f"ERROR: Unexpected error during slice assignment: {e}")
                return

        self.scene.update_display(frame=start_frame,
                                  layer=layer_num,
                                  is_draw_line=self.checkBox_DisplayLines.isChecked())

    def key_press_count_check(self):
        """Auto-save the results after a certain number of actions."""
        self.keypress_count += 1
        if self.keypress_count >= 100:
            self.keypress_count = 0
            self.save_current_result()
            print('Auto-saving...')

    # --- MODIFICATION START: Updated save function to use a dedicated backup folder ---
    def save_current_result(self):
        """
        Saves the current segmentation results.
        Before saving, this function creates a timestamped backup of any existing
        layers file and moves it to a dedicated 'backups' subfolder.
        """
        # 1. Validate that a save path has been set.
        if not hasattr(img_obj, 'save_path') or not img_obj.save_path:
            QMessageBox.warning(self, "Save Error", "Save path is not set. Cannot save layers.")
            print("ERROR: Save path not set. Cannot save.")
            return

        # 2. Determine paths and create a dedicated backup directory.
        target_save_path = img_obj.save_path
        dirname, basename = os.path.split(target_save_path)
        base_name_no_ext, _ = os.path.splitext(basename)
        backup_dir = os.path.join(dirname, "backups")
        os.makedirs(backup_dir, exist_ok=True) # Creates the backup folder if it doesn't exist.

        # 3. Define the original files to process.
        files_to_process = {
            'npy': os.path.join(dirname, base_name_no_ext + '.npy'),
            'mat': os.path.join(dirname, base_name_no_ext + '.mat')
        }

        # 4. Create timestamped backups in the 'backups' folder.
        for ext, file_path in files_to_process.items():
            if os.path.exists(file_path):
                try:
                    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                    backup_filename = f"{base_name_no_ext}_before_modification_{timestamp}.{ext}"
                    backup_filepath = os.path.join(backup_dir, backup_filename) # Path is now inside the backup folder.

                    # Rename (move) the current file to the backup path.
                    os.rename(file_path, backup_filepath)
                    print(f"INFO: Created backup in 'backups' folder: {backup_filename}")
                except OSError as e:
                    print(f"ERROR: Could not create backup for {os.path.basename(file_path)}. Error: {e}")
                    reply = QMessageBox.question(self, 'Backup Failed',
                                                 f"Could not create a backup for {os.path.basename(file_path)}.\nDo you still want to save and overwrite it?",
                                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply == QMessageBox.No:
                        print("Save operation cancelled by user.")
                        return

        # 5. Proceed with saving the new (modified) data to the original path.
        try:
            img_obj.flatten_loc = int(self.spinBox_flatten_location.value())

            npy_path = files_to_process['npy']
            mat_path = files_to_process['mat']
            img_obj.save_layers(npy_path)
            img_obj.save_layers(mat_path)

            print(f'SUCCESS: Modified layers saved to:\n  - {npy_path}\n  - {mat_path}')

            if self.is_flattened_mode:
                self.save_triggered.emit()
            else:
                QMessageBox.information(self, "Save Successful", f"Results saved to:\n{base_name_no_ext}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save the layers file.\nError: {e}")
            print(f"ERROR: Failed to save layers file: {e}")
    # --- MODIFICATION END ---


from collections import deque
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPen, QBrush, QColor
from PyQt5.QtWidgets import QGraphicsScene


# Note: Ensure other necessary imports like numpy, interp1d, etc. are at the top of your file.

class GraphicsScene(QGraphicsScene):
    """
    The graphicsscene class for manual correction, with crash prevention and undo functionality.
    Supports two modes: 'point' and 'draw'.
    """

    def __init__(self, parent=None):
        super(GraphicsScene, self).__init__(parent)
        # --- Mode-agnostic variables ---
        self.copy_line_num = 1
        self.line_fit_status = True
        self.fit_bounds = 5
        self.fitRPE = True

        # --- Internal state for the current view ---
        self.frame = 0
        self.current_layer = 0
        self.is_draw_line_enabled = True

        # --- Mode-specific variables ---
        self.correction_mode = 'draw'
        self.correction_points = []
        self.zoomer = None
        self.drawing = False
        self.lastPoint = QPoint()
        self.new_line = None

        # --- MODIFICATION: Create one CLAHE filter to be reused ---
        # To achieve a strong effect, we use a higher clip limit and a moderate grid size.
        self.clahe_filter = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))

        self._undo_stack = deque(maxlen=20)
        self.update_display(frame=0, layer=0, is_draw_line=True)

    # --- Undo functionality ---
    def _push_to_undo_stack(self, frame, layer, x_min, x_max):
        """Saves the current state of a line segment before it's modified."""
        try:
            # We save the data for the current frame only for undo
            original_slice = img_obj.layers[x_min:x_max + 1, frame, layer].copy()
            self._undo_stack.append((frame, layer, x_min, x_max, original_slice))
            print(f"Undo state saved for frame {frame}, layer {layer}.")
        except IndexError as e:
            print(f"Warning: Could not save undo state. {e}")

    def undo_last_action(self):
        """Restores the last saved state from the undo stack."""
        if not self._undo_stack:
            print("Undo stack is empty.")
            return

        frame, layer, x_min, x_max, original_slice = self._undo_stack.pop()

        # Restore the data
        img_obj.layers[x_min:x_max + 1, frame, layer] = original_slice
        print(f"Undo successful. Restored data for frame {frame}, layer {layer}.")

        # Redraw the scene to show the change
        self.update_display()

    # --- Configuration Methods ---
    def set_correction_mode(self, mode):
        """Switches the mouse interaction mode between 'point' and 'draw'."""
        if mode in ['point', 'draw'] and self.correction_mode != mode:
            self.clear_correction_points()  # Clear state before switching
            if self.drawing: self.drawing = False
            self.correction_mode = mode

    def set_values(self, copy_num=1):
        self.copy_line_num = copy_num

    def set_line_fit_status(self, status=True):
        self.line_fit_status = status

    def set_bounds(self, bounds=5):
        self.fit_bounds = bounds

    def set_fit_mode(self, RPEmode, BMmode):
        self.fitRPE = RPEmode
        self.fitBM = BMmode

    def is_in_the_boundary(self, x, y):
        return 0 <= x < img_obj.img_width and 0 <= y < img_obj.img_depth

    # --- Mouse Event Handling ---
    def mousePressEvent(self, event):
        if self.correction_mode == 'point':
            self._handle_point_mode_press(event)
        elif self.correction_mode == 'draw':
            self._handle_draw_mode_press(event)

    def mouseMoveEvent(self, event):
        if self.correction_mode == 'draw' and self.drawing and (event.buttons() & Qt.LeftButton):
            pos = event.scenePos()
            self.addLine(self.lastPoint.x(), self.lastPoint.y(), pos.x(), pos.y(), QPen(Qt.green))
            self.lastPoint = pos
            x, y = int(pos.x()), int(pos.y())
            if self.is_in_the_boundary(x, y):
                self.new_line[x, :] = [x, y]
            self.update()

    def mouseReleaseEvent(self, event):
        if self.correction_mode == 'point':
            if event.button() == Qt.LeftButton and self.zoomer:
                self.zoomer.hide()
        elif self.correction_mode == 'draw':
            self._handle_draw_mode_release(event)

    # --- Helper methods for mouse events ---
    def _handle_point_mode_press(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.scenePos()
            if not self.is_in_the_boundary(pos.x(), pos.y()): return
            # --- MODIFICATION: Changed point marker color from cyan to blue ---
            marker = self.addEllipse(pos.x() - 2, pos.y() - 2, 4, 4, QPen(Qt.blue), QBrush(Qt.blue))
            # --- MODIFICATION END ---
            self.correction_points.append((pos, marker))
            self.show_zoomer(event)
        elif event.button() == Qt.RightButton:
            if not self.correction_points: return
            try:
                if len(self.correction_points) == 1:
                    self.apply_peak_valley_correction(self.correction_points[0][0])
                else:
                    self.apply_curve_correction()
            finally:
                self.clear_correction_points()
                self.update_display()

    def _handle_draw_mode_press(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.scenePos()
            self.new_line = np.full((img_obj.img_width, 2), np.nan)
            x, y = int(self.lastPoint.x()), int(self.lastPoint.y())
            if self.is_in_the_boundary(x, y):
                self.new_line[x, :] = [x, y]

    def _handle_draw_mode_release(self, event):
        if not self.drawing or event.button() != Qt.LeftButton:
            return

        self.drawing = False
        new_x, new_y = self.inter_the_line(self.new_line)

        if new_x is not None and new_y is not None:
            try:
                x_min, x_max = int(np.min(new_x)), int(np.max(new_x))
                self._push_to_undo_stack(self.frame, self.current_layer, x_min, x_max)
                if self.line_fit_status:
                    self.fit_lines(x_min, x_max, new_y)
                else:
                    self.change_line(x_min, x_max, new_y)
            except Exception as e:
                print(f"Error during line application logic: {e}")

        self.update_display()

    # --- Correction Logic ---
    def apply_peak_valley_correction(self, point_pos):
        x_click, y_click = int(point_pos.x()), int(point_pos.y())
        if not self.is_in_the_boundary(x_click, y_click): return

        influence_radius = 30
        x_min = max(0, x_click - influence_radius)
        x_max = min(img_obj.img_width - 1, x_click + influence_radius)

        self._push_to_undo_stack(self.frame, self.current_layer, x_min, x_max)

        num_of_copy = min(self.copy_line_num, img_obj.img_framenum - self.frame)
        x_range = np.arange(x_min, x_max + 1)
        weights = 0.5 * (1.0 + np.cos(np.pi * (np.abs(x_range - x_click) / influence_radius)))
        for frame_start in range(self.frame, self.frame + num_of_copy, MEMORY_FRIENDLY_BATCH_SIZE):
            frame_end = min(frame_start + MEMORY_FRIENDLY_BATCH_SIZE, self.frame + num_of_copy)
            layer_slice = img_obj.layers[x_min:x_max + 1, frame_start:frame_end, self.current_layer].copy()
            correction = (float(y_click) - layer_slice) * weights[:, np.newaxis]
            corrected_slice = layer_slice + correction
            corrected_slice.clip(0, img_obj.img_depth - 1, out=corrected_slice)
            img_obj.layers[x_min:x_max + 1, frame_start:frame_end, self.current_layer] = corrected_slice
        gc.collect()

    def apply_curve_correction(self):
        if len(self.correction_points) < 2: return
        points = sorted([(int(pos.x()), int(pos.y())) for pos, _ in self.correction_points])
        x, y = np.array(points).T
        unique_x, idx = np.unique(x, return_index=True)
        if len(unique_x) < 2: return

        x_min, x_max = int(np.min(unique_x)), int(np.max(unique_x))
        self._push_to_undo_stack(self.frame, self.current_layer, x_min, x_max)

        kind = 'linear' if len(unique_x) < 3 else 'quadratic' if len(unique_x) == 3 else 'cubic'
        f = interp1d(unique_x, y[idx], kind=kind, fill_value='extrapolate', bounds_error=False)
        x_interp = np.arange(x_min, x_max + 1)
        y_interp = f(x_interp)
        y_interp.clip(0, img_obj.img_depth - 1, out=y_interp)
        num_of_copy = min(self.copy_line_num, img_obj.img_framenum - self.frame)
        for frame_start in range(self.frame, self.frame + num_of_copy, MEMORY_FRIENDLY_BATCH_SIZE):
            frame_end = min(frame_start + MEMORY_FRIENDLY_BATCH_SIZE, self.frame + num_of_copy)
            num_frames_in_batch = frame_end - frame_start
            tiled_y = np.tile(y_interp, (num_frames_in_batch, 1)).T
            img_obj.layers[x_min:x_max + 1, frame_start:frame_end, self.current_layer] = tiled_y
        gc.collect()

    def clear_correction_points(self):
        """Safely removes correction point markers from the scene and clears the list."""
        for _, marker in self.correction_points:
            if marker.scene() == self:
                self.removeItem(marker)
        self.correction_points.clear()

    def show_zoomer(self, event):
        if self.zoomer is None: self.zoomer = ZoomerWidget()
        pos = event.scenePos()
        patch_size = self.zoomer.width() // 4
        img_data = img_obj.stru3d[:, :, self.frame]
        h, w = img_data.shape
        x, y = int(pos.x()), int(pos.y())
        x_start, y_start = max(0, x - patch_size // 2), max(0, y - patch_size // 2)
        patch = img_data[y_start:y_start + patch_size, x_start:x_start + patch_size]
        pixmap = get_pix_map_util(patch)
        zoomed_pixmap = pixmap.scaled(self.zoomer.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.zoomer.set_image(zoomed_pixmap)
        self.zoomer.move(event.screenPos().x() + 15, event.screenPos().y() + 15)
        self.zoomer.show()

    def fit_lines(self, x_min, x_max, new_y):
        num_of_copy = min(self.copy_line_num, img_obj.img_framenum - self.frame)
        for frame_start in range(self.frame, self.frame + num_of_copy, MEMORY_FRIENDLY_BATCH_SIZE):
            frame_end = min(frame_start + MEMORY_FRIENDLY_BATCH_SIZE, self.frame + num_of_copy)
            try:
                slices = img_obj.stru3d[:, x_min:x_max + 1, frame_start:frame_end].copy()
                slices = gaussian_filter(slices, (1, 0.5, 0.5), mode='nearest')
                for i in range(len(new_y)):
                    y_coord = int(new_y[i])
                    lower_bound = max(0, y_coord - self.fit_bounds)
                    upper_bound = min(img_obj.img_depth, y_coord + self.fit_bounds)
                    slices[:lower_bound, i, :] = 0
                    slices[upper_bound:, i, :] = 0
                new_loc = gaussian_filter(np.argmax(slices, axis=0), (3, 2), mode='nearest')
                img_obj.layers[x_min:x_max + 1, frame_start:frame_end, self.current_layer] = new_loc
                del slices, new_loc
                gc.collect()
            except MemoryError:
                print(f"Memory Error during fitting batch. Try reducing MEMORY_FRIENDLY_BATCH_SIZE.")
                return
        x_1, x_2 = max(0, x_min - 20), min(img_obj.img_width, x_max + 20)
        for frame_start in range(self.frame, self.frame + num_of_copy, MEMORY_FRIENDLY_BATCH_SIZE):
            frame_end = min(frame_start + MEMORY_FRIENDLY_BATCH_SIZE, self.frame + num_of_copy)
            sub_layer = img_obj.layers[x_1:x_2, frame_start:frame_end, self.current_layer]
            smoothed = gaussian_filter(gaussian_filter(sub_layer, (1, 0.5)), (3, 0.5))
            img_obj.layers[x_1:x_2, frame_start:frame_end, self.current_layer] = smoothed
        gc.collect()

    def change_line(self, x_min, x_max, new_y):
        num_of_copy = min(self.copy_line_num, img_obj.img_framenum - self.frame)
        for frame_start in range(self.frame, self.frame + num_of_copy, MEMORY_FRIENDLY_BATCH_SIZE):
            frame_end = min(frame_start + MEMORY_FRIENDLY_BATCH_SIZE, self.frame + num_of_copy)
            num_frames_in_batch = frame_end - frame_start
            tiled_y = np.tile(new_y, (num_frames_in_batch, 1)).T
            img_obj.layers[x_min:x_max + 1, frame_start:frame_end, self.current_layer] = tiled_y
        gc.collect()

    def inter_the_line(self, line):
        if line is None: return None, None
        x_coords, y_coords = line[:, 0], line[:, 1]
        valid_indices = ~np.isnan(x_coords)
        if np.sum(valid_indices) < 2: return None, None
        x_valid, y_valid = x_coords[valid_indices], y_coords[valid_indices]
        unique_x = np.unique(x_valid)
        avg_y = np.array([y_valid[x_valid == x].mean() for x in unique_x])
        if len(unique_x) < 2: return None, None
        try:
            kind = 'linear' if len(unique_x) < 4 else 'cubic'
            f = interp1d(unique_x, avg_y, kind=kind, fill_value="extrapolate", bounds_error=False)
            x_min, x_max = int(np.min(unique_x)), int(np.max(unique_x))
            x_new = np.arange(x_min, x_max + 1)
            y_new = f(x_new)
            return x_new, y_new
        except Exception as e:
            print(f"An unexpected error occurred during interpolation: {e}")
            return None, None

    # --- MODIFICATION START: update_display now handles CLAHE filtering and new line color ---
    def update_display(self, frame=None, layer=None, is_draw_line=None):
        self.clear_correction_points()
        if frame is not None: self.frame = frame
        if layer is not None: self.current_layer = layer
        if is_draw_line is not None: self.is_draw_line_enabled = is_draw_line
        self.clear()

        main_window = self.parent()
        if not main_window:
            return

        # 1. Get the original image data
        original_image = img_obj.stru3d[:, :, self.frame].copy()

        # 2. STAGE 1: Apply ImageJ-style contrast stretch
        contrast_min, contrast_max = main_window.contrastRangeSlider.value()

        processed_image = original_image.astype(np.float32)

        # --- This is the core logic for contrast stretching ---
        # Handle the case where min and max are the same to avoid division by zero
        if contrast_max == contrast_min:
            # If handles are at the same spot, make everything below it black, and above it white
            processed_image = np.where(processed_image < contrast_min, 0, 255)
        else:
            # Subtract the black point (min)
            processed_image = processed_image - contrast_min
            # Scale by the new range (max - min)
            processed_image = processed_image * (255.0 / (contrast_max - contrast_min))

        # Clip values to the 0-255 range and convert to 8-bit for display
        img_for_display = np.clip(processed_image, 0, 255).astype(np.uint8)

        # 3. STAGE 2: Apply CLAHE if checked (unchanged)
        if main_window.checkBox_applyCLAHE.isChecked():
            img_for_display = self.clahe_filter.apply(img_for_display)

        # 4. Add the final image to the scene (unchanged)
        self.addPixmap(get_pix_map_util(img_for_display))

        # 5. Draw segmentation lines (unchanged)
        if self.is_draw_line_enabled and img_obj.layer_num > 0:
            # ... (rest of the line drawing code is exactly the same) ...
            config = Config()
            img_w = img_obj.layers.shape[0]
            inds = list(range(img_obj.layer_num))
            lines_data = np.squeeze(img_obj.layers[:, self.frame, inds])
            if lines_data.ndim == 1: lines_data = lines_data[:, np.newaxis]
            pen_all = QPen(Qt.red, 1.0)
            pen_current = QPen(QColor(0, 0, 128), 1.5)

            if config.is_distorted:
                pen_all.setWidthF(config.dist_fact * 1.5)
                pen_current.setWidthF(config.dist_fact * 1.6)
            for i in range(lines_data.shape[1]):
                for j in range(img_w - 1):
                    self.addLine(j, lines_data[j, i], j + 1, lines_data[j + 1, i], pen_all)
            for j in range(img_w - 1):
                self.addLine(j, lines_data[j, self.current_layer], j + 1, lines_data[j + 1, self.current_layer],
                             pen_current)


class GraphicsSceneColormap(QGraphicsScene):
    """Scene for displaying the thickness map of the current layer."""

    def __init__(self, parent=None):
        super(GraphicsSceneColormap, self).__init__(parent)
        # self.drawCurrentLine(0, 0)

    def drawCurrentLine(self, frame=0, current_layer=0):
        self.clear()
        img_thickness = img_obj.layers[:, :, current_layer].T.copy()
        min_val, max_val = img_thickness.min(), img_thickness.max()
        if max_val > min_val:
            img_thickness = (img_thickness - min_val) / (max_val - min_val) * 255
        img_color = get_cmap('jet')(np.uint8(img_thickness))
        self.addPixmap(get_pix_map_util(img_color, normalize=True))
        self.addLine(0, frame, img_obj.img_framenum, frame, QPen(Qt.white, 1))


class GraphicsSceneZslice(QGraphicsScene):
    """Scene for displaying the en-face projection (Z-slice)."""

    def __init__(self, parent=None):
        super(GraphicsSceneZslice, self).__init__(parent)
        # self.drawCurrentLine(0, 0)

    def drawCurrentLine(self, frame=0, current_layer=0):
        self.clear()
        img_proj = img_obj.plot_proj(current_layer, current_layer, 'stru', 'max', start_offset=0, end_offset=1,
                                     display=False, rotate=True)
        self.addPixmap(get_pix_map_util(img_proj, normalize=True))
        self.addLine(0, frame, img_obj.img_framenum, frame, QPen(Qt.white, 1))