import numpy as np
import os
import time
from PyQt5 import QtWidgets, QtCore, QtGui
import qimage2ndarray

# Import your existing utilities
from Utils.data_class import state
from Utils.OCTpy import Oct3Dimage

# Import the logic classes
# Ensure these imports match where you save the new unified file
from Analysis.SM.Unified_Segmentation import UnifiedSegmentationManager

PROJECT_ROOT = os.getcwd()
SAMPLE_STRU_PATH = os.path.join(PROJECT_ROOT, "Reference_Data", "ZeissAttnStruc.avi")
SAMPLE_SEG_PATH = os.path.join(PROJECT_ROOT, "Reference_Data", "SDKlayers.mat")


class PreviewScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.img_data = None
        self.layer_data = None
        self.active_layer_indices = []
        self.available_indices = []

    def set_data(self, img_data, layer_data):
        self.img_data = img_data
        self.layer_data = layer_data
        if self.layer_data is not None:
            self.available_indices = list(range(self.layer_data.shape[1]))
        self.update_view()

    def set_active_layers(self, indices):
        self.active_layer_indices = indices
        self.update_view()

    def update_view(self):
        self.clear()
        if self.img_data is not None:
            # Normalize and display image
            disp_img = self.img_data.astype(float)
            if disp_img.max() > 0:
                disp_img = disp_img / disp_img.max() * 255
            disp_img = disp_img.astype(np.uint8)

            qimg = qimage2ndarray.array2qimage(disp_img)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.addPixmap(pixmap)

        if self.layer_data is not None:
            width = self.layer_data.shape[0]
            # Define Colors
            pen_selected = QtGui.QPen(QtCore.Qt.cyan, 2.0)  # Changed to Cyan for visibility
            pen_unselected = QtGui.QPen(QtGui.QColor(255, 0, 0, 50), 1.0)  # Transparent Red

            for layer_idx in self.available_indices:
                current_pen = pen_selected if layer_idx in self.active_layer_indices else pen_unselected

                y_coords = self.layer_data[:, layer_idx]

                # Optimized drawing: create a path instead of individual lines for speed
                path = QtGui.QPainterPath()
                started = False
                for x in range(width):
                    y = y_coords[x]
                    if np.isnan(y) or y <= 0:
                        started = False
                        continue

                    if not started:
                        path.moveTo(x, y)
                        started = True
                    else:
                        path.lineTo(x, y)

                self.addPath(path, current_pen)


class SegWizardWindow(QtWidgets.QDialog):
    def __init__(self, device_type, scan_size, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Layer Segmentation Wizard")
        self.resize(1100, 650)  # Slightly tighter initial size

        self.img_obj = state.img_obj
        self.device_type = device_type
        self.scan_size = scan_size  # e.g., "6x6", "12x12", "3x3"

        self.layer_map = {'ILM': 0, 'IPL': 1, 'OPL': 2, 'ONL': 3, 'RPE': 4, 'BM': 5, 'CSI': 6}
        self.checkboxes = {}

        self.init_ui()
        QtCore.QTimer.singleShot(100, self.load_sample_data)

    def init_ui(self):
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # --- LEFT PANEL (Controls) ---
        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setAlignment(QtCore.Qt.AlignTop)  # Align everything to top

        # 1. Info Box
        info_group = QtWidgets.QGroupBox("Scan Info")
        info_layout = QtWidgets.QVBoxLayout()
        info_layout.addWidget(QtWidgets.QLabel(f"Device: {self.device_type}"))
        info_layout.addWidget(QtWidgets.QLabel(f"Scan Size: {self.scan_size}"))
        info_group.setLayout(info_layout)
        left_panel.addWidget(info_group)

        # 2. Selection Group (Compact)
        sel_group = QtWidgets.QGroupBox("Target Layers")
        sel_layout = QtWidgets.QVBoxLayout()
        sel_layout.setSpacing(5)

        layers = ['ILM', 'IPL', 'OPL', 'ONL', 'RPE', 'BM', 'CSI']
        defaults = ['ILM', 'RPE', 'CSI']

        for name in layers:
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(name in defaults)
            cb.stateChanged.connect(self.refresh_preview)
            self.checkboxes[name] = cb
            sel_layout.addWidget(cb)

        # Logic check for checkboxes based on device
        if self.device_type == "SD-OCT" or self.scan_size == "3x3" or self.scan_size == "12x12":
            # ML specific layers might be disabled or warned about
            self.checkboxes['IPL'].setToolTip("Not available for this scan type (ML Restricted)")

        sel_group.setLayout(sel_layout)
        left_panel.addWidget(sel_group)

        # Spacer to push buttons down
        left_panel.addStretch(1)

        # 3. Progress Section
        self.lbl_progress = QtWidgets.QLabel("Ready")
        left_panel.addWidget(self.lbl_progress)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        left_panel.addWidget(self.progress_bar)

        # 4. Action Buttons
        self.btn_run = QtWidgets.QPushButton("Run Segmentation")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.btn_run.clicked.connect(self.run_segmentation_logic)
        left_panel.addWidget(self.btn_run)

        # Wrap Left Panel in a Frame for visual separation
        left_frame = QtWidgets.QFrame()
        left_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        left_frame.setLayout(left_panel)
        left_frame.setFixedWidth(250)  # Fixed width for tidiness

        # --- RIGHT PANEL (Graphics) ---
        right_panel = QtWidgets.QVBoxLayout()
        self.graphics_view = QtWidgets.QGraphicsView()
        self.graphics_view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.graphics_view.setStyleSheet("background-color: black;")
        self.scene = PreviewScene(self)
        self.graphics_view.setScene(self.scene)
        right_panel.addWidget(self.graphics_view)

        # Add to main
        main_layout.addWidget(left_frame)
        main_layout.addLayout(right_panel, 1)  # Give priority to image
        self.setLayout(main_layout)

    def load_sample_data(self):
        # (Same as your original code, kept for brevity)
        if not os.path.exists(SAMPLE_STRU_PATH) or not os.path.exists(SAMPLE_SEG_PATH):
            return

        try:
            sample_obj = Oct3Dimage()
            sample_obj.read_stru_data(SAMPLE_STRU_PATH)
            sample_obj.read_seg_layers(SAMPLE_SEG_PATH)
            mid_frame_idx = sample_obj.img_framenum // 2
            frame_img = sample_obj.stru3d[:, :, mid_frame_idx]

            if sample_obj.exist_seg and sample_obj.layers is not None:
                frame_layers = sample_obj.layers[:, mid_frame_idx, :]
            else:
                frame_layers = None

            self.scene.set_data(frame_img, frame_layers)
            self.refresh_preview()
            self.graphics_view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

        except Exception as e:
            print(f"Error loading sample: {e}")

    def refresh_preview(self):
        selected_indices = []
        for name, index in self.layer_map.items():
            if name in self.checkboxes and self.checkboxes[name].isChecked():
                selected_indices.append(index)
        self.scene.set_active_layers(selected_indices)

    def update_progress_ui(self, value, message=None):
        self.progress_bar.setValue(int(value))
        if message:
            self.lbl_progress.setText(message)
        QtWidgets.QApplication.processEvents()  # Force UI update

    def run_segmentation_logic(self):
        if not self.img_obj.exist_stru:
            QtWidgets.QMessageBox.warning(self, "Error", "No Patient Data Loaded")
            return

        self.btn_run.setEnabled(False)
        self.update_progress_ui(0, "Initializing...")

        try:
            # 1. Collect Selections
            selections = {k: v.isChecked() for k, v in self.checkboxes.items()}

            # 2. Initialize Manager
            seg_manager = UnifiedSegmentationManager(PROJECT_ROOT)

            # 3. Run with Callback
            start_time = time.time()
            computed_layers, report_str = seg_manager.run_segmentation(
                self.img_obj,
                self.device_type,
                self.scan_size,
                selections,
                progress_callback=self.update_progress_ui
            )

            # 4. Save to Object
            self.img_obj.layers = computed_layers
            self.img_obj.layer_num = 7
            self.img_obj.exist_seg = True

            self.update_progress_ui(100, "Completed")

            # 5. Confirmation Popup
            elapsed = time.time() - start_time
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Segmentation Complete")
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText(f"Processing finished in {elapsed:.1f} seconds.")
            msg.setInformativeText("<b>Layer Source Report:</b><br>" + report_str.replace('\n', '<br>'))
            msg.exec_()

            self.accept()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Processing Error", str(e))
            self.btn_run.setEnabled(True)
            self.lbl_progress.setText("Failed")