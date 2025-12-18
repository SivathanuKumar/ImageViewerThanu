from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import qimage2ndarray
from Utils.data_class import state

# Import your segmentation functions
from Analysis.SM.layer_seg_parallel import seg_video_SD_parallel
from Analysis.SM.ORL_Segmentation_UNet_reserve2 import ORL_segmentation_UNet_ch


# Import other segmentation functions as needed...

class SegWizardWindow(QtWidgets.QDialog):
    def __init__(self, device_type, scan_size, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Layer Segmentation Wizard")
        self.resize(1000, 700)
        self.img_obj = state.img_obj
        self.device_type = device_type
        self.scan_size = scan_size

        # UI Layout
        layout = QtWidgets.QHBoxLayout()

        # --- Left Panel: Controls ---
        left_panel = QtWidgets.QVBoxLayout()

        # Info Box
        info_group = QtWidgets.QGroupBox("Acquisition Info")
        info_layout = QtWidgets.QVBoxLayout()
        info_layout.addWidget(QtWidgets.QLabel(f"Device: {self.device_type}"))
        info_layout.addWidget(QtWidgets.QLabel(f"Size: {self.scan_size}"))
        info_group.setLayout(info_layout)
        left_panel.addWidget(info_group)

        # Layer Selection
        self.layer_group = QtWidgets.QGroupBox("Select Layers to Segment")
        self.layer_layout = QtWidgets.QVBoxLayout()

        # Define checkboxes
        self.cb_ilm = QtWidgets.QCheckBox("ILM (Inner Limiting Membrane)")
        self.cb_rpe = QtWidgets.QCheckBox("RPE (Retinal Pigment Epithelium)")
        self.cb_bm = QtWidgets.QCheckBox("BM (Bruch's Membrane)")
        self.cb_orl = QtWidgets.QCheckBox("ORL (Outer Retinal Layers)")
        self.cb_ipl = QtWidgets.QCheckBox("IPL (Inner Plexiform Layer)")

        # Default checks based on device type (Logic Customization)
        self.cb_ilm.setChecked(True)
        self.cb_rpe.setChecked(True)
        if "SD-OCT" in self.device_type:
            self.cb_bm.setChecked(True)

        self.layer_layout.addWidget(self.cb_ilm)
        self.layer_layout.addWidget(self.cb_rpe)
        self.layer_layout.addWidget(self.cb_bm)
        self.layer_layout.addWidget(self.cb_orl)
        self.layer_layout.addWidget(self.cb_ipl)
        self.layer_group.setLayout(self.layer_layout)
        left_panel.addWidget(self.layer_group)

        # Method Preview (Text)
        self.method_label = QtWidgets.QLabel("Method: Auto-Selection")
        self.method_label.setWordWrap(True)
        left_panel.addWidget(self.method_label)

        # Action Buttons
        self.btn_run = QtWidgets.QPushButton("Run Segmentation")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.run_segmentation_logic)

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        left_panel.addStretch()
        left_panel.addWidget(self.btn_run)
        left_panel.addWidget(self.btn_cancel)

        # --- Right Panel: Graphics View (B-Scan Preview) ---
        right_panel = QtWidgets.QVBoxLayout()
        self.graphics_view = QtWidgets.QGraphicsView()
        self.scene = QtWidgets.QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        right_panel.addWidget(QtWidgets.QLabel("B-Scan Preview (Middle Slice)"))
        right_panel.addWidget(self.graphics_view)

        # Assemble Main Layout
        layout.addLayout(left_panel, 1)  # Ratio 1
        layout.addLayout(right_panel, 3)  # Ratio 3
        self.setLayout(layout)

        # Initialize
        self.update_method_label()
        self.load_preview()

    def update_method_label(self):
        """Updates text based on what algorithm will be used"""
        if "SD-OCT" in self.device_type:
            self.method_label.setText(f"Method: Standard Graph Search\nTarget: {self.device_type}")
        elif "ZEISS" in self.device_type or "Intalight" in self.device_type:
            self.method_label.setText(f"Method: U-Net Deep Learning\nTarget: {self.device_type} ({self.scan_size})")
        else:
            self.method_label.setText("Method: Standard Fallback")

    def load_preview(self):
        """Loads the middle frame B-scan into the graphics view"""
        if self.img_obj.exist_stru:
            mid_frame = self.img_obj.stru3d.shape[2] // 2
            img = self.img_obj.stru3d[:, :, mid_frame]

            # Normalize for display (uint16 to uint8)
            img = img / np.max(img) * 255
            img = img.astype(np.uint8)

            qimg = qimage2ndarray.array2qimage(img)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.scene.clear()
            self.scene.addPixmap(pixmap)
            self.graphics_view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        else:
            self.scene.addText("No Structural Data Loaded")

    def run_segmentation_logic(self):
        """
        The Core Logic: Dispatches to specific functions based on Device/Size/Checkboxes
        """
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Processing...")
        QtWidgets.QApplication.processEvents()  # Keep UI responsive

        try:
            # 1. Prepare Parameters
            paras = {
                'retina_lower_offset': 50,  # You can make these inputs in the wizard too
                'retina_upper_offset': 150,
                'upper_bound': 5,
                'lower_bound': 5
            }

            # 2. Dispatcher Logic

            # CASE A: SD-OCT (Standard Algorithms)
            if self.device_type == "SD-OCT":
                print("Running Standard SD-OCT Segmentation...")
                # Call function from code snippet 1
                self.img_obj.layers = seg_video_SD_parallel(self.img_obj, retina_th=paras['retina_lower_offset'])
                self.img_obj.layer_num = self.img_obj.layers.shape[2]
                self.img_obj.exist_seg = True

            # CASE B: Deep Learning (Zeiss/Intalight/SS-OCT)
            elif self.device_type in ["ZEISS PLEX Elite 9000", "ZEISS Cirrus 6000", "Intalight DREAM OCT", "SS-OCT"]:
                print(f"Running U-Net Segmentation for {self.device_type}...")

                # Check which specific U-Net to run based on checkboxes
                if self.cb_orl.isChecked():
                    # Call function from code snippet 2
                    # Note: You might need to adjust parameters passed to match exact function signature
                    layers_unet = ORL_segmentation_UNet_ch(self.img_obj, flag=1)

                    # Map the U-Net output (usually [Width, Frames, Layers]) to the main object
                    # Assuming layers_unet returns shape (Width, Frames, 2) for ORL/RPE
                    X, Z, num_layers = self.img_obj.stru3d.shape[1], self.img_obj.stru3d.shape[2], 2

                    # Create placeholder if it doesn't exist, or overwrite
                    full_layers = np.zeros((X, Z, 4))  # Example: ILM, RPE, ORL_1, ORL_2

                    # Fill in the specific layers returned by U-Net
                    # This mapping depends on exactly what your UNet returns
                    full_layers[:, :, 0] = layers_unet[:, :, 0]  # Example mapping
                    full_layers[:, :, 1] = layers_unet[:, :, 1]

                    self.img_obj.layers = full_layers
                    self.img_obj.layer_num = 4
                    self.img_obj.exist_seg = True

            # 3. Finalize
            print("Segmentation Complete.")
            self.accept()  # Closes window with "Success" result

        except Exception as e:
            print(f"Error during segmentation: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            self.btn_run.setEnabled(True)
            self.btn_run.setText("Run Segmentation")