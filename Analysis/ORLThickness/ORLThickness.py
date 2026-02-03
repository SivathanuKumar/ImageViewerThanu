import sys
import os
import glob
import numpy as np
import cv2
import scipy.io
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
from scipy.ndimage import median_filter, gaussian_filter, center_of_mass
import qimage2ndarray
import io  # Standard library (contains StringIO)
from skimage import io as skio

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QPen, QColor
from PyQt5.QtCore import Qt, pyqtSignal

# -------------------------------------------------------------------------
# 1. SETUP PATHS & UI LOADING
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if root_dir not in sys.path: sys.path.append(root_dir)
if current_dir not in sys.path: sys.path.append(current_dir)

# Define path to the .ui file
UI_FILE_PATH = os.path.join(current_dir, r".\thicknessmapui.ui")  # Ensure filename matches yours

if not os.path.exists(UI_FILE_PATH):
    raise FileNotFoundError(f"UI file not found at: {UI_FILE_PATH}")

with open(UI_FILE_PATH, 'r', encoding='utf-8') as f:
    ui_xml = f.read()

ui_xml = ui_xml.replace('Qt::AlignmentFlag::', 'Qt::')
ui_xml = ui_xml.replace('Qt::Orientation::', 'Qt::')
ui_xml = ui_xml.replace('Qt::WindowType::', 'Qt::')
ui_xml = ui_xml.replace('QFrame::Shadow::', 'QFrame::')
f_io = io.StringIO(ui_xml)
Ui_MainWindow, QtBaseClass = uic.loadUiType(f_io)


# -------------------------------------------------------------------------
# 2. DUMMY CLASSES FOR STANDALONE MODE
# -------------------------------------------------------------------------
class DummyData:
    """ Mock object to mimic the data structure when running standalone """

    def __init__(self):
        self.exist_stru = False
        self.exist_seg = False
        self.img_width = 512
        self.img_depth = 512
        self.img_framenum = 1
        self.layer_num = 0
        self.url_stru = ""
        self.url_seg = ""
        self.stru3d = np.zeros((512, 512, 1), dtype=np.uint8)
        self.layers = None  # Will init on load

    def read_stru_data(self, filepath):
        # In a real scenario, you'd implement actual loading here (e.g., SimpleITK or custom)
        # For this standalone demo, we just simulate successful load
        self.url_stru = filepath
        self.exist_stru = True
        self.img_width = 500
        self.img_framenum = 500
        self.img_depth = 1536
        # Create dummy noise data
        self.stru3d = np.random.randint(0, 255, (1536, 500, 1), dtype=np.uint8)
        print(f"Standalone: Mock Structure Loaded from {filepath}")

    def read_seg_layers(self, filepath):
        self.url_seg = filepath
        self.exist_seg = True
        # Create dummy layers (Width x Frames x 11 layers)
        # Layer 0 approx at pixel 200, Layer 1 at pixel 300
        self.layer_num = 11
        self.layers = np.zeros((500, 500, 11))
        for i in range(11):
            self.layers[:, :, i] = 200 + (i * 20)
        print(f"Standalone: Mock Layers Loaded from {filepath}")

    def plot_proj(self, layer1, layer2, type, projmethod, start_offset, end_offset, display, rotate):
        # Mock projection generator
        return np.random.randint(0, 255, (500, 500), dtype=np.uint8)


# -------------------------------------------------------------------------
# 3. CUSTOM SCENE (INTERACTIVE CLICKING)
# -------------------------------------------------------------------------
class ClickableGraphicsScene(QGraphicsScene):
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.scenePos()
            x, y = int(pos.x()), int(pos.y())
            self.clicked.emit(x, y)
        super().mousePressEvent(event)

    def update_cursor(self, x, y):
        # Remove old cursor if exists
        for item in self.items():
            if hasattr(item, 'is_cursor'):
                self.removeItem(item)

        pen = QPen(Qt.red)
        pen.setWidth(2)

        # Draw Crosshair
        size = 15
        path = QtWidgets.QGraphicsPathItem()
        path.is_cursor = True  # Tag
        p = path.path()
        p.moveTo(x - size, y)
        p.lineTo(x + size, y)
        p.moveTo(x, y - size)
        p.lineTo(x, y + size)
        path.setPath(p)
        path.setPen(pen)
        self.addItem(path)


# -------------------------------------------------------------------------
# 4. MAIN WINDOW CLASS
# -------------------------------------------------------------------------
class ORLThicknessWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, img_obj=None):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.horizontalScrollBar.setOrientation(Qt.Horizontal)


        # --- Handle Data Object ---
        if img_obj is None:
            print("Mode: STANDALONE")
            self.img_obj = DummyData()
            self.is_standalone = True
        else:
            print("Mode: INTEGRATED")
            self.img_obj = img_obj
            self.is_standalone = False
            # Pre-fill UI if data exists
            if hasattr(self.img_obj, 'url_stru'):
                self.lineEdit_loadstruct.setText(str(self.img_obj.url_stru))
            if hasattr(self.img_obj, 'url_seg'):
                self.lineEdit_loadseg.setText(str(self.img_obj.url_seg))

        # --- Scene Initialization ---
        self.scene = QGraphicsScene()
        self.graphicsView_Bscan.setScene(self.scene)

        self.scene_thickness = QGraphicsScene()
        self.graphicsView_thicknessmap_wo_mask.setScene(self.scene_thickness)

        # Interactive Projection Scene
        self.scene_proj = ClickableGraphicsScene()
        self.scene_proj.clicked.connect(self.on_proj_clicked)
        self.graphicsView_ORLProj.setScene(self.scene_proj)

        self.scene_cd = QGraphicsScene()
        self.graphicsView_thicknessmap_w_mask.setScene(self.scene_cd)

        # --- Connections ---
        self.pushButton_loadstruct.clicked.connect(self.on_button_load_struct)
        self.pushButton_loadseg.clicked.connect(self.on_button_load_seg)
        self.pushButton_Compute_and_Display.clicked.connect(self.compute_ORL_thickness)
        self.pushButton_save_csv.clicked.connect(self.on_button_clicked_save_results)

        # New Feature Connections
        # Check if widgets exist (in case UI file isn't updated yet)
        if hasattr(self, 'comboBox_MaskList'):
            self.comboBox_MaskList.currentIndexChanged.connect(self.on_mask_selected)
        if hasattr(self, 'pushButton_AutoFovea'):
            self.pushButton_AutoFovea.clicked.connect(self.auto_detect_fovea)

        # Setup Spinboxes
        self.spinBox_2.valueChanged.connect(self.update_cursor_from_spinbox)
        self.spinBox_3.valueChanged.connect(self.update_cursor_from_spinbox)

        self.fovea_center = None
        self.cdlayer = None
        self.thickness_data = None
        self.set_thickness_data_frame()

        # If running integrated and data is ready, trigger load immediately
        if not self.is_standalone and self.img_obj.exist_stru:
            self.refresh_ui_state()

    # --------------------------------------------------------------------
    # DATA LOADING & PREP
    # --------------------------------------------------------------------
    def on_button_load_struct(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Structure File", "", "Images (*.avi *.dcm *.img *.vol)")
        if file_path:
            self.lineEdit_loadstruct.setText(file_path)
            self.img_obj.read_stru_data(file_path)

            if self.img_obj.exist_stru:
                self.send_log(f"Structure loaded: {os.path.basename(file_path)}")

                # Init dummy layers if they don't exist yet
                if not self.img_obj.exist_seg or self.img_obj.layers is None:
                    self.img_obj.layers = np.zeros((self.img_obj.img_width, self.img_obj.img_framenum, 1))
                    self.img_obj.layer_num = 1

                self.populate_mask_list()
                self.refresh_ui_state()

    def on_button_load_seg(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Segmentation File", "", "Files (*.txt *.mat *.npy)")
        if file_path:
            self.lineEdit_loadseg.setText(file_path)
            self.img_obj.read_seg_layers(file_path)
            if self.img_obj.exist_seg:
                self.send_log("Segmentation loaded.")
                self.refresh_ui_state()

    def refresh_ui_state(self):
        """ Called whenever data changes. Updates ranges and plots projection. """
        if self.img_obj.img_width == 0: return

        # Set Ranges
        self.spinBox.setMaximum(self.img_obj.img_framenum - 1)
        self.horizontalScrollBar.setMaximum(self.img_obj.img_framenum - 1)

        max_layer = max(0, self.img_obj.layer_num - 1)
        self.spinBox_CurrentLayer.setMaximum(max_layer)
        self.spinBox_StartLayer.setMaximum(max_layer)
        self.spinBox_EndLayer.setMaximum(max_layer)

        # Connect Scrollbar to Spinbox
        try:
            self.horizontalScrollBar.valueChanged.disconnect()
        except:
            pass  # disconnect all previous to avoid duplicates

        self.horizontalScrollBar.valueChanged.connect(self.spinBox.setValue)
        self.spinBox.valueChanged.connect(self.horizontalScrollBar.setValue)

        # Connect drawing of B-scan
        self.horizontalScrollBar.valueChanged.connect(self.draw_bscan)

        # AUTO-PLOT PROJECTION (Feature Request 1)
        self.generate_and_plot_projection()

    def generate_and_plot_projection(self):
        """ Automatically generates projection from structure data """
        if not self.img_obj.exist_stru: return

        self.send_log("Generating Auto-Projection...")
        try:
            # Use layer 0 or 0-10 if layers exist, else entire depth
            # Simplified: Just project the structure sum
            if hasattr(self.img_obj, 'plot_proj'):
                # Use existing method if available (Integrated mode)
                self.img_proj_stru = self.img_obj.plot_proj(
                    0, 0, 'stru', projmethod='sum', start_offset=-10, end_offset=10, display=False, rotate=False
                )
            else:
                # Standalone fallback
                self.img_proj_stru = np.mean(self.img_obj.stru3d, axis=0).astype(np.uint8)

            # Process image for display
            img = self.img_proj_stru
            if img.max() > 0: img = (img / img.max() * 255).astype(np.uint8)

            # Rotate/Flip to match convention
            img = cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1)

            # Display
            pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(img))
            self.scene_proj.clear()
            self.scene_proj.addPixmap(pixmap)

            # Restore cursor if exists
            if self.fovea_center:
                self.scene_proj.update_cursor(*self.fovea_center)

            self.graphicsView_ORLProj.fitInView(self.scene_proj.sceneRect(), Qt.KeepAspectRatio)

        except Exception as e:
            self.send_log(f"Auto-Projection Error: {e}")

    # --------------------------------------------------------------------
    # MASK & FOVEA FEATURES
    # --------------------------------------------------------------------
    def populate_mask_list(self):
        if not hasattr(self, 'comboBox_MaskList'): return
        if not self.img_obj.url_stru: return

        parent_dir = Path(self.img_obj.url_stru).parent

        # Look for files with 'mask' in name
        patterns = ["*mask*.png", "*mask*.tif", "*mask*.jpg", "*mask*.mat"]
        mask_files = []
        for p in patterns:
            mask_files.extend(glob.glob(os.path.join(parent_dir, p)))
            mask_files.extend(glob.glob(os.path.join(parent_dir, p.upper())))

        self.comboBox_MaskList.clear()
        self.comboBox_MaskList.addItem("No Mask Selected", None)

        for f in mask_files:
            self.comboBox_MaskList.addItem(os.path.basename(f), f)

        self.send_log(f"Found {len(mask_files)} mask candidates.")

    def on_mask_selected(self, index):
        if index <= 0:
            self.cdlayer = None
            self.send_log("Mask cleared.")
            return

        mask_path = self.comboBox_MaskList.itemData(index)
        try:
            target_shape = (self.img_obj.img_width, self.img_obj.img_framenum)
            if mask_path.endswith('.mat'):
                mat = scipy.io.loadmat(mask_path)
                key = [k for k in mat.keys() if not k.startswith('_')][0]
                layer = mat[key]
                if layer.ndim == 3: layer = layer[:, :, 0]
            else:
                layer = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            layer = cv2.resize(layer, target_shape, interpolation=cv2.INTER_NEAREST)
            self.cdlayer = layer
            self.send_log(f"Mask Loaded: {os.path.basename(mask_path)}")
        except Exception as e:
            self.send_log(f"Mask Load Error: {e}")

    def on_proj_clicked(self, x, y):
        # Update spinboxes
        self.spinBox_2.blockSignals(True)
        self.spinBox_3.blockSignals(True)
        self.spinBox_2.setValue(x)
        self.spinBox_3.setValue(y)
        self.spinBox_2.blockSignals(False)
        self.spinBox_3.blockSignals(False)

        self.fovea_center = (x, y)
        self.scene_proj.update_cursor(x, y)

    def update_cursor_from_spinbox(self):
        x = self.spinBox_2.value()
        y = self.spinBox_3.value()
        self.fovea_center = (x, y)
        self.scene_proj.update_cursor(x, y)

    def auto_detect_fovea(self):
        if not hasattr(self, 'img_proj_stru'):
            self.send_log("No projection data for auto-detect.")
            return

        try:
            # Simple Center of Mass
            cy, cx = center_of_mass(self.img_proj_stru)
            self.on_proj_clicked(int(cx), int(cy))
            self.send_log(f"Auto-Fovea: {int(cx)}, {int(cy)}")
        except Exception as e:
            self.send_log(f"Auto-Detect Failed: {e}")

    # --------------------------------------------------------------------
    # COMPUTATION LOGIC
    # --------------------------------------------------------------------
    def compute_ORL_thickness(self):
        if self.img_obj.layers is None: return

        # Inputs
        start = self.spinBox_StartLayer.value()
        end = self.spinBox_EndLayer.value()
        axial = self.doubleSpinBox_Axial.value() if hasattr(self, 'doubleSpinBox_Axial') else 1.96
        is_rpebm = self.checkBox_RPEBM.isChecked() if hasattr(self, 'checkBox_RPEBM') else False

        self.send_log(f"Computing: L{start}-L{end}, Spacing={axial}, RPE-BM={is_rpebm}")

        # Calc Raw Difference (Pixels)
        diff_px = abs(self.img_obj.layers[:, :, start] - self.img_obj.layers[:, :, end])

        # Convert to Microns and Transpose for map view
        # Map view is usually (Frames x Width)
        map_um = np.transpose(diff_px * axial)
        map_um = median_filter(gaussian_filter(map_um, sigma=(5, 1)), (1, 3))

        # Metrics Setup
        self.thickness_data = pd.DataFrame(
            columns=['All', '1mm', '3mm', '5mm'],
            index=['With Mask', 'W/O Mask', 'Fovea (X,Y)']
        )
        if is_rpebm:
            self.thickness_data.loc['Drusen Area (mm2)'] = [0] * 4
            self.thickness_data.loc['Drusen Vol (mm3)'] = [0] * 4

        # Masks for ETDRS circles
        h, w = map_um.shape
        scan_width_mm = 6.0
        px_per_mm = w / scan_width_mm  # Approx

        mask1 = self.create_circular_mask(h, w, radius=(0.5 * px_per_mm))
        mask3 = self.create_circular_mask(h, w, radius=(1.5 * px_per_mm))
        mask5 = self.create_circular_mask(h, w, radius=(2.5 * px_per_mm))

        masks = [np.ones_like(map_um, dtype=bool), mask1, mask3, mask5]

        # Calculate Basic Thickness
        vals_nomask = []
        vals_mask = []

        # External Mask Handling
        ext_mask = None
        if hasattr(self, 'cdlayer') and self.cdlayer is not None:
            # Resize mask to match map
            ext_mask = cv2.resize(self.cdlayer, (w, h), interpolation=cv2.INTER_NEAREST).T

        for m in masks:
            # W/O Mask
            tmp = map_um.copy()
            tmp[~m] = np.nan
            vals_nomask.append(np.nanmean(tmp))

            # With Mask
            tmp2 = map_um.copy()
            tmp2[~m] = np.nan
            if ext_mask is not None:
                tmp2[ext_mask > 0] = np.nan
            vals_mask.append(np.nanmean(tmp2))

        # Update Table
        self.update_table_row(0, vals_mask)
        self.update_table_row(1, vals_nomask)

        # Drusen Logic (Old Code adaptation)
        if is_rpebm:
            ele2 = np.transpose(diff_px)  # In pixels
            ele2 = median_filter(ele2, size=(5, 5))

            # Thresholding
            thresh_samples = 6  # approx 12um
            drusen_mask = ele2 > thresh_samples

            # If external mask exists (GA), exclude it from Drusen calc
            if ext_mask is not None:
                drusen_mask[ext_mask > 0] = False

            # Area
            pixel_area_mm2 = (scan_width_mm / w) * (scan_width_mm / h)
            area = np.sum(drusen_mask) * pixel_area_mm2

            # Volume (microns * area / 1000)
            vol = np.sum(ele2[drusen_mask]) * axial * pixel_area_mm2 / 1000.0

            # Add to table (Rows 3 and 4)
            tab = self.tableWidget_ORLThicknessNumber
            if tab.rowCount() < 5:
                tab.insertRow(3);
                tab.insertRow(4)
                tab.setVerticalHeaderItem(3, QTableWidgetItem("Drusen Area"))
                tab.setVerticalHeaderItem(4, QTableWidgetItem("Drusen Vol"))

            tab.setItem(3, 0, QTableWidgetItem(f"{area:.3f}"))
            tab.setItem(4, 0, QTableWidgetItem(f"{vol:.5f}"))

        # Save to DF
        self.thickness_data.loc['With Mask'] = vals_mask
        self.thickness_data.loc['W/O Mask'] = vals_nomask
        self.thickness_data.loc['Fovea (X,Y)'] = [self.fovea_center[0], self.fovea_center[1], 0, 0]

        self.display_map(map_um, ext_mask)

    # --------------------------------------------------------------------
    # VISUALIZATION & UTILS
    # --------------------------------------------------------------------
    def display_map(self, map_um, ext_mask):
        # Create Plot
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

        # Colormap
        jet = plt.cm.jet
        colors = jet(np.linspace(0, 1, 256))
        colors[0] = [1, 1, 1, 1]  # White background for 0
        cmap = ListedColormap(colors)

        display_img = map_um.copy()
        if ext_mask is not None:
            display_img[ext_mask > 0] = 0  # Mask out

        cax = ax.imshow(display_img, cmap=cmap, vmin=0, vmax=300)
        ax.axis('off')

        # Draw Circles
        if self.fovea_center:
            cx, cy = self.fovea_center
            h, w = map_um.shape
            # Assuming fovea coordinates were relative to projection (0-500)
            # Need to ensure they match map coordinates
            scan_width_mm = 6.0
            px_per_mm = w / scan_width_mm

            for r_mm in [0.5, 1.5, 2.5]:
                circ = Circle((cx, cy), r_mm * px_per_mm, fill=False, color='black', linewidth=0.5)
                ax.add_patch(circ)

        plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        # Save temp
        tmp_path = os.path.join(current_dir, "temp_map.png")
        plt.savefig(tmp_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Load to scene
        pix = QPixmap(tmp_path)
        self.scene_cd.clear()
        self.scene_cd.addPixmap(pix)
        self.graphicsView_thicknessmap_w_mask.fitInView(self.scene_cd.sceneRect(), Qt.KeepAspectRatio)

    def draw_bscan(self):
        # Placeholder for drawing B-scan lines
        # (Requires converting logic from 'scenes.py' to here or using imported scenes)
        # Assuming you have the 'scene' object from imported scenes.py working
        pass

    def update_table_row(self, row, vals):
        for i, v in enumerate(vals):
            self.tableWidget_ORLThicknessNumber.setItem(row, i, QTableWidgetItem(f"{v:.2f}"))

    def create_circular_mask(self, h, w, radius):
        center = self.fovea_center if self.fovea_center else (w // 2, h // 2)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        return dist <= radius

    def set_thickness_data_frame(self):
        self.thickness_data = pd.DataFrame(columns=['All', '1mm', '3mm', '5mm'], index=['With Mask', 'W/O Mask'])

    def on_button_clicked_save_results(self):
        if not self.img_obj.exist_stru: return
        path = self.img_obj.url_stru if hasattr(self.img_obj, 'url_stru') else "ORL_Results.csv"
        save_path = os.path.splitext(path)[0] + "_thickness.csv"
        self.thickness_data.to_csv(save_path)
        self.send_log(f"Saved: {save_path}")

    def send_log(self, text):
        t = datetime.now().strftime("%H:%M:%S")
        self.plainTextEdit.appendPlainText(f"{t} -- {text}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ORLThicknessWindow()
    window.show()
    sys.exit(app.exec_())