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
import io
from skimage import io as skio

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QTableWidgetItem, QGraphicsPathItem
from PyQt5.QtGui import QPixmap, QPen, QColor, QPainterPath
from PyQt5.QtCore import Qt, pyqtSignal

# -------------------------------------------------------------------------
# 1. SETUP PATHS & UI LOADING
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if root_dir not in sys.path: sys.path.append(root_dir)
if current_dir not in sys.path: sys.path.append(current_dir)

# UI File Logic
UI_FILE_PATH = os.path.join(current_dir, r"./thicknessmapui.ui")
if not os.path.exists(UI_FILE_PATH):
    # Fallback if file missing (for standalone logic)
    pass

try:
    with open(UI_FILE_PATH, 'r', encoding='utf-8') as f:
        ui_xml = f.read()
    # Fix namespace issues in older QT Designer files
    ui_xml = ui_xml.replace('Qt::AlignmentFlag::', 'Qt::')
    ui_xml = ui_xml.replace('Qt::Orientation::', 'Qt::')
    ui_xml = ui_xml.replace('Qt::WindowType::', 'Qt::')
    ui_xml = ui_xml.replace('QFrame::Shadow::', 'QFrame::')
    f_io = io.StringIO(ui_xml)
    Ui_MainWindow, QtBaseClass = uic.loadUiType(f_io)
except:
    # If UI load fails, use generic object to allow code inspection
    Ui_MainWindow = object
    QtBaseClass = object


# -------------------------------------------------------------------------
# 2. DUMMY CLASSES (For Standalone Testing)
# -------------------------------------------------------------------------
class DummyData:
    def __init__(self):
        self.exist_stru = False
        self.exist_seg = False
        self.img_width = 500
        self.img_depth = 800
        self.img_framenum = 100
        self.layer_num = 0
        self.url_stru = ""
        self.url_seg = ""
        # Mock Data: (Depth, Width, Frames)
        self.stru3d = np.random.randint(0, 100, (800, 500, 100), dtype=np.uint8)
        self.layers = None

    def read_stru_data(self, filepath):
        self.url_stru = filepath
        self.exist_stru = True
        self.img_width = 500
        self.img_framenum = 100
        self.img_depth = 800
        # Generate some visible noise
        self.stru3d = np.random.randint(0, 255, (self.img_depth, self.img_width, self.img_framenum), dtype=np.uint8)
        print(f"Standalone: Mock Structure Loaded.")

    def read_seg_layers(self, filepath):
        self.url_seg = filepath
        self.exist_seg = True
        self.layer_num = 2
        # Layers: (Width, Frames, Layers)
        self.layers = np.zeros((500, 100, 2))
        # Layer 0 at y=200, Layer 1 at y=300 (roughly)
        for f in range(100):
            self.layers[:, f, 0] = 200 + np.sin(np.linspace(0, 10, 500)) * 20
            self.layers[:, f, 1] = 300 + np.cos(np.linspace(0, 10, 500)) * 20
        print(f"Standalone: Mock Layers Loaded.")

    def plot_proj(self, layer1, layer2, type, projmethod, start_offset, end_offset, display, rotate):
        # Return simple mean projection
        return np.mean(self.stru3d, axis=0).astype(np.uint8)


# -------------------------------------------------------------------------
# 3. CUSTOM SCENE (Interactive)
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
        # Remove old cursor
        for item in self.items():
            if hasattr(item, 'is_cursor'):
                self.removeItem(item)

        pen = QPen(Qt.red)
        pen.setWidth(2)

        path = QtWidgets.QGraphicsPathItem()
        path.is_cursor = True
        p = path.path()
        size = 15
        p.moveTo(x - size, y);
        p.lineTo(x + size, y)
        p.moveTo(x, y - size);
        p.lineTo(x, y + size)
        path.setPath(p)
        path.setPen(pen)
        self.addItem(path)


# -------------------------------------------------------------------------
# 4. MAIN WINDOW CLASS
# -------------------------------------------------------------------------
class ORLThicknessWindow(QtWidgets.QMainWindow, Ui_MainWindow if Ui_MainWindow != object else QtWidgets.QMainWindow):
    def __init__(self, img_obj=None):
        super(self.__class__, self).__init__()
        if hasattr(self, 'setupUi'):
            self.setupUi(self)

        # --- Handle Data Object ---
        if img_obj is None:
            self.img_obj = DummyData()
            self.is_standalone = True
        else:
            self.img_obj = img_obj
            self.is_standalone = False
            if hasattr(self.img_obj, 'url_stru'):
                self.lineEdit_loadstruct.setText(str(self.img_obj.url_stru))
            if hasattr(self.img_obj, 'url_seg'):
                self.lineEdit_loadseg.setText(str(self.img_obj.url_seg))

        # --- Scene Initialization ---
        self.scene = QGraphicsScene()
        self.graphicsView_Bscan.setScene(self.scene)

        self.scene_thickness = QGraphicsScene()
        self.graphicsView_thicknessmap_wo_mask.setScene(self.scene_thickness)

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

        # Optional UI element checks
        if hasattr(self, 'comboBox_MaskList'):
            self.comboBox_MaskList.currentIndexChanged.connect(self.on_mask_selected)
        if hasattr(self, 'pushButton_AutoFovea'):
            self.pushButton_AutoFovea.clicked.connect(self.auto_detect_fovea)

        self.spinBox_2.valueChanged.connect(self.update_cursor_from_spinbox)
        self.spinBox_3.valueChanged.connect(self.update_cursor_from_spinbox)

        # --- Variables ---
        self.fovea_center = None
        self.cdlayer = None
        self.thickness_data = None
        self.current_proj_pixmap = None  # Cache for the expensive projection image
        self.set_thickness_data_frame()

        # Initialize
        # Only refresh if we actually have data loaded in the passed object
        if not self.is_standalone and hasattr(self.img_obj, 'exist_stru') and self.img_obj.exist_stru:
            self.refresh_ui_state()
            # Try to populate masks if path exists
            self.populate_mask_list()

    # --------------------------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------------------------
    def on_button_load_struct(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Structure File", "", "Images (*.avi *.dcm *.img *.vol)")
        if file_path:
            self.lineEdit_loadstruct.setText(file_path)
            self.img_obj.read_stru_data(file_path)
            if self.img_obj.exist_stru:
                self.send_log(f"Structure loaded: {os.path.basename(file_path)}")
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
                # Trigger projection ONLY when seg loads, not every frame
                self.generate_and_plot_projection()

    def refresh_ui_state(self):
        """Called when data is loaded to set scrollbar ranges."""
        if self.img_obj.img_width == 0: return

        # Set Ranges
        self.spinBox.setMaximum(self.img_obj.img_framenum - 1)
        self.horizontalScrollBar.setMaximum(self.img_obj.img_framenum - 1)

        max_layer = max(0, self.img_obj.layer_num - 1)
        self.spinBox_CurrentLayer.setMaximum(max_layer)
        self.spinBox_StartLayer.setMaximum(max_layer)
        self.spinBox_EndLayer.setMaximum(max_layer)

        # Cleanup old connections to prevent duplicates
        try:
            self.horizontalScrollBar.valueChanged.disconnect()
        except:
            pass
        try:
            self.spinBox.valueChanged.disconnect()
        except:
            pass
        try:
            self.spinBox_CurrentLayer.valueChanged.disconnect()
        except:
            pass

        # Synch Scrollbar <-> Spinbox
        self.horizontalScrollBar.valueChanged.connect(self.spinBox.setValue)
        self.spinBox.valueChanged.connect(self.horizontalScrollBar.setValue)

        # 1. Update B-Scan (Heavy drawing of lines) when scrollbar moves
        self.horizontalScrollBar.valueChanged.connect(self.draw_bscan)

        # 2. Update Projection LINE ONLY (Lightweight) when scrollbar moves
        self.horizontalScrollBar.valueChanged.connect(self.update_proj_line)

        # 3. Update ACTUAL Projection (Heavy Calc) only when Layer changes
        self.spinBox_CurrentLayer.valueChanged.connect(self.generate_and_plot_projection)

        # Initial Draw
        self.draw_bscan()

    # --------------------------------------------------------------------
    # PROJECTION LOGIC (Split into Calc and Draw)
    # --------------------------------------------------------------------
    def normalize_image(self, img):
        """
        Converts float/uint16 data to uint8 (0-255) for display.
        Matches the logic in your Parent MainWindow to fix the blurry/noise issue.
        """
        if img is None: return None

        # Handle Float or 16-bit integers
        if np.issubdtype(img.dtype, np.floating) or img.dtype == np.uint16:
            img = np.nan_to_num(img)  # Remove NaNs
            if img.max() > 0:
                # Normalize 0 to 1, then scale to 255
                img = (img / img.max()) * 255
            img = img.astype(np.uint8)

        return img

    def generate_and_plot_projection(self):
        if not self.img_obj.exist_stru: return

        # Ensure we have layers before trying to project specific layers
        if not self.img_obj.exist_seg or self.img_obj.layers is None: return
        if self.img_obj.layer_num < 1: return

        try:
            current_layer = self.spinBox_CurrentLayer.value()

            # Generate Projection
            if hasattr(self.img_obj, 'plot_proj'):
                # Integrated Mode: Use the main software's projection logic
                self.img_proj_stru = self.img_obj.plot_proj(
                    current_layer, current_layer, 'stru', projmethod='sum',
                    start_offset=-2, end_offset=2, display=False, rotate=False
                )
            else:
                # Standalone Fallback
                self.img_proj_stru = np.mean(self.img_obj.stru3d, axis=0)

            # --- APPLY FIX HERE ---
            img = self.normalize_image(self.img_proj_stru)

            # Rotate/Flip for display convention
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.flip(img, 1)

            # Cache & Display
            self.current_proj_pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(img))
            self.update_proj_line()

        except Exception as e:
            self.send_log(f"Proj Error: {e}")

    def update_proj_line(self):
        """Redraws the cached projection image + the yellow current frame line."""
        if self.current_proj_pixmap is None: return

        self.scene_proj.clear()
        self.scene_proj.addPixmap(self.current_proj_pixmap)

        # Draw line indicating current scroll position
        # Because of ROTATE_90_CLOCKWISE and FLIP, coordinates transform.
        # Original: Width x Frames
        # Rotated: Frames x Width
        # Flipped Horizontally: Frames x Width (but X is inverted)

        current_frame = self.horizontalScrollBar.value()

        # Based on typical OCT UI logic for this specific transformation:
        # The Frame index usually maps to the Y-axis of the rotated image.

        w = self.current_proj_pixmap.width()
        h = self.current_proj_pixmap.height()

        pen = QPen(Qt.green)
        pen.setWidth(2)

        # Drawing a horizontal line at Y = current_frame
        # Ensure frame is within height limits
        y_pos = min(max(current_frame, 0), h - 1)
        self.scene_proj.addLine(0, y_pos, w, y_pos, pen)

        if self.fovea_center:
            self.scene_proj.update_cursor(*self.fovea_center)

        self.graphicsView_ORLProj.fitInView(self.scene_proj.sceneRect(), Qt.KeepAspectRatio)

    # --------------------------------------------------------------------
    # B-SCAN DRAWING (The missing logic)
    # --------------------------------------------------------------------
    def draw_bscan(self):
        """Manually draws the B-scan image and segmentation lines."""
        if not self.img_obj.exist_stru: return

        frame_idx = self.horizontalScrollBar.value()

        # 1. Get Image Slice
        try:
            # Assuming Dimensions: (Depth, Width, Frames) which is standard for raw reads
            # or (Frames, Depth, Width). We try the most common for python oct:
            if self.img_obj.stru3d.shape[2] == self.img_obj.img_framenum:
                # (Depth, Width, Frames)
                img_slice = self.img_obj.stru3d[:, :, frame_idx]
            else:
                # (Frames, Depth, Width)
                img_slice = self.img_obj.stru3d[frame_idx, :, :]
        except IndexError:
            return

        # 2. Normalize & Display Image
        if img_slice.max() > 0:
            img_disp = (img_slice / img_slice.max() * 255).astype(np.uint8)
        else:
            img_disp = img_slice.astype(np.uint8)

        pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(img_disp))

        self.scene.clear()
        self.scene.addPixmap(pixmap)

        # 3. Draw Segmentation Lines
        # Data structure for layers is usually: (Width, Frames, Layers)
        if self.img_obj.exist_seg and self.img_obj.layers is not None:

            selected_layer_idx = self.spinBox_CurrentLayer.value()
            width = img_slice.shape[1]

            # Iterate over all layers
            for l_idx in range(self.img_obj.layer_num):
                path = QPainterPath()
                started = False

                # Extract Y-coordinates for this specific layer and frame
                # shape: (Width, Frames, Layers) -> [:, frame_idx, l_idx]
                try:
                    y_coords = self.img_obj.layers[:, frame_idx, l_idx]
                except IndexError:
                    continue

                # Build Path
                for x in range(len(y_coords)):
                    y = y_coords[x]
                    if y > 0:  # Check valid segmentation
                        if not started:
                            path.moveTo(x, y)
                            started = True
                        else:
                            path.lineTo(x, y)

                # Color Selection
                if l_idx == selected_layer_idx:
                    pen = QPen(Qt.cyan)  # Blue/Cyan for current
                    pen.setWidth(2)
                else:
                    pen = QPen(Qt.red)  # Red for others
                    pen.setWidth(1)

                self.scene.addPath(path, pen)

        self.graphicsView_Bscan.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    # --------------------------------------------------------------------
    # MASK & OTHER UTILS
    # --------------------------------------------------------------------
    def populate_mask_list(self):
        if not hasattr(self, 'comboBox_MaskList'): return
        if not self.img_obj.url_stru: return
        parent_dir = Path(self.img_obj.url_stru).parent
        patterns = ["*mask*.png", "*mask*.tif", "*mask*.jpg", "*mask*.mat"]
        mask_files = []
        for p in patterns:
            mask_files.extend(glob.glob(os.path.join(parent_dir, p)))
        self.comboBox_MaskList.clear()
        self.comboBox_MaskList.addItem("No Mask Selected", None)
        for f in mask_files:
            self.comboBox_MaskList.addItem(os.path.basename(f), f)

    def on_mask_selected(self, index):
        if index <= 0:
            self.cdlayer = None
            self.send_log("Mask cleared.")
            return
        mask_path = self.comboBox_MaskList.itemData(index)
        try:
            # Simple loading logic for mask
            if mask_path.endswith('.mat'):
                mat = scipy.io.loadmat(mask_path)
                key = [k for k in mat.keys() if not k.startswith('_')][0]
                layer = mat[key]
            else:
                layer = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            layer = cv2.resize(layer, (self.img_obj.img_width, self.img_obj.img_framenum),
                               interpolation=cv2.INTER_NEAREST)
            self.cdlayer = layer
            self.send_log(f"Mask Loaded: {os.path.basename(mask_path)}")
        except Exception as e:
            self.send_log(f"Mask Load Error: {e}")

    def on_proj_clicked(self, x, y):
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
        if self.scene_proj: self.scene_proj.update_cursor(x, y)

    def auto_detect_fovea(self):
        if self.current_proj_pixmap is None: return
        try:
            # Convert pixmap back to image or use cached array if stored
            # For simplicity, using center of image
            w, h = self.current_proj_pixmap.width(), self.current_proj_pixmap.height()
            self.on_proj_clicked(w // 2, h // 2)
            self.send_log(f"Auto-Fovea set to center.")
        except:
            pass

    # --------------------------------------------------------------------
    # RESTORED COMPUTATION & DATA LOGIC
    # --------------------------------------------------------------------
    def set_thickness_data_frame(self):
        """
        Initializes the pandas DataFrame used to store thickness results.
        """
        self.thickness_data = pd.DataFrame(
            columns=['All', '1mm', '3mm', '5mm'],
            index=['With Mask', 'W/O Mask', 'Fovea (X,Y)']
        )

    def compute_ORL_thickness(self):
        """
        Calculates thickness maps, handling layer swapping automatically (e.g. 8 vs 10).
        """
        if self.img_obj.layers is None:
            self.send_log("Error: No layers found to compute thickness.")
            return

        table = self.tableWidget_ORLThicknessNumber

        # 1. Handle Layer Selection (Auto-Swap)
        # We take the absolute difference, so order technically doesn't matter for the math,
        # but sorting them ensures consistent behavior if we add directional logic later.
        l1 = self.spinBox_StartLayer.value()
        l2 = self.spinBox_EndLayer.value()
        startlayer, endlayer = sorted((l1, l2))

        self.send_log(f"Computing thickness between Layer {startlayer} and {endlayer}")

        # 2. Calculate Thickness Map (Pixels -> Microns)
        # Using the absolute difference ensures it works even if layers are inverted
        scale_factor = 3 / 1.536
        diff_abs = np.abs(self.img_obj.layers[:, :, startlayer] - self.img_obj.layers[:, :, endlayer])

        # Transpose to match map orientation (Width x Frames -> Frames x Width)
        img_thickness = np.transpose(diff_abs * scale_factor)

        # Apply Filters (Median + Gaussian)
        img_thickness = median_filter(gaussian_filter(img_thickness, sigma=(5, 1)), (1, 3))

        # 3. Setup Masks (ETDRS Circles: 1mm, 3mm, 5mm)
        h, w = img_thickness.shape[0], img_thickness.shape[1]

        # Radius logic: 42 pixels approx 1mm diameter (radius=0.5mm)
        mask1 = self.create_circular_mask(h, w, radius=42)
        mask2 = self.create_circular_mask(h, w, radius=42 * 3)
        mask3 = self.create_circular_mask(h, w, radius=42 * 5)

        # 4. Calculate Statistics

        # -- External Mask Handling (for "With Mask" row) --
        if hasattr(self, 'cdlayer') and self.cdlayer is not None:
            # Ensure cdlayer is resized to match current data
            ext_mask = self.cdlayer
            if ext_mask.shape != img_thickness.shape:
                ext_mask = cv2.resize(ext_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            with_mask_map = img_thickness.copy()
            with_mask_map[ext_mask != 0] = 0  # Mask out pathology
            with_mask_map[with_mask_map < 0] = 0
        else:
            with_mask_map = np.zeros_like(img_thickness)

        # -- Helper to calculate mean for a specific map --
        def get_means(map_data):
            # Create masked copies
            m1 = map_data.copy();
            m1[~mask1] = 0
            m3 = map_data.copy();
            m3[~mask2] = 0
            m5 = map_data.copy();
            m5[~mask3] = 0

            # Use nanmean, treating 0 as no-data (NaN) for accurate average
            # (If 0 is a valid thickness for you, remove the '== 0' checks)
            map_data_n = map_data.copy();
            map_data_n[map_data_n == 0] = np.nan
            m1[m1 == 0] = np.nan
            m3[m3 == 0] = np.nan
            m5[m5 == 0] = np.nan

            return [np.nanmean(map_data_n), np.nanmean(m1), np.nanmean(m3), np.nanmean(m5)]

        # Get means
        means_wo = get_means(img_thickness)  # Row 1: W/O Mask

        if hasattr(self, 'cdlayer') and self.cdlayer is not None:
            means_w = get_means(with_mask_map)  # Row 0: With Mask
        else:
            means_w = [0, 0, 0, 0]

        # 5. Update UI Table
        # Row 0: With Mask
        for col, val in enumerate(means_w):
            val_str = "NaN" if np.isnan(val) else f"{val:.2f}"
            table.setItem(0, col, QTableWidgetItem(val_str))

        # Row 1: W/O Mask
        for col, val in enumerate(means_wo):
            val_str = "NaN" if np.isnan(val) else f"{val:.2f}"
            table.setItem(1, col, QTableWidgetItem(val_str))

        # 6. Update DataFrame (Fixing the broadcasting error)
        # We explicitly convert NaNs to 0.0 or keep them as float for the CSV
        clean_w = [0.0 if np.isnan(x) else float(f"{x:.6f}") for x in means_w]
        clean_wo = [0.0 if np.isnan(x) else float(f"{x:.6f}") for x in means_wo]

        self.thickness_data.loc['With Mask'] = clean_w
        self.thickness_data.loc['W/O Mask'] = clean_wo

        # Update Fovea
        x_coord = self.spinBox_2.value()
        y_coord = self.spinBox_3.value()
        self.thickness_data.loc["Fovea (X,Y)"] = [x_coord / 1000, y_coord / 1000, 0, 0]

        self.send_log("Thickness Computation Complete")
        self.display_cd_mask_thickness_map()

    def on_button_clicked_save_results(self):
        """
        Saves the thickness_data DataFrame to a CSV file.
        """
        if self.thickness_data is None:
            self.send_log("No data to save.")
            return

        # Determine Save Path based on loaded struct file
        if hasattr(self.img_obj, 'url_stru') and self.img_obj.url_stru:
            filepath = os.path.dirname(self.img_obj.url_stru)
            filename_stem = Path(self.img_obj.url_stru).stem
        else:
            filepath = os.getcwd()
            filename_stem = "Default"

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # Naming logic
        if hasattr(self, 'cdlayer') and self.cdlayer is not None:
            name = filename_stem + '_thickness_number.csv'
            self.send_log("Saved: Thickness files saved (With Mask)")
        else:
            name = filename_stem + '_thickness_number_without_mask_only.csv'
            self.send_log("Saved: Thickness without mask saved")

        save_loc = os.path.join(filepath, name)
        self.thickness_data.to_csv(save_loc, float_format='%.6f')

    def send_log(self, text):
        """
        Appends text to the log window with a timestamp.
        """
        t = datetime.now().strftime("%H:%M:%S")
        self.plainTextEdit.appendPlainText(f"{t} -- {text}")

    # --------------------------------------------------------------------
    # REQUIRED HELPER (Must be present for compute_ORL_thickness to work)
    # --------------------------------------------------------------------
    def create_circular_mask(self, h, w, center=None, radius=None):
        if center is None:
            # Note: fovea_center stored as (x,y), but numpy grid often needs (y,x) or proper indexing
            # Your original code used: center = self.fovea_center if... else (w/2, h/2)
            if self.fovea_center:
                center = self.fovea_center
            else:
                center = (int(w / 2), int(h / 2))

        if radius is None:
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    def set_thickness_data_frame(self):
        """
        Initializes the pandas DataFrame with ALL 4 columns to prevent shape mismatch errors.
        """
        self.thickness_data = pd.DataFrame(
            columns=['All', '1mm', '3mm', '5mm'],
            index=['With Mask', 'W/O Mask', 'Fovea (X,Y)']
        )

    def send_log(self, text):
        t = datetime.now().strftime("%H:%M:%S")
        self.plainTextEdit.appendPlainText(f"{t} -- {text}")

    def display_cd_mask_thickness_map(self):
        """
        Generates thickness maps (0-100um, 0-200um), saves them in a dedicated folder,
        and fixes aspect ratio issues.
        """
        if self.img_obj.layers is None: return

        # --- 1. SETUP DATA & PATHS ---
        try:
            import tifffile
        except ImportError:
            tifffile = None

        # Get Scale (Axial Res in microns/pixel)
        if hasattr(self, 'doubleSpinBox_Axial'):
            axial_res = self.doubleSpinBox_Axial.value()
        else:
            axial_res = 3 / 1.536  # ~1.953 um/pixel

        startlayer = self.spinBox_StartLayer.value()
        endlayer = self.spinBox_EndLayer.value()
        l1, l2 = sorted((startlayer, endlayer))

        # Calculate Thickness (Pixel & Micron)
        # Shape: (Width, Frames) -> Transpose to (Frames, Width) for correct orientation
        diff_px = np.transpose(np.abs(self.img_obj.layers[:, :, l1] - self.img_obj.layers[:, :, l2]))

        # Smooth
        diff_px = median_filter(gaussian_filter(diff_px, sigma=(5, 1)), (1, 3))

        # Convert to Microns (Values are now in um)
        diff_um = diff_px * axial_res

        # --- NEW: Create Designated Results Folder ---
        if hasattr(self.img_obj, 'url_stru') and self.img_obj.url_stru:
            parent_dir = os.path.dirname(self.img_obj.url_stru)
            stem = Path(self.img_obj.url_stru).stem
        else:
            parent_dir = os.getcwd()
            stem = "result"

        # Create a specific folder for these results
        results_dir = os.path.join(parent_dir, f"{stem}_Analysis_Results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        self.send_log(f"Saving outputs to: {results_dir}")

        # --- 2. SAVE RAW DATA (CSV) ---
        # Save Pixel Data
        np.savetxt(os.path.join(results_dir, f"{stem}_Thickness_Pixels.csv"), diff_px, delimiter=",", fmt="%.4f")
        # Save Micron Data (In um, not mm)
        np.savetxt(os.path.join(results_dir, f"{stem}_Thickness_Microns.csv"), diff_um, delimiter=",", fmt="%.4f")

        # Save Metadata
        with open(os.path.join(results_dir, f"{stem}_Metadata.txt"), "w") as f:
            f.write(f"Analyzed File: {stem}\n")
            f.write(f"Layer Range: {l1} - {l2}\n")
            f.write(f"Axial Resolution: {axial_res} um/pixel\n")
            f.write(f"Unit: Microns (um)\n")

        # --- 3. PLOTTING HELPER ---
        def save_fancy_map(data_um, suffix, vmin, vmax, add_circles=False, add_colorbar=False, mask_zeros=False):
            # Create Figure
            # Note: aspect='equal' is crucial here to prevent vertical compression
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

            # Setup Colormap
            cmap = plt.cm.jet
            cmap.set_bad(color='black')

            plot_data = data_um.copy()
            if mask_zeros:
                # Mask out 0 values so they appear white/background color
                plot_data = np.ma.masked_where(plot_data <= 0, plot_data)
                cmap.set_bad(color='white')

            # PLOT IMAGE
            # aspect='equal' ensures 1 pixel width = 1 pixel height (Square)
            im = ax.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
            ax.axis('off')

            # Add Circles
            if add_circles and self.fovea_center:
                cx, cy = self.fovea_center
                for r in [42, 42 * 3, 42 * 5]:
                    c = Circle((cx, cy), r, color='white' if mask_zeros else 'black', fill=False, linewidth=1)
                    ax.add_patch(c)

            # Add Colorbar
            if add_colorbar:
                # fraction=0.046 and pad=0.04 are magic numbers to make colorbar match image height
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Thickness (µm)')

            # Construct Filename
            range_tag = f"_{vmin}-{vmax}um"
            feat_tag = "_Circles" if add_circles else ""
            bar_tag = "_Cbar" if add_colorbar else ""
            fname = f"{stem}_{suffix}{range_tag}{feat_tag}{bar_tag}.png"

            full_path = os.path.join(results_dir, fname)
            plt.savefig(full_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            return full_path

        # --- 4. GENERATE MAPS ---
        datasets = [("Raw", diff_um)]

        # Handle Masking (CDLayer)
        if hasattr(self, 'cdlayer') and self.cdlayer is not None:
            ext_mask = self.cdlayer
            if ext_mask.shape != diff_um.shape:
                ext_mask = cv2.resize(ext_mask, (diff_um.shape[1], diff_um.shape[0]), interpolation=cv2.INTER_NEAREST)

            masked_um = diff_um.copy()
            masked_um[ext_mask != 0] = 0
            datasets.append(("Masked", masked_um))

            # Save Masked CSV
            np.savetxt(os.path.join(results_dir, f"{stem}_Thickness_Microns_Masked.csv"), masked_um, delimiter=",",
                       fmt="%.4f")

        ranges = [(0, 100), (0, 200)]
        last_image_path = None

        for name, data in datasets:
            for (vmin, vmax) in ranges:
                # 1. Plain (Viewing)
                path = save_fancy_map(data, name, vmin, vmax, add_circles=False, add_colorbar=False)
                # 2. Circles + Colorbar
                save_fancy_map(data, name, vmin, vmax, add_circles=True, add_colorbar=True)
                # 3. Just Colorbar
                save_fancy_map(data, name, vmin, vmax, add_circles=False, add_colorbar=True)

                # Update GUI with the 0-200 masked version if available
                if vmax == 200:
                    last_image_path = path

        # --- 5. UPDATE GUI ---
        if last_image_path and os.path.exists(last_image_path):
            pixmap = QPixmap(last_image_path)
            self.scene_cd.clear()
            self.scene_cd.addPixmap(pixmap)
            self.graphicsView_thicknessmap_w_mask.fitInView(self.scene_cd.sceneRect(), Qt.KeepAspectRatio)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ORLThicknessWindow()
    window.show()
    sys.exit(app.exec_())