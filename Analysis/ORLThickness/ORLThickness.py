import sys
import os
import numpy as np
import cv2
import scipy.io
import pandas as pd
from pathlib import Path
from datetime import datetime
import ctypes
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
from scipy.ndimage import median_filter, gaussian_filter
import qimage2ndarray
from skimage import io

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QPen
from PyQt5.QtCore import Qt

# -------------------------------------------------------------------------
# 1. SETUP PATHS & IMPORTS
# -------------------------------------------------------------------------
# Calculate paths relative to this file to ensure imports work standalone
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))

if root_dir not in sys.path:
    sys.path.append(root_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import Data Class
from Utils.data_class import state
# Import Config
from UI.config import Config
# Import Scenes
from UI.scenes import (GraphicsScene, GraphicsSceneThicknessMap,
                       GraphicsSceneThicknessMapProjection)
# Import UI Layout
import thicknessmapui


class ORLThicknessWindow(QtWidgets.QMainWindow, thicknessmapui.Ui_MainWindow):
    """
    ORL Thickness Analysis Window
    """

    def __init__(self, img_obj=None):
        """
        Args:
            img_obj: The shared data object passed from Main.
                     If None, we are in standalone mode.
        """
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.is_standalone = False

        # -----------------------------------------------------------
        # 1. DATA OBJECT INITIALIZATION & STATE SYNC
        # -----------------------------------------------------------
        if img_obj is None:
            print("Running in STANDALONE mode. Using global state.")
            self.is_standalone = True

            # In standalone, we use the existing global instance
            self.img_obj = state.img_obj

            # --- FIX: INITIALIZE DUMMY ATTRIBUTES TO PREVENT CRASH ---
            # The 'scenes.py' initialization tries to read stru3d immediately.
            # We must provide empty data so the GUI can build itself.

            # 1. Ensure basic flags exist
            if not hasattr(self.img_obj, 'exist_stru'): self.img_obj.exist_stru = False
            if not hasattr(self.img_obj, 'exist_seg'): self.img_obj.exist_seg = False

            # 2. Ensure dimensions exist (Default to 512x512 with 1 frame to be safe)
            if not hasattr(self.img_obj, 'img_width') or self.img_obj.img_width == 0:
                self.img_obj.img_width = 512
            if not hasattr(self.img_obj, 'img_depth') or self.img_obj.img_depth == 0:
                self.img_obj.img_depth = 512
            if not hasattr(self.img_obj, 'img_framenum') or self.img_obj.img_framenum == 0:
                self.img_obj.img_framenum = 1
            if not hasattr(self.img_obj, 'layer_num'):
                self.img_obj.layer_num = 0

            # 3. CRITICAL: Initialize 'stru3d' and 'layers' with zeros
            # This satisfies 'scenes.py' -> 'drawCurrentLine' -> 'stru3d[:,:,frame]'
            if not hasattr(self.img_obj, 'stru3d'):
                # Shape: (Height, Width, Frames) - using uint8 for display compatibility
                self.img_obj.stru3d = np.zeros((512, 512, 1), dtype=np.uint8)

            if not hasattr(self.img_obj, 'layers') or self.img_obj.layers is None:
                # Shape: (Width, Frames, Layers)
                self.img_obj.layers = np.zeros((512, 1, 2))

        else:
            print("Running in CALLABLE mode.")
            self.is_standalone = False
            self.img_obj = img_obj

            # Sync the global state so scenes.py (which uses state.img_obj) sees the passed data
            state.img_obj = self.img_obj

            # Auto-fill UI with existing paths if available
            if hasattr(self.img_obj, 'url_stru'):
                self.lineEdit_loadstruct.setText(str(self.img_obj.url_stru))
            if hasattr(self.img_obj, 'url_seg'):
                self.lineEdit_loadseg.setText(str(self.img_obj.url_seg))
        self.pushButton_loadstruct.clicked.connect(self.on_button_load_struct)
        self.pushButton_loadseg.clicked.connect(self.on_button_load_seg)
        self.scene = GraphicsScene()
        self.graphicsView_Bscan.setScene(self.scene)

        self.scene_thickness = GraphicsSceneThicknessMap()
        self.graphicsView_thicknessmap_wo_mask.setScene(self.scene_thickness)

        self.scene_proj = GraphicsSceneThicknessMapProjection()
        self.graphicsView_ORLProj.setScene(self.scene_proj)

        self.scene_cd = QGraphicsScene()
        self.graphicsView_thicknessmap_w_mask.setScene(self.scene_cd)
        self.fovea_center = None
        self.set_thickness_data_frame()
        self.setup_spin_boxes()

        # Connect existing buttons
        self.pushButton_Compute_and_Display.clicked.connect(self.update_fovea_center)
        self.pushButton_Compute_and_Display.clicked.connect(self.compute_ORL_thickness)
        self.pushButton_Compute_and_Display.clicked.connect(self.display_cd_mask_thickness_map)
        self.pushButton_load_cdmask.clicked.connect(self.on_button_clicked_load_cdmask)
        self.pushButton_load_ORLproj.clicked.connect(self.on_button_clicked_load_ORLproj)
        self.pushButton_save_csv.clicked.connect(self.on_button_clicked_save_results)

        # If data was passed in (Callable mode), refresh the UI immediately
        if not self.is_standalone and self.img_obj.exist_stru:
            self.refresh_ui_state()

    def on_button_load_struct(self):
        """Handle Structure File Loading"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Structure File", "", "Images (*.avi *.dcm *.img)")
        if file_path:
            self.lineEdit_loadstruct.setText(file_path)

            # Read real data (overwriting the dummy data)
            self.img_obj.read_stru_data(file_path)

            if self.img_obj.exist_stru:
                self.send_and_display_the_log(f"Structure loaded: {os.path.basename(file_path)}")

                # If loading fresh in standalone, init layers to match NEW dimensions
                if not self.img_obj.exist_seg or self.img_obj.layers is None:
                    # Create empty layers based on loaded dimensions
                    self.img_obj.layers = np.zeros((self.img_obj.img_width, self.img_obj.img_framenum, 1))
                    self.img_obj.layer_num = 1

                self.refresh_ui_state()

    def on_button_load_seg(self):
        """Handle Segmentation File Loading"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Segmentation File", "", "Files (*.txt *.mat *.npy)")
        if file_path:
            self.lineEdit_loadseg.setText(file_path)
            self.img_obj.read_seg_layers(file_path)

            if self.img_obj.exist_seg:
                self.send_and_display_the_log(f"Segmentation loaded: {os.path.basename(file_path)}")
                self.refresh_ui_state()

    def refresh_ui_state(self):
        """Call this after data is loaded to update ranges and views"""
        if self.img_obj.img_width == 0: return

        # 1. Update Ranges
        self.set_init_range()

        # 2. Update Scaling
        # Check to ensure img_width is valid to prevent division errors
        w_val = self.img_obj.img_width if self.img_obj.img_width > 0 else 1
        scale = self.graphicsView_Bscan.size().width() / (w_val + 15)

        self.graphicsView_Bscan.resetTransform()
        self.graphicsView_Bscan.scale(scale, scale)

        self.graphicsView_thicknessmap_wo_mask.resetTransform()
        self.graphicsView_thicknessmap_wo_mask.scale(0.5, 0.5)

        self.graphicsView_ORLProj.resetTransform()
        self.graphicsView_ORLProj.scale(0.5, 0.5)

        self.graphicsView_thicknessmap_w_mask.resetTransform()
        self.graphicsView_thicknessmap_w_mask.scale(0.5, 0.5)

        # 3. Reconnect Scene Signals
        self.connect_to_scene()

        # 4. Trigger Initial Plot
        try:
            self.on_button_clicked_load_ORLproj()
        except Exception as e:
            print(f"Could not auto-plot projection (normal if struct not ready): {e}")

    # --------------------------------------------------------------------
    # FUNCTIONAL LOGIC
    # --------------------------------------------------------------------
    def update_fovea_center(self):
        if self.img_obj.img_width == 0: return
        x = self.spinBox_2.value()
        y = self.img_obj.img_width - self.spinBox_3.value()
        self.fovea_center = (x, y)
        self.compute_ORL_thickness()

    def setup_spin_boxes(self):
        self.spinBox_2.valueChanged.connect(self.update_cursor_position)
        self.spinBox_3.valueChanged.connect(self.update_cursor_position)

    def update_cursor_position(self):
        if self.img_obj.img_width == 0: return
        x = self.spinBox_2.value()
        y = self.img_obj.img_width - self.spinBox_3.value()
        if hasattr(self, 'scene_proj'):
            self.scene_proj.update_cursor_position(x, y)
        self.selected_center = (x, y)

    def plot_on_the_scene_ORL(self, graphicsView_ORLProj, img, add_line=False,
                              add_slab_lines=False, lines=None):
        if np.issubdtype(img.dtype, np.uint16):
            img = np.uint8(img / 256.0)

        img_pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(img))

        # Safety check for scene_proj
        if not hasattr(self, 'scene_proj'):
            self.scene_proj = GraphicsSceneThicknessMapProjection()

        self.scene_proj.clear()
        self.scene_proj.addPixmap(img_pixmap)
        self.scene_proj.create_cursor()
        graphicsView_ORLProj.setScene(self.scene_proj)
        graphicsView_ORLProj.fitInView(self.scene_proj.sceneRect(), Qt.KeepAspectRatio)

    def transfer_image_range_to_uint8(self, img):
        if img.max() == 0: return img.astype(np.uint8)
        img_out = img / img.max()
        img_out = np.uint8(img_out * 255)
        img_out[img_out < 0] = 0
        img_out[img_out == -np.inf] = 0
        img_out[img_out > 255] = 255
        return img_out

    def on_button_clicked_load_ORLproj(self):
        # 1. Check if structural data exists before proceeding
        if not hasattr(self.img_obj, 'exist_stru') or not self.img_obj.exist_stru:
            return

        config = Config()
        current_layer = self.spinBox_CurrentLayer.value()

        # Use 'self.img_obj'
        try:
            self.img_proj_stru = self.img_obj.plot_proj(
                current_layer, current_layer, 'stru', projmethod='sum',
                start_offset=-2, end_offset=2, display=False, rotate=False
            )
        except Exception as e:
            print(f"Error plotting projection: {e}")
            return

        to_proj = self.transfer_image_range_to_uint8(self.img_proj_stru)

        # Handle missing w_ORL/h_ORL in Config safely
        new_width = getattr(config, 'w_ORL', 0)
        new_height = getattr(config, 'h_ORL', 0)

        # Only resize if valid dimensions exist in config
        if new_width > 0 and new_height > 0:
            resized_to_proj = cv2.resize(to_proj, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized_to_proj = to_proj

        rotated_to_proj = cv2.rotate(resized_to_proj, cv2.ROTATE_90_CLOCKWISE)
        flipped_to_proj = cv2.flip(rotated_to_proj, 1)

        self.plot_on_the_scene_ORL(self.graphicsView_ORLProj, flipped_to_proj)

    def on_button_clicked_save_results(self):
        if not self.img_obj.exist_stru: return

        base_path = self.img_obj.url_stru if hasattr(self.img_obj, 'url_stru') else "ORL_Result"
        filepath = os.path.dirname(base_path)
        if not os.path.exists(filepath): os.makedirs(filepath)

        filename = Path(base_path).stem + '_thickness_number.csv'
        self.thickness_data.to_csv(os.path.join(filepath, filename), float_format='%.6f')
        self.send_and_display_the_log(f'Saved to: {filepath}')

    def on_button_clicked_load_cdmask(self):
        if self.img_obj.img_width == 0:
            print("Please load OCT data first.")
            return

        ctypes.windll.user32.MessageBoxW(0, "Select CD Mask", "Explanation", 64)

        # Use QFileDialog for consistency if tk filedialog fails or looks out of place
        fm_i, _ = QFileDialog.getOpenFileName(self, "Open CD Mask", "", "Files (*.png *.jpg *.mat *.tif)")

        if not fm_i: return

        self.file = fm_i
        names = fm_i.split('.')
        file_format = names[-1].lower()
        target_shape = (self.img_obj.img_width, self.img_obj.img_framenum)

        try:
            if file_format in ['png', 'jpg', 'mat', 'tif']:
                if file_format == 'mat':
                    mat = scipy.io.loadmat(fm_i)
                    layer = mat["cdlayer"][:, :, 0]
                else:
                    layer = cv2.imread(fm_i, cv2.IMREAD_GRAYSCALE)

                layer = cv2.resize(layer, target_shape, interpolation=cv2.INTER_NEAREST)
                self.cdlayer = layer
                self.send_and_display_the_log("CD Mask Load Success")
            else:
                self.send_and_display_the_log("File type is not supported")
        except Exception as e:
            print(f"Mask load error: {e}")

    def set_init_range(self):
        if self.img_obj.img_framenum > 0:
            try:
                self.spinBox.setMaximum(self.img_obj.img_framenum - 1)
                self.spinBox.setMinimum(0)

                # Safe access to layer_num
                ln = getattr(self.img_obj, 'layer_num', 0)
                max_layer = ln - 1 if ln > 0 else 0

                self.spinBox_CurrentLayer.setMaximum(max_layer)
                self.spinBox_CurrentLayer.setMinimum(0)

                self.horizontalScrollBar.setMaximum(self.img_obj.img_framenum - 1)
                self.horizontalScrollBar.setMinimum(0)
            except Exception as e:
                print(f"Range init warning: {e}")

    def set_thickness_data_frame(self):
        self.thickness_data = pd.DataFrame(
            columns=['All', '1mm', '3mm', '5mm'],
            index=['With Mask', 'W/O Mask', 'Fovea (X,Y)'])

    def plot_on_the_scene(self, graphicsView_Bscan, img, add_line=False,
                          add_slab_lines=False, lines=None):
        if np.issubdtype(img.dtype, np.uint16):
            img = np.uint8(img / 256.0)

        scene = QGraphicsScene()
        img_pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(img))
        scene.addPixmap(img_pixmap)
        graphicsView_Bscan.setScene(scene)
        graphicsView_Bscan.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def add_white_to_colormap(self, color):
        ori_color = plt.colormaps.get_cmap(str(color))
        newcolors = ori_color(np.linspace(0, 1, 256))
        white = np.array([256 / 256, 256 / 256, 256 / 256, 1])
        newcolors[0] = white
        newcmp = ListedColormap(newcolors)
        return newcmp

    def display_cd_mask_thickness_map(self):
        if self.img_obj.layers is None: return

        import tifffile
        startlayer = self.spinBox_StartLayer.value()
        endlayer = self.spinBox_EndLayer.value()

        img_thickness = np.transpose(
            abs(self.img_obj.layers[:, :, startlayer] - self.img_obj.layers[:, :, endlayer]) * (3 / 1.536)
        )
        img_thickness = median_filter(gaussian_filter(img_thickness, sigma=(5, 1)), (1, 3))

        # Determine paths
        base_path = self.img_obj.url_stru if hasattr(self.img_obj, 'url_stru') else "ORL_Result"
        save_dir = os.path.dirname(base_path)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        # Basic save
        tifffile.imwrite(os.path.join(save_dir, Path(base_path).stem + 'Tiff_Regi_womask.tif'),
                         img_thickness.astype(np.float32))

        ccmap = self.add_white_to_colormap('jet')
        h, w = img_thickness.shape[:2]
        fovea_center = getattr(self, 'fovea_center', (h // 2, w // 2))
        fovea_center = fovea_center[::-1]

        def apply_colormap(image, cmap, vmin, vmax):
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            return plt.colormaps.get_cmap(cmap)(norm(image))

        # Check for CD Mask
        if hasattr(self, 'cdlayer') and self.cdlayer is not None:
            plot_img = img_thickness + self.cdlayer
            plot_img[plot_img < 0] = 0
            log_msg = "CD Mask Thickness Map Displayed"
        else:
            plot_img = img_thickness
            plot_img[plot_img < 0] = 0
            log_msg = "No Mask Thickness Map Displayed"

        colored_map = apply_colormap(plot_img, ccmap, 0, 300)

        # Plot with Circles
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.imshow(plot_img, cmap=ccmap, vmin=0, vmax=300)
        ax.axis('off')

        circ1 = Circle(fovea_center[::-1], 42, color='black', fill=False, linewidth=0.5)
        circ3 = Circle(fovea_center[::-1], 42 * 3, color='black', fill=False, linewidth=0.5)
        circ5 = Circle(fovea_center[::-1], 42 * 5, color='black', fill=False, linewidth=0.5)
        ax.add_patch(circ1)
        ax.add_patch(circ3)
        ax.add_patch(circ5)

        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        sm = plt.cm.ScalarMappable(cmap=ccmap, norm=plt.Normalize(vmin=0, vmax=300))
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('ORL Thickness (um)')

        final_path = os.path.join(save_dir, Path(base_path).stem + 'ThicknessMapWithCircles.png')
        plt.savefig(final_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)

        # Update UI
        img_display = io.imread(final_path)
        self.plot_on_the_scene(self.graphicsView_thicknessmap_w_mask, img_display)
        self.send_and_display_the_log(log_msg)

    def send_and_display_the_log(self, text):
        time = datetime.now().strftime('%m-%d %H:%M:%S -- ')
        str_send = time + text
        self.plainTextEdit.appendPlainText(str_send)

    def compute_ORL_thickness(self):
        if self.img_obj.layers is None: return

        table = self.tableWidget_ORLThicknessNumber
        startlayer = self.spinBox_StartLayer.value()
        endlayer = self.spinBox_EndLayer.value()

        img_thickness = np.transpose(
            abs(self.img_obj.layers[:, :, startlayer] - self.img_obj.layers[:, :, endlayer]) * (3 / 1.536)
        )
        img_thickness = median_filter(gaussian_filter(img_thickness, sigma=(5, 1)), (1, 3))

        rownames = self.thickness_data.index

        for row in range(0, self.thickness_data.shape[0] - 1):
            h, w = img_thickness.shape[0], img_thickness.shape[1]
            mask1 = self.create_circular_mask(h, w, radius=42)
            mask2 = self.create_circular_mask(h, w, radius=42 * 3)
            mask3 = self.create_circular_mask(h, w, radius=42 * 5)

            def get_stats(data, mask):
                masked = data.copy()
                masked[~mask] = 0
                masked[masked == 0] = np.nan
                return np.nanmean(masked)

            mean0 = np.nanmean(img_thickness)
            mean1 = get_stats(img_thickness, mask1)
            mean2 = get_stats(img_thickness, mask2)
            mean3 = get_stats(img_thickness, mask3)

            m0, m1, m2, m3 = [0, mean0], [0, mean1], [0, mean2], [0, mean3]

            if hasattr(self, 'cdlayer') and self.cdlayer is not None:
                cdlayer = self.cdlayer
                with_mask = img_thickness.copy()
                with_mask[cdlayer != 0] = 0
                with_mask[with_mask < 0] = 0
                with_mask[with_mask == 0] = np.nan

                with_mean0 = np.nanmean(with_mask)
                with_mean1 = get_stats(with_mask, mask1)
                with_mean2 = get_stats(with_mask, mask2)
                with_mean3 = get_stats(with_mask, mask3)

                m0 = [with_mean0, mean0]
                m1 = [with_mean1, mean1]
                m2 = [with_mean2, mean2]
                m3 = [with_mean3, mean3]

            self.set_item_on_table(table, row, 0, "{:.2f}".format(m0[row]))
            self.set_item_on_table(table, row, 1, "{:.2f}".format(m1[row]))
            self.set_item_on_table(table, row, 2, "{:.2f}".format(m2[row]))
            self.set_item_on_table(table, row, 3, "{:.2f}".format(m3[row]))

            self.thickness_data.loc[rownames[row]] = float("{:.6f}".format(m0[row])), \
                float("{:.6f}".format(m1[row])), \
                float("{:.6f}".format(m2[row])), \
                float("{:.6f}".format(m3[row]))

        self.send_and_display_the_log("Thickness Computation Complete")

        x_coord = self.spinBox_2.value()
        y_coord = self.spinBox_3.value()
        self.thickness_data.loc["Fovea (X,Y)"] = float("{:.3f}".format(x_coord / 1000)), \
            float("{:.3f}".format(y_coord / 1000)), 0, 0

    def add_row_on_table(self, table, text='ProjNew'):
        rowPosition = table.rowCount()
        table.insertRow(rowPosition)
        table.setVerticalHeaderItem(rowPosition, QTableWidgetItem(text))

    def set_item_on_table(self, table, row, column, text):
        table.setItem(row, column, QTableWidgetItem(text))

    def create_circular_mask(self, h, w, center=None, radius=None):
        if center is None:
            center = self.fovea_center if self.fovea_center else (int(w / 2), int(h / 2))
        if radius is None:
            radius = min(center[0], center[1], w - center[0], h - center[1])

        x = np.linspace(0, h - 1, h)
        y = np.linspace(0, w - 1, w)
        X, Y = np.meshgrid(x, y)
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        return dist_from_center <= radius

    def connect_to_scene(self):
        # FIX: Safety check to prevent crash if data isn't loaded yet
        if self.img_obj.layers is None or not hasattr(self.img_obj, 'layer_num') or self.img_obj.layer_num == 0:
            return

        inds = list(range(0, self.img_obj.layer_num))

        self.horizontalScrollBar.valueChanged.connect(
            lambda: self.scene.drawCurrentLine(self.img_obj.layers, inds=inds,
                                               frame=self.horizontalScrollBar.value(),
                                               current_layer=self.spinBox_CurrentLayer.value()))
        self.horizontalScrollBar.valueChanged.connect(
            lambda: self.scene_thickness.drawCurrentLine(frame=self.horizontalScrollBar.value()))
        self.horizontalScrollBar.valueChanged.connect(
            lambda: self.scene_proj.drawCurrentLine(frame=self.horizontalScrollBar.value()))

        self.horizontalScrollBar.valueChanged.connect(lambda: self.spinBox.setValue(self.horizontalScrollBar.value()))
        self.spinBox.valueChanged.connect(lambda: self.horizontalScrollBar.setValue(self.spinBox.value()))


# -------------------------------------------------------------------------
# STANDALONE ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    # FIX: Set the attribute BEFORE creating the application instance
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    app = QtWidgets.QApplication(sys.argv)

    # Create and show the window
    window = ORLThicknessWindow()
    window.show()

    sys.exit(app.exec_())