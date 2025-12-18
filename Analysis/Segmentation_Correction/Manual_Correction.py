import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from Analysis.Segmentation_Correction import SegWindowUI
from UI.config import Config
from UI.scenes import GraphicsScene, GraphicsSceneColormap, GraphicsSceneZslice
from Utils.data_class import state


class SegWindow(QtWidgets.QMainWindow, SegWindowUI.Ui_MainWindow):
    '''
    Manual segementaiton mode
    '''

    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        # CHANGED: img_obj -> state.img_obj
        if not (state.img_obj.exist_seg):
            state.img_obj.layers = np.zeros((state.img_obj.img_width, state.img_obj.img_framenum))

        self.set_init_range()
        self.original_layers = None

        # CHANGED: img_obj -> state.img_obj
        self.lineEdit_LayerNumber.setText(str(state.img_obj.layer_num))

        # init the scene for displaying the segmetation lines
        self.scene = GraphicsScene()
        self.graphicsView_ManualEdit.setScene(self.scene)

        # set scale to fit the view
        # CHANGED: img_obj -> state.img_obj
        scale = self.graphicsView_ManualEdit.size().width() / (state.img_obj.img_width + 15)  # fit with width
        self.graphicsView_ManualEdit.scale(scale, scale)
        self.scale_w = scale
        self.scale_h = scale

        # Radio
        self.pushButton_EZseg.clicked.connect(self.update_scene)

        # init the scene for depth image
        self.scene_thickness = GraphicsSceneColormap()
        self.graphicsView_Thickness.setScene(self.scene_thickness)

        # fit in the view, ugly solution
        # CHANGED: img_obj -> state.img_obj
        self.graphicsView_Thickness.scale(
            self.graphicsView_Thickness.size().width() / (state.img_obj.img_width + 10),
            self.graphicsView_Thickness.size().height() / (state.img_obj.img_framenum + 10))

        # depth slice
        self.scene_Zslice = GraphicsSceneZslice()
        self.graphicsView_Zslice.setScene(self.scene_Zslice)
        # CHANGED: img_obj -> state.img_obj
        self.graphicsView_Zslice.scale(
            self.graphicsView_Zslice.size().width() / (state.img_obj.img_width + 10),
            self.graphicsView_Zslice.size().height() / (state.img_obj.img_framenum + 10))

        # set the value to the scene
        self.spinBox_CopyLine.valueChanged.connect(
            lambda: self.scene.set_values(self.spinBox_CopyLine.value()))

        self.connect_to_scene()
        # add a new layer
        self.pushButton_AddNewLine.clicked.connect(self.add_new_line)
        self.pushButton_DeleteLine.clicked.connect(self.delete_a_line)

        self.pushButton_Save.clicked.connect(self.save_current_result)
        self.pushButton_Smooth.clicked.connect(self.on_button_clicked_smooth)
        self.pushButton_line_shift.clicked.connect(self.on_button_clicked_line_shift)
        self.pushButton_edge_correct.clicked.connect(self.on_button_clicked_edge_correct)
        self.pushButton_edge_correct_2.clicked.connect(self.on_button_clicked_edge_correct)

        self.checkBox_DisplayLines.stateChanged.connect(
            self.update_display_line)

        self.keypress_count = 0

    def update_scene(self):
        self.graphicsView_ManualEdit.resetTransform()
        config = Config()

        # Get the viewport size
        viewport_size = self.graphicsView_ManualEdit.viewport().size()
        viewport_width = viewport_size.width()
        viewport_height = viewport_size.height()

        # Get the current scene rect
        scene_rect = self.scene.sceneRect()
        scene_width = scene_rect.width()
        scene_height = scene_rect.height()

        if self.pushButton_EZseg.isChecked():
            config.is_distorted = True
            kern = float(self.lineEdit_Kernel_EZ.text())
            distort_factor = kern
            config.dist_fact = kern

            # Calculate the new width and height after distortion
            new_width = scene_width * distort_factor
            new_height = scene_height / distort_factor

            # Calculate scale to fit the width
            scale_w = viewport_width / new_width

            # Apply the scale
            self.graphicsView_ManualEdit.scale(scale_w * distort_factor,
                                               scale_w / distort_factor)
        else:
            config.is_distorted = False
            # Calculate scale to fit the width
            scale_w = viewport_width / scene_width
            self.graphicsView_ManualEdit.scale(scale_w, scale_w)

        # Update the scene
        self.scene.update()

    def toggle_EZseg(self):
        self.pushButton_EZseg.setChecked(not self.pushButton_EZseg.isChecked())
        self.update_scene()

    def set_init_range(self):
        """Set the range for widgets, spinbox, scrollbar
        """
        try:
            # CHANGED: img_obj -> state.img_obj
            self.spinBox_FrameNumber.setMaximum(state.img_obj.img_framenum - 1)
            self.spinBox_FrameNumber.setMinimum(0)
            self.spinBox_CurrentLayer.setMaximum(state.img_obj.layer_num - 1)
            self.spinBox_CurrentLayer.setMinimum(0)
            self.horizontalScrollBar_ManualEdit.setMaximum(state.img_obj.img_framenum - 1)
            self.horizontalScrollBar_ManualEdit.setMinimum(0)
            self.spinBox_CopyLine.setMaximum(state.img_obj.img_framenum - 1)
        except Exception as e:
            print(e)

    def update_display_line(self):
        """ Display the lines
        :return:
        """
        # CHANGED: img_obj -> state.img_obj
        inds = list(range(0, state.img_obj.layer_num))
        self.connect_to_scene(disconnect=True)
        self.scene.drawCurrentLine(state.img_obj.layers, inds=inds,
                                   frame=self.spinBox_FrameNumber.value(),
                                   current_layer=self.spinBox_CurrentLayer.value(),
                                   is_draw_line=self.checkBox_DisplayLines.isChecked())

    def add_new_line(self):
        """Add a new layer, the layer is copied from current selected layer
        """
        # CHANGED: img_obj -> state.img_obj
        new_layer = np.copy(state.img_obj.layers[:, :, self.spinBox_CurrentLayer.value()])
        state.img_obj.layers = np.dstack((state.img_obj.layers, new_layer))

        # updates
        state.img_obj.layer_num = state.img_obj.layer_num + 1
        self.lineEdit_LayerNumber.setText(str(state.img_obj.layer_num))
        self.spinBox_CurrentLayer.setMaximum(state.img_obj.layer_num - 1)
        self.connect_to_scene(disconnect=True)

    def delete_a_line(self):
        """Delete a line on the scene
        """
        # CHANGED: img_obj -> state.img_obj
        if state.img_obj.layers.shape[2] > 1:
            state.img_obj.layers = np.delete(state.img_obj.layers,
                                             self.spinBox_CurrentLayer.value(),
                                             axis=2)

            # updates the numbers
            state.img_obj.layer_num = state.img_obj.layers.shape[2]
            print('shape of layers', state.img_obj.layers.shape, '  ', state.img_obj.layer_num)
            self.disconnect_signals()
            self.lineEdit_LayerNumber.setText(str(state.img_obj.layer_num))
            self.spinBox_CurrentLayer.setValue(state.img_obj.layer_num - 1)
            self.spinBox_CurrentLayer.setMaximum(state.img_obj.layer_num - 1)
            self.connect_to_scene(disconnect=False)

    def connect_to_scene(self, disconnect=False):
        """connect and disconnect the signals
        """
        # CHANGED: img_obj -> state.img_obj
        inds = list(range(0, state.img_obj.layer_num))

        if disconnect:
            self.disconnect_signals()

        # set the segmentation window
        self.spinBox_FrameNumber.valueChanged.connect(
            lambda: self.scene.drawCurrentLine(state.img_obj.layers, inds=inds,
                                               frame=self.spinBox_FrameNumber.value(),
                                               current_layer=self.spinBox_CurrentLayer.value(),
                                               is_draw_line=self.checkBox_DisplayLines.isChecked()))
        self.spinBox_CurrentLayer.valueChanged.connect(
            lambda: self.scene.drawCurrentLine(state.img_obj.layers, inds=inds,
                                               frame=self.spinBox_FrameNumber.value(),
                                               current_layer=self.spinBox_CurrentLayer.value(),
                                               is_draw_line=self.checkBox_DisplayLines.isChecked()))
        self.horizontalScrollBar_ManualEdit.valueChanged.connect(
            lambda: self.scene.drawCurrentLine(state.img_obj.layers, inds=inds,
                                               frame=self.horizontalScrollBar_ManualEdit.value(),
                                               current_layer=self.spinBox_CurrentLayer.value(),
                                               is_draw_line=self.checkBox_DisplayLines.isChecked()))

        # set the thickness window
        self.spinBox_CurrentLayer.valueChanged.connect(
            lambda: self.scene_thickness.drawCurrentLine(
                frame=self.spinBox_FrameNumber.value(),
                current_layer=self.spinBox_CurrentLayer.value()))
        self.horizontalScrollBar_ManualEdit.valueChanged.connect(
            lambda: self.scene_thickness.drawCurrentLine(
                frame=self.horizontalScrollBar_ManualEdit.value(),
                current_layer=self.spinBox_CurrentLayer.value()))

        # set the z-slice window
        self.spinBox_CurrentLayer.valueChanged.connect(
            lambda: self.scene_Zslice.drawCurrentLine(
                frame=self.spinBox_FrameNumber.value(),
                current_layer=self.spinBox_CurrentLayer.value()))
        self.horizontalScrollBar_ManualEdit.valueChanged.connect(
            lambda: self.scene_Zslice.drawCurrentLine(
                frame=self.horizontalScrollBar_ManualEdit.value(),
                current_layer=self.spinBox_CurrentLayer.value()))

        # synchronize the spinbox with the horizontal scroll bar
        self.horizontalScrollBar_ManualEdit.valueChanged.connect(
            lambda: self.spinBox_FrameNumber.setValue(
                self.horizontalScrollBar_ManualEdit.value()))
        self.spinBox_FrameNumber.valueChanged.connect(
            lambda: self.horizontalScrollBar_ManualEdit.setValue(
                self.spinBox_FrameNumber.value()))

        # set the fit line switch
        self.checkBox_LineFit.stateChanged.connect(
            lambda: self.scene.set_line_fit_status(
                self.checkBox_LineFit.isChecked()))

        self.spinBox_FitBound.valueChanged.connect(
            lambda: self.scene.set_bounds(self.spinBox_FitBound.value()))

        self.radioButton_FitRPE.toggled.connect(
            lambda: self.scene.set_fit_mode(self.radioButton_FitRPE.isChecked(),
                                            self.radioButton_FitBM.isChecked()))
        self.radioButton_FitBM.toggled.connect(
            lambda: self.scene.set_fit_mode(self.radioButton_FitRPE.isChecked(),
                                            self.radioButton_FitBM.isChecked()))

    def disconnect_signals(self):
        """ Disconnect the signals
        """
        self.spinBox_FrameNumber.valueChanged.disconnect()
        self.spinBox_CurrentLayer.valueChanged.disconnect()
        self.horizontalScrollBar_ManualEdit.valueChanged.disconnect()

    def keyPressEvent(self, event):
        """ set the keyboard shortcuts QE ,AD, WS, ZC
        """
        if event.key() == Qt.Key_A:
            self.spinBox_FrameNumber.setValue(
                self.spinBox_FrameNumber.value() - 1)
            self.key_press_count_check()
        elif event.key() == Qt.Key_D:
            self.spinBox_FrameNumber.setValue(
                self.spinBox_FrameNumber.value() + 1)
            self.key_press_count_check()
        elif event.key() == Qt.Key_Q:
            self.spinBox_FrameNumber.setValue(
                self.spinBox_FrameNumber.value() - 10)
            self.key_press_count_check()
        elif event.key() == Qt.Key_E:
            self.spinBox_FrameNumber.setValue(
                self.spinBox_FrameNumber.value() + 10)
            self.key_press_count_check()
        elif event.key() == Qt.Key_W:
            self.spinBox_CurrentLayer.setValue(
                self.spinBox_CurrentLayer.value() - 1)
        elif event.key() == Qt.Key_S:
            self.spinBox_CurrentLayer.setValue(
                self.spinBox_CurrentLayer.value() + 1)
        elif event.key() == Qt.Key_Z:
            self.spinBox_CopyLine.setValue(self.spinBox_CopyLine.value() - 1)
        elif event.key() == Qt.Key_C:
            self.spinBox_CopyLine.setValue(self.spinBox_CopyLine.value() + 1)
        elif event.key() == Qt.Key_Shift:
            self.checkBox_DisplayLines.nextCheckState()

    def on_button_clicked_smooth(self):
        """
        Smoothing the current layer
        Returns:
        """
        layer_num = self.spinBox_CurrentLayer.value()
        kernel_size = float(self.lineEdit_Kernel.text())
        # CHANGED: img_obj -> state.img_obj
        state.img_obj.layers[:, :, layer_num] = gaussian_filter(
            state.img_obj.layers[:, :, layer_num], (kernel_size, 1))

    def on_button_clicked_line_shift(self):
        """
        Line shift the curent seg line by an amount vertically
        """

        layer_num = self.spinBox_CurrentLayer.value()
        line_shift_amt = self.spinBox_line_shift_amt.value()
        if self.original_layers is None:
            # CHANGED: img_obj -> state.img_obj
            self.original_layers = state.img_obj.layers.copy()

            # Perform the shift from the original position
        for i in range(0, state.img_obj.layers.shape[1] - 1):
            state.img_obj.layers[:, i, layer_num] = self.original_layers[:, i,
            layer_num] + line_shift_amt

    def on_button_clicked_edge_correct(self):
        """Edge Correct the segmentation lines for the current layer across all images"""
        layer_num = self.spinBox_CurrentLayer.value()
        left_edge = self.spinBox_edge_left.value()
        right_edge = self.spinBox_edge_right.value()

        if self.original_layers is None:
            # CHANGED: img_obj -> state.img_obj
            self.original_layers = state.img_obj.layers.copy()

        # Restore original segmentation if both edges are 0
        if left_edge == 0 and right_edge == 0:
            state.img_obj.layers = self.original_layers.copy()
            return

        # For each image in the volume
        # CHANGED: img_obj -> state.img_obj
        for img_idx in range(state.img_obj.layers.shape[1]):
            positions = state.img_obj.layers[:, img_idx, layer_num]

            # Left edge correction
            if left_edge > 0:
                ref_slice = positions[left_edge:left_edge + 10]
                # Avoid crash if slice is empty or all zeros
                valid_pixels = ref_slice[ref_slice != 0]
                if valid_pixels.size > 0:
                    ref_median = np.median(valid_pixels)
                    for i in range(left_edge):
                        if positions[i] != 0 and abs(positions[i] - ref_median) > 30:
                            state.img_obj.layers[i, img_idx, layer_num] = ref_median

            # Right edge correction
            if right_edge > 0:
                right_start = positions.shape[0] - right_edge
                ref_slice = positions[right_start - 10:right_start]
                valid_pixels = ref_slice[ref_slice != 0]
                if valid_pixels.size > 0:
                    ref_median = np.median(valid_pixels)
                    for i in range(right_start, positions.shape[0]):
                        if positions[i] != 0 and abs(positions[i] - ref_median) > 30:
                            state.img_obj.layers[i, img_idx, layer_num] = ref_median

    def key_press_count_check(self):
        """auto save the result
        :return:
        """
        self.keypress_count += 1
        if self.keypress_count >= 10:
            self.keypress_count = 0
            self.save_current_result()
            print('auto saving...')

    def save_current_result(self):
        """save the current segmentation results to the local disk
        """
        # get dir name and file name
        # CHANGED: img_obj -> state.img_obj
        state.img_obj.flatten_loc = int(self.spinBox_flatten_location.value())
        dirname = os.path.dirname(state.img_obj.save_path)
        base = os.path.basename(state.img_obj.save_path)
        file_name = os.path.splitext(base)[0]

        url = os.path.join(dirname, file_name)
        if not os.path.exists(url):
            os.makedirs(url)

        # save seg layers
        file_name_mat = os.path.join(url, 'layers.mat')
        file_name_npy = os.path.join(url, 'layers.npy')
        state.img_obj.save_layers(file_name_mat)
        state.img_obj.save_layers(file_name_npy)
        print('segmentation files saved to :', url)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SegWindow()
    window.show()
    sys.exit(app.exec_())