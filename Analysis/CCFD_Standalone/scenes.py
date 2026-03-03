from PyQt5.QtWidgets import QGraphicsScene, QGraphicsLineItem, QGraphicsEllipseItem
from PyQt5.QtGui import QPixmap, QPen, QBrush, QColor
from PyQt5.QtCore import Qt, QPoint
import numpy as np
import qimage2ndarray
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter,median_filter
import weakref

from data_class import *


class GraphicsSceneTail(QGraphicsScene):
    """
    The graphicsscene class for tail removal window
    """

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)

    def img_to_colormap(self, img):
        """ Transfer nd array to colormap image
        :param img:
        :return:
        """
        img = np.uint8(img)
        cm = get_cmap('jet')
        img_color = cm(img)
        return img_color

    def update_image(self, img):
        '''Add lines on the scene
        Args:
            layers: layers data
            lines: [img_width, number_of_lines]
        '''
        self.clear()
        self.addPixmap(self.get_pix_map(img))
        self.setSceneRect(0, 0, img.shape[1], img.shape[0])

    def get_pix_map(self, img):
        """ transfer the numpy array to the QT pixmap
        :param img: 2d or RGB numpy array
        :return: Qt pixmap
        """
        if np.issubdtype(img.dtype, np.uint16):
            img = np.uint8(img / 256.0)
        # img = np.uint8(img)
        img_pixmap = QPixmap.fromImage(
            qimage2ndarray.array2qimage(img, normalize=True))
        return img_pixmap

    def draw_rect(self, img, x1, y1, x2, y2):
        """
        draw the lines on the img
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :return:
        """

        self.clear()
        self.addPixmap(self.get_pix_map(img))
        # all layers
        pen = QPen(Qt.red, 3)
        brush = QBrush()
        self.addRect(x1, y1, x2 - x1, y2 - y1, pen, brush)

        self.setSceneRect(0, 0, img.shape[1], img.shape[0])


class GraphicsScene(QGraphicsScene):
    """
    The graphicsscene class for the manual correction
    """

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        # self.setSceneRect(-100, -100, 200, 200)
        self.drawing = False
        self.lastPoint = QPoint()
        self.copy_line_num = 1  # default value of the line parpageation
        self.line_fit_status = True
        self.fit_bounds = 5
        self.fitRPE = True
        self.drawCurrentLine(state.img_obj.layers,
                             inds=list(range(0, state.img_obj.layer_num)), frame=0,
                             current_layer=0)

    def set_values(self, copy_num=1):
        self.copy_line_num = copy_num

    def set_line_fit_status(self, status=True):
        self.line_fit_status = status

    def set_bounds(self, bounds=5):
        self.fit_bounds = bounds

    def set_fit_mode(self, RPEmode, BMmode):
        self.fitRPE = RPEmode
        self.fitBM = BMmode

    def mousePressEvent(self, event):
        """ When press the mouse down, get the location of the mouse pointer and create a new line to store the
        locations
        """
        if event.button() == Qt.LeftButton:
            self.drawing = True
            pen = QPen(Qt.red)
            brush = QBrush(Qt.red)
            x = event.scenePos().x()
            y = event.scenePos().y()
            # self.addEllipse(x, y, 4, 4, pen, brush)
            self.lastPoint = event.scenePos()
            # print(x, y)

            # set up the line for drawing
            self.new_line = np.empty((state.img_obj.img_width, 2))
            self.new_line[:] = np.nan

            if self.is_in_the_boundary(x, y):
                self.new_line[np.int32(x), :] = [x, y]

    def mouseReleaseEvent(self, event):
        """ When release the mouse button, save the line to the global variable
        :param event:
        """
        try:
            if event.button() == Qt.LeftButton:
                self.drawing = False

                # if the line has more than 3 points
                if np.sum(~np.isnan(self.new_line[:, 0])) >= 3:
                    new_x, new_y = self.inter_the_line(self.new_line)
                    new_x = np.int32(new_x)
                    new_y = np.int32(new_y)
                    x_min = np.nanmin(new_x)
                    x_max = np.nanmax(new_x)
                    # print('interp xy done: ', x_min, x_max)
                    # print(new_x, new_y)
                    # state.img_obj.layers[x_min:x_max+1, self.frame, self.current_layer] = new_y   #can change here
                    #

                    if self.line_fit_status:
                        self.fit_lines(x_min, x_max,
                                       new_y)  # fit the line to the max response
                    else:
                        self.change_line(x_min, x_max, new_y)  # direct draw


        except Exception as e:
            print(e)

    def mouseMoveEvent(self, event):
        """ When move the mouse, draw the line on the scene
        :param event:
        """
        if event.buttons() == Qt.LeftButton and self.drawing:
            pen = QPen(Qt.green)
            brush = QBrush(Qt.green)
            x = event.scenePos().x()
            y = event.scenePos().y()
            # print('mouse pointer = ', x, y)
            # self.addEllipse(x, y, 2, 2, pen, brush)
            self.addLine(self.lastPoint.x(), self.lastPoint.y(), x, y, pen)
            # self.lastPoint = event.pos()
            self.lastPoint = event.scenePos()

            # check the boundary here
            if self.is_in_the_boundary(x, y):
                self.new_line[int(x), :] = [x, y]
            self.update()

    def fit_lines(self, x_min, x_max, new_y):
        """
        fit the lines to the maxinum intensity location of the image
        Args:
            bound: the upper and lower bounds of the mask

        Returns:

        """
        num_of_copy = self.copy_line_num

        # check bounds
        if self.frame + num_of_copy > state.img_obj.img_framenum:
            num_of_copy = state.img_obj.img_framenum - self.frame

        # put the MASK on!
        slices = state.img_obj.stru3d[:, x_min:x_max + 1,
                 self.frame:self.frame + num_of_copy]
        slices = gaussian_filter(slices, (1, 0.5, 0.5), mode='nearest')
        for i in range(0, x_max - x_min + 1):
            slices[0:new_y[i] - self.fit_bounds, i,
            :] = 0  # todo: should check the bounds here
            slices[new_y[i] + self.fit_bounds::, i, :] = 0

        # find the max intensity
        if self.fitRPE:
            new_loc = gaussian_filter(np.argmax(slices, axis=0), (3, 2),
                                      mode='nearest')
        else:
            slices = median_filter(slices, (5, 5, 5), mode='nearest')
            slices = scharr(slices)
            new_loc = gaussian_filter(np.argmax(slices, axis=0), (3, 2),
                                      mode='nearest')

        # save change to the layers
        state.img_obj.layers[x_min:x_max + 1, self.frame:self.frame + num_of_copy,
        self.current_layer] = new_loc

        # return new_loc

        # smooth the drawing lines
        x_1 = x_min - 20
        x_2 = x_max + 20
        if x_1 < 0:
            x_1 = 0
        if x_2 > state.img_obj.img_width - 1:
            x_2 = state.img_obj.img_width - 1

        sub_layer = state.img_obj.layers[x_1:x_2, self.frame:self.frame + num_of_copy,
                    self.current_layer]
        state.img_obj.layers[x_1:x_2, self.frame:self.frame + num_of_copy,
        self.current_layer] = gaussian_filter(
            gaussian_filter(sub_layer, (1, 0.5), mode='nearest'), (3, 0.5),
            mode='nearest')

        # draw_the_fit_line with yellow color
        pen = QPen(Qt.yellow)
        for i in range(0, x_max - x_min):
            self.addLine(x_min + i, new_loc[i, 0], x_min + i + 1,
                         new_loc[i + 1, 0], pen)
            # self.lastPoint = event.pos()

    # def draw_the_fit_line(self, ):

    def change_line(self, x_min, x_max, new_y):
        """ put the line to the global variable
        :param x_min:
        :param x_max:
        :param new_y:
        """
        num_of_copy = self.copy_line_num
        # print('x_min, xmax', x_min, x_max)
        # print(new_y.shape)

        # propagate the line to following frames
        if self.frame + num_of_copy <= state.img_obj.img_framenum:
            state.img_obj.layers[x_min:x_max + 1, self.frame:self.frame + num_of_copy,
            self.current_layer] = np.tile(new_y, (num_of_copy, 1)).T
        else:
            num_of_copy = state.img_obj.img_framenum - self.frame
            state.img_obj.layers[x_min:x_max + 1, self.frame::,
            self.current_layer] = np.tile(new_y, (num_of_copy, 1)).T

    def is_in_the_boundary(self, x, y):
        """ check if the point (x,y) is inside the boundary
        :param x: x
        :param y: y
        :return: True or False
        """
        if ((x >= 0) and (x < state.img_obj.img_width) and (y >= 0) and (
                y < state.img_obj.img_depth)):
            return True
        else:
            return False

    def change_xy_in_bound(self, x, dim):
        """
        check if Xs are in bound, if not, change them
        Args:
            x:
            dim:

        Returns:

        """
        if dim == 'x':
            max_l = state.img_obj.img_width
        elif dim == 'y':
            max_l = state.img_obj.img_framenum
        elif dim == 'z':
            max_l = state.img_obj.img_depth

        x[x < 0] = 0
        x[x > max_l - 1] = max_l - 1

        return

    def drawCurrentLine(self, layers, inds=[0, 1], frame=10, current_layer=0,
                        is_draw_line=True):
        '''Add lines on the scene
        Args:
            layers: layers data
            lines: [img_width, number_of_lines]
        '''
        from UI.config import Config
        config = Config()
        self.clear()
        self.addPixmap(self.get_pix_map(state.img_obj.stru3d[:, :, frame]))

        # if draw the line on the scene
        if is_draw_line:
            if inds != []:
                self.current_layer = current_layer
                self.frame = frame

                # print('current lines', inds)

                img_w = layers.shape[0]
                lines = np.squeeze(state.img_obj.layers[:, frame, inds])
                x=1
                # all layers
                pen = QPen(Qt.red, 1)
                if (config.is_distorted):
                    df = config.dist_fact
                    pen = QPen(Qt.red, df*1.5)
                for i in inds:
                    for j in range(0, img_w - 1):
                        self.addLine(j, lines[j, i], j + 1, lines[j + 1, i],
                                     pen)

                # current editing line
                pen = QPen(Qt.blue, 1.5)
                if (config.is_distorted):
                    df = config.dist_fact
                    pen = QPen(Qt.blue, df*1.6)
                for j in range(0, img_w - 1):
                    self.addLine(j, lines[j, current_layer], j + 1,
                                 lines[j + 1, current_layer], pen)

    def get_pix_map(self, img):
        """ transfer the numpy array to the QT pixmap
        :param img: 2d or RGB numpy array
        :return: Qt pixmap
        """
        if np.issubdtype(img.dtype, np.uint16):
            img = np.uint8(img / 256.0)
        img_pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(img))
        return img_pixmap

    def inter_the_line(self, line):
        """Interpolate the drawn line
        """
        x = line[:, 0]
        y = line[:, 1]
        x, y = self.set_xy_in_bound(x, y)
        try:
            try:
                f1 = interp1d(x[~np.isnan(x)], y[~np.isnan(y)], kind='cubic',
                              fill_value='extrapolate')
            except:
                try:
                    f1 = interp1d(x[~np.isnan(x)], y[~np.isnan(y)],
                                  kind='linear', fill_value="extrapolate")
                    print('less than 4 points, use linear interpolation')
                except:
                    print('line with shape=', line.shape)

            x_min = np.nanmin(x)
            # if x_min < 0:
            #     x_min = 0
            x_max = np.nanmax(x)
            # if x_max > state.img_obj.img_width-1:
            #     x_max = state.img_obj.img_width-1
            x_new = np.arange(x_min, x_max + 1)
            y_new = f1(x_new)
        except:
            x_new = x
            y_new = y

        return x_new, y_new

    def set_xy_in_bound(self, x, y):
        """ If the the x,  y are out of bound, set them in bounds
        :param x: 1-d numpy array
        :param y: 1-d numpy array
        :return: the corrected coordinates
        """
        x[x > state.img_obj.img_width - 1] = state.img_obj.img_width - 1
        x[x < 0] = 0
        y[y > state.img_obj.img_depth] = state.img_obj.img_depth - 1
        y[y < 0] = 0
        return x, y


class GraphicsScene2d(QGraphicsScene):
    """
    The graphicsscene class for image registration
    """

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)

    def img_to_colormap(self, img):
        """ Transfer nd array to colormap image
        :param img:
        :return:
        """
        img = np.uint8(img)
        cm = get_cmap('jet')
        img_color = cm(img)
        return img_color

    def update_image(self, img):
        '''Add lines on the scene
        Args:
            layers: layers data
            lines: [img_width, number_of_lines]
        '''
        self.clear()
        self.addPixmap(self.get_pix_map(img))

    def get_pix_map(self, img):
        """ transfer the numpy array to the QT pixmap
        :param img: 2d or RGB numpy array
        :return: Qt pixmap
        """
        if np.issubdtype(img.dtype, np.uint16):
            img = np.uint8(img / 256.0)
        # img = np.uint8(img)
        img_pixmap = QPixmap.fromImage(
            qimage2ndarray.array2qimage(img, normalize=True))
        return img_pixmap

    def clean_img(self):
        """
        clean the pixmap
        :return:
        """
        self.clear()


class GraphicsSceneColormap(QGraphicsScene):
    """
    The graphicsscene class for the manual correction, depth colormap
    """

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.drawCurrentLine(0, 0)

    def img_to_colormap(self, img):
        """ Transfer nd array to colormap image
        :param img:
        :return:
        """
        img = np.uint8(img)
        cm = get_cmap('jet')
        img_color = cm(img)
        return img_color

    def drawCurrentLine(self, frame=0, current_layer=0):
        '''Add lines on the scene
        Args:
            layers: layers data
            lines: [img_width, number_of_lines]
        '''
        self.clear()
        self.current_layer = current_layer
        img_thickness = state.img_obj.layers[:, :, current_layer].T
        img_thickness = (img_thickness - img_thickness.min()) / (
                    img_thickness.max() - img_thickness.min()) * 255
        self.addPixmap(self.get_pix_map(self.img_to_colormap(img_thickness)))
        self.frame = frame

        img_w = state.img_obj.img_width
        img_h = state.img_obj.img_framenum

        # draw all layers

        pen = QPen(Qt.white, 1)
        #self.addLine(frame, 0, frame, img_w, pen)
        self.addLine(0, frame, img_h, frame, pen)


    def get_pix_map(self, img):
        """ transfer the numpy array to the QT pixmap
        :param img: 2d or RGB numpy array
        :return: Qt pixmap
        """
        if np.issubdtype(img.dtype, np.uint16):
            img = np.uint8(img / 256.0)
        # img = np.uint8(img)
        img_pixmap = QPixmap.fromImage(
            qimage2ndarray.array2qimage(img, normalize=True))
        return img_pixmap


class GraphicsSceneZslice(GraphicsScene2d):
    """
    The graphicsscene class for the manual correction, depth colormap
    """

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.drawCurrentLine(0, 0)

    def drawCurrentLine(self, frame=0, current_layer=0):
        '''Add lines on the scene
        Args:
            layers: layers data
            lines: [img_width, number_of_lines]
        '''
        self.clear()
        self.current_layer = current_layer
        # get the z slice of the current image volume
        # img_thickness = state.img_obj.layers[:, :, current_layer]
        # img_thickness = (img_thickness-img_thickness.min())/(img_thickness.max()-img_thickness.min())*255

        img_porj = state.img_obj.plot_proj(current_layer, current_layer, 'stru',
                                     'max',
                                     start_offset=0, end_offset=1,
                                     display=False,
                                     rotate=True)
        self.addPixmap(self.get_pix_map(img_porj))
        self.frame = frame

        img_w = state.img_obj.img_width
        img_h = state.img_obj.img_framenum

        # draw all layers

        pen = QPen(Qt.white, 1)
        #self.addLine(frame, 0, frame, img_w, pen)
        self.addLine(0, frame, img_h, frame, pen)


import weakref  # Ensure this is imported at the top of scenes.py if not already


class GraphicsSceneThicknessMapProjection(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_line = None
        self.cursor = None
        self._current_line = None

        # Initialize with dummy values to prevent crash on startup
        self.drawCurrentLine(0, 0)

    @property
    def current_line(self):
        return self._current_line() if self._current_line is not None else None

    @current_line.setter
    def current_line(self, line):
        self._current_line = weakref.ref(line) if line is not None else None

    def create_cursor(self):
        self.cursor = QGraphicsEllipseItem(0, 0, 10, 10)
        self.cursor.setBrush(QColor(255, 0, 0))  # Red cursor
        self.cursor.setZValue(1000)  # Ensure cursor is on top
        self.addItem(self.cursor)

    def update_cursor_position(self, x, y):
        if self.cursor:
            self.cursor.setPos(x - 5, y - 5)  # Adjust for cursor size

    def drawCurrentLine(self, frame=0, current_layer=0):
        """
        Updates the horizontal green line indicating the current B-scan frame.
        Does NOT redraw the background image (to preserve the projection map).
        """
        # --- FIX 1: REMOVE BAD IMPORTS/CALLS ---
        # Do NOT call self.clear() (It deletes the projection map)
        # Do NOT call self.addPixmap() (It overwrites the map with a B-scan)

        # Just remove the old indicator line
        if self.current_line is not None:
            self.removeItem(self.current_line)
            self.current_line = None

        # Safety check: if data isn't loaded, stop here
        if not hasattr(state.img_obj, 'img_width') or state.img_obj.img_width == 0:
            return

        # Calculate line coordinates (Horizontal line at y = frame)
        start_x = 0
        start_y = frame
        end_x = state.img_obj.img_width
        end_y = frame

        # Create the new green line
        line = QGraphicsLineItem(start_x, start_y, end_x, end_y)
        pen = QPen(QColor(0, 255, 0))  # Green
        pen.setWidth(3)
        line.setPen(pen)

        self.addItem(line)
        self.current_line = line
        self.update()


class GraphicsSceneThicknessMap(QGraphicsScene):
    """
    The graphicsscene class for the manual correction, depth colormap
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        QGraphicsScene.__init__(self, parent)
        self.drawCurrentLine(0, 0)

    def img_to_colormap(self, img):
        """ Transfer nd array to colormap image """
        img = np.uint8(img)
        cm = get_cmap('jet')
        img_color = cm(img)
        return img_color

    def add_white_to_colormap(self, color):
        ori_color = plt.colormaps.get_cmap(str(color))
        newcolors = ori_color(np.linspace(0, 1, 256))
        white = np.array([256 / 256, 256 / 256, 256 / 256, 1])
        newcolors[0] = white
        newcmp = ListedColormap(newcolors)
        return newcmp

    def drawCurrentLine(self, frame=0, current_layer=0):
        '''Add lines on the scene'''

        # --- FIX: ADD THIS IMPORT HERE ---
        from UI.config import Config

        config = Config()
        self.clear()

        # Safety check: Ensure layers exist before trying to access index 0 and 1
        if state.img_obj.layers is None or state.img_obj.layers.shape[2] < 2:
            return

        img_thickness = np.transpose(
            abs(state.img_obj.layers[:, :, 0] - state.img_obj.layers[:, :, 1]) * (
                    3 / 1.536))

        img_thickness = median_filter(
            gaussian_filter(img_thickness, sigma=(4, 1)), (1, 3))

        ccmap = self.add_white_to_colormap('jet')
        h, w = img_thickness.shape[:2]
        config.h_ORL = h
        config.w_ORL = w
        fovea_center = getattr(self, 'fovea_center', (h // 2, w // 2))

        fig1 = plt.figure()
        plt.imshow(img_thickness, cmap=ccmap, vmin=0, vmax=300)
        plt.axis('off')
        plt.close(fig1)

        fig, ax = plt.subplots()
        ax.imshow(img_thickness, cmap=ccmap, vmin=0, vmax=300)
        ax.axis('off')

        circ1 = Circle(fovea_center[::-1], 42, color='black', fill=False, linewidth=0.5)
        circ3 = Circle(fovea_center[::-1], 42 * 3, color='black', fill=False, linewidth=0.5)
        circ5 = Circle(fovea_center[::-1], 42 * 5, color='black', fill=False, linewidth=0.5)

        plt.imshow(img_thickness, cmap=ccmap, vmin=0, vmax=300)
        plt.xlabel('Fast Scan')
        plt.ylabel('Slow Scan')
        plt.title('ORL Thickness Map Preview')
        bar = plt.colorbar()
        bar.set_label('ORL Thickness (um)')

        img = self.canvas_to_array(plt.gcf())
        plt.close(fig)
        self.addPixmap(self.get_pix_map(img))
        img_w = state.img_obj.img_width
        pen = QPen(Qt.white, 1)
        self.frame = frame
        self.addLine(0, frame * (370 / 499) + 58, img_w,
                     frame * (370 / 499) + 58, pen)

    def canvas_to_array(self, fig):
        '''Convert the plt canvas to array (Robust for RGB and RGBA)'''
        fig.canvas.draw()

        # Get dimensions
        w, h = fig.canvas.get_width_height()

        # Get the raw buffer
        try:
            # Modern Matplotlib preferred method
            buf = fig.canvas.buffer_rgba()
        except AttributeError:
            # Fallback for older versions
            buf = fig.canvas.tostring_rgb()

        figData = np.frombuffer(buf, dtype=np.uint8)

        # Calculate channels (depth) dynamically to prevent reshape errors
        # Avoid division by zero
        if w * h == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)

        depth = len(figData) // (w * h)

        # Reshape based on the actual depth (3 for RGB, 4 for RGBA)
        fData = figData.reshape((h, w, depth))

        # If we got RGBA (4 channels), drop the Alpha channel to get RGB
        if depth == 4:
            fData = fData[:, :, :3]

        return fData

    def get_pix_map(self, img):
        """ transfer the numpy array to the QT pixmap """
        if np.issubdtype(img.dtype, np.uint16):
            img = np.uint8(img / 256.0)
        img_pixmap = QPixmap.fromImage(
            qimage2ndarray.array2qimage(img, normalize=True))
        return img_pixmap


class GraphicsScenePreview(GraphicsScene2d):
    """
    For the segmentation preview window
    """

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        # self.setSceneRect(-100, -100, 200, 200)

        # self.lastPoint = QPoint()
        # self.copy_line_num = 1  # default value of the line parpageation
        # self.line_fit_status = True
        self.fit_bounds = 5
        # self.fitRPE = True
        self.drawCurrenFrame(frame=0)

    def drawCurrenFrame(self, frame=10, type=None):
        """
        Draw the OCT B-scan to the scene
        """
        self.clear()
        self.addPixmap(self.get_pix_map(state.img_obj.stru3d[:, :, frame]))

    def drawSegLines(self, lines):
        """
        Draw the lines for current segmentation settings
        Args:
            lines:
        Returns:

        """
        # self.current_layer = current_layer
        # self.frame = frame
        inds = [0,1]
        # print('current lines', inds)

        img_w = lines.shape[0]

        # all layers
        pen = QPen(Qt.red, 1)
        for i in inds:
            for j in range(0, img_w - 1):
                self.addLine(j, lines[j, i], j + 1, lines[j + 1, i], pen)

