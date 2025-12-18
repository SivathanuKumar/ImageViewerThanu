class ChoroidWindow(QtWidgets.QMainWindow, ChoroidWindowUI.Ui_MainWindow):
    """
    Choroid quantification window
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        # display
        self.scene_Thickness = GraphicsScene2d()
        self.graphicsView_Thickness.setScene(self.scene_Thickness)

        self.scene_Vessel = GraphicsScene2d()
        self.graphicsView_Vessel.setScene(self.scene_Vessel)

        self.scene_CVI = GraphicsScene2d()
        self.graphicsView_CVI.setScene(self.scene_CVI)

        # process
        self.pushButton_OpenFile.clicked.connect(
            self.on_button_clicked_load_img)
        self.pushButton_Run.clicked.connect(self.on_button_clicked_run)
        self.pushButton_Save.clicked.connect(
            self.on_button_clicked_save_results)

    def get_current_path(self):
        """ Get the current path of the file, forward it to the lineEdit_OpenPath
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
        elif type == 'image':
            fileName, _ = QFileDialog.getOpenFileName(None, "Open file", url,
                                                      "Image Files (*.jpg *.bmp *.tif *.png)",
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

    def on_button_clicked_load_img(self):
        """
        load image 1 OCTA CC to class member
        :return:
        """

        url = self.lineEdit_OpenPath.text()
        file_url = self.openFileNameDialog(url)
        if os.path.exists(file_url):
            self.filename = file_url
            self.lineEdit_OpenPath.setText(
                os.path.dirname(file_url))  # set the url to the window
            inputs = loadmat(file_url)
            vol = inputs['inputs']
            self.StrucAttnCub = vol[0, 0]['StrucAttnCub']
            self.BMSeg = vol[0, 0]['BMSeg']
            self.choroidSeg = vol[0, 0]['choroidSeg']
            self.ONHseg = vol[0, 0]['ONHseg']

            # self.graphicsView_OCTACC.fitInView(self.scene_OCTACC.sceneRect(), Qt.KeepAspectRatio)
            # self.send_and_display_the_log('Load image: ' + file_url)
        else:
            print('load failed')
            # self.send_and_display_the_log('File does not exist: ' + file_url)

    def on_button_clicked_run(self):
        """
        run the algorithm
        Returns:
        """

        # use the data from the main panel
        try:
            self.StrucAttnCub = img_obj.stru3d
            self.BMSeg = img_obj.layers[:, :, 0]
            self.choroidSeg = img_obj.layers[:, :, 1]
            self.ONHseg = img_obj.layers[:, :, 2]

        except Exception as e:
            print('Load image first. ', e)

        self.cvi, self.mct, self.map_thickness, self.map_vessel, self.map_cvi, self.vol_cvi = choroid_BM_CSI(
            self.StrucAttnCub, self.BMSeg, self.choroidSeg, self.ONHseg)
        self.map_thickness = self.scene_Thickness.img_to_colormap(
            self.map_thickness)
        self.scene_Thickness.update_image(self.map_thickness)
        self.graphicsView_Thickness.fitInView(self.scene_Thickness.sceneRect(),
                                              Qt.KeepAspectRatio)

        self.scene_Vessel.update_image(self.map_vessel)
        self.graphicsView_Vessel.fitInView(self.scene_Vessel.sceneRect(),
                                           Qt.KeepAspectRatio)

        self.map_cvi = self.scene_CVI.img_to_colormap(self.map_cvi)
        self.scene_CVI.update_image(self.map_cvi)
        self.graphicsView_CVI.fitInView(self.scene_CVI.sceneRect(),
                                        Qt.KeepAspectRatio)

        self.lineEdit_cvi.setText(str(round(self.cvi, 3)))
        self.lineEdit_mct.setText(str(round(self.mct, 3)))

    def on_button_clicked_save_results(self):
        """
        save the figrues to the drive
        Returns:
        """
        dirname = os.path.dirname(img_obj.save_path)
        base = os.path.basename(img_obj.save_path)
        file_name = os.path.splitext(base)[0]

        url = os.path.join(dirname, file_name)
        if not os.path.exists(url):
            os.makedirs(url)

        # save seg layers
        file_name_mat = os.path.join(url, 'layers.mat')
        file_name_npy = os.path.join(url, 'layers.npy')
        img_obj.save_layers(file_name_mat)
        img_obj.save_layers(file_name_npy)
        print('segmentation files saved to :', url)

        # save images
        imwrite(os.path.join(url, 'choroid_thickness.png'), self.map_thickness)
        imwrite(os.path.join(url, 'choroid_vessel.png'), self.map_vessel)
        imwrite(os.path.join(url, 'choroid_map_cvi.png'), self.map_cvi)

        # save numbers
        result = pd.DataFrame([[self.mct, self.cvi]], columns=['mct', 'cvi'])
        result.to_csv(os.path.join(url, 'choroid_result.csv'))

        # save volume
        img_obj.save_video(self.vol_cvi, os.path.join(url, 'choroid_cvi.dcm'))
