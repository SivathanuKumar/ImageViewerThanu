class RegiWindow(QtWidgets.QMainWindow, RegiWindowUI.Ui_MainWindow):
    '''
    Manual segementaiton mode
    '''

    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self.pushButton_BatchOpen.clicked.connect(
            self.on_button_clicked_open_batch)

        self.pushButton_LoadStruRef.clicked.connect(
            self.on_button_clicked_load_stru_ref)
        # self.pushButton_LoadFlow.clicked.connect(self.on_button_clicked_load_flow)
        self.pushButton_LoadStruMov.clicked.connect(
            self.on_button_clicked_load_stru_mov)

        # spinbox
        self.spinBox_ImageGroup.valueChanged.connect(self.display_img_groups)
        self.spinBox_nV.valueChanged.connect(self.init_range)
        self.init_range()

        # run the registration
        self.pushButton_Run.clicked.connect(self.run_registration)

        # display
        self.scene_ref = GraphicsScene2d()
        self.graphicsView_Ref.setScene(self.scene_ref)
        self.scene_mov = GraphicsScene2d()
        self.graphicsView_Mov.setScene(self.scene_mov)
        self.scene_ave = GraphicsScene2d()
        self.graphicsView_Ave.setScene(self.scene_ave)
        self.scene_error = GraphicsScene2d()
        self.graphicsView_Error.setScene(self.scene_error)

        self.global_scale = 1.0

        self.connect_to_scene()

        # scale
        self.pushButton_SliceZoomIn.clicked.connect(
            self.on_button_clicked_scale_slice_zoom_in)
        self.pushButton_ZoomToFit.clicked.connect(
            self.on_button_clicked_scale_to_fit)
        self.pushButton_SliceZoomOut.clicked.connect(
            self.on_button_clicked_scale_slice_zoom_out)

        # batch
        self.pushButton_UpdateRule.clicked.connect(
            self.on_button_clicked_open_batch)
        self.pushButton_RunSingle.clicked.connect(
            self.on_button_clicked_run_single)
        self.pushButton_RunBatch.clicked.connect(
            self.on_button_clicked_run_batch)

        # 2D/3D
        # self.radioButton_2D.toggled.connect(self.on_radio_2d_clicked)
        # self.radioButton_2D.toggled.connect(self.on_radio_2d_clicked)

        # self.pushButton_UpdateRule.clicked.connect(self.on_button_clicked_open_batch)

    def init_range(self):
        self.spinBox_ImageGroup.setMaximum(self.spinBox_nV.value() - 1)

    def on_radio_2d_clicked(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.reg_obj.regi_type = '2D'

    def radio_style_check(self):
        try:
            if self.radioButton_2D.isChecked():
                self.reg_obj.regi_type = '2D'
            elif self.radioButton_3D.isChecked():
                self.reg_obj.regi_type = '3D'
        except Exception as e:
            print('Radio Button Warning: ', e)

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

    def on_button_clicked_open_batch(self):
        """ Open the folder for batch processing
        :return:
        """
        url = self.lineEdit_OpenPath.text()
        try:
            url = self.open_folder(url)

            # put the url to the line edit
            url = str(Path(url))
            self.lineEdit_OpenPath.setText(url)
            self.lineEdit_SavePath.setText(url)
        except Exception as e:
            print(e)

    def on_button_clicked_load_stru_ref(self):
        '''Load the stru file
        '''

        # set the mode and parameters
        # init the instance
        self.reg_obj = ImageRegistration()
        self.reg_obj.set_nV(self.spinBox_nV.value())
        self.radio_style_check()

        print('Start loading...')
        url = self.lineEdit_OpenPath.text()
        if self.radioButton_2D.isChecked():
            files = self.openFileNameDialog(url, 'image')
        elif self.radioButton_3D.isChecked():
            files = self.openFileNameDialog(url, 'video')

        if not files:
            print('cannot load')
            self.pop_up_alert('Load failed')
        else:
            self.ori_file = files

            # img_obj.read_stru_data(files)
            self.img_ref_dir = os.path.join(url, files)
            print(self.img_ref_dir, ' has been loaded')
            self.send_and_display_the_log(
                'Load structural file: ' + files)  # send the message
            self.pop_up_alert('Load success')
            # self.radioButton_DisplayStru.setEnabled(True)  # enable the radio button

            self.filename = self.get_file_name(files)
            cwd = str(Path(os.path.dirname(files)))

            # rewrite the cwd
            self.lineEdit_OpenPath.setText(cwd)
            self.lineEdit_SavePath.setText(cwd)

            if self.radioButton_2D.isChecked():
                # update display, single file
                if self.radioButton_SingleFile.isChecked():
                    self.img_ref = self.reg_obj.imread(self.img_ref_dir)


                # multiple images
                elif self.radioButton_MultiFile.isChecked():
                    print('Multi Files, 2D')

                    # splict the files
                    nV = self.spinBox_nV.value()
                    self.reg_obj.set_nV(nV)
                    img = self.reg_obj.imread(self.img_ref_dir)
                    self.reg_obj.data_url = cwd
                    self.send_and_display_the_log(
                        'Split the file to ' + str(nV) + ' parts')
                    self.reg_obj.split_file(img, nV)
                    self.img_ref_dir = os.path.join(cwd, 'split_image_0.png')
                    self.img_ref = self.reg_obj.imread(self.img_ref_dir)

                    # display the 1st moving image
                    self.img_mov_dir = os.path.join(cwd, 'split_image_1.png')
                    self.img_mov = self.reg_obj.imread(self.img_mov_dir)
                    self.scene_mov.update_image(self.img_mov)
                    self.graphicsView_Mov.fitInView(self.scene_mov.sceneRect(),
                                                    Qt.KeepAspectRatio)

                # display the ref image
                self.scene_ref.update_image(self.img_ref)
                self.graphicsView_Ref.fitInView(self.scene_ref.sceneRect(),
                                                Qt.KeepAspectRatio)

                # group display
            elif self.radioButton_3D.isChecked():
                # print('current_regi_type = ', self.reg_obj.regi_type)
                # update display, single file
                if self.radioButton_SingleFile.isChecked():
                    self.img_ref = self.reg_obj.imread(self.img_ref_dir)


                # multiple images
                elif self.radioButton_MultiFile.isChecked():
                    print('Multi Files, 3D')

                    # splict the files
                    nV = self.spinBox_nV.value()
                    self.reg_obj.set_nV(nV)
                    img = self.reg_obj.imread(self.img_ref_dir)
                    print('image shape=', img.shape)
                    self.reg_obj.data_url = cwd
                    self.send_and_display_the_log(
                        'Split the file to ' + str(nV) + ' parts...')
                    self.reg_obj.split_file(img, nV)
                    self.img_ref_dir = os.path.join(cwd, 'split_image_0.dcm')
                    self.img_ref = self.reg_obj.imread(self.img_ref_dir)

                    # display the 1st moving image
                    self.img_mov_dir = os.path.join(cwd, 'split_image_1.dcm')
                    self.img_mov = self.reg_obj.imread(self.img_mov_dir)
                    # self.scene_mov.update_image(self.img_mov)
                    # self.graphicsView_Mov.fitInView(self.scene_mov.sceneRect(), Qt.KeepAspectRatio)

                # display the ref image
                # self.scene_ref.update_image(self.img_ref)
                # self.graphicsView_Ref.fitInView(self.scene_ref.sceneRect(), Qt.KeepAspectRatio)

                # group display

    def on_button_clicked_load_stru_mov(self):
        '''Load the stru file
        '''
        print('Start loading')
        url = self.lineEdit_OpenPath.text()

        # load 2D images
        if self.radioButton_2D.isChecked():
            files = self.openFileNameDialog(url, 'image')
            if not files:
                print('cannot load')
                self.pop_up_alert('Load failed')
            else:
                # img_obj.read_stru_data(files)
                self.img_mov_dir = os.path.join(url, files)
                print(self.img_ref_dir, ' has been loaded')
                self.send_and_display_the_log(
                    'Load moving file: ' + files)  # send the message
                self.pop_up_alert('Load success')
                # self.radioButton_DisplayStru.setEnabled(True)  # enable the radio button

                self.filename = self.get_file_name(files)

                cwd = str(Path(os.path.dirname(files)))
                # rewrite the cwd
                self.lineEdit_OpenPath.setText(cwd)
                self.lineEdit_SavePath.setText(cwd)

                # update display
                self.img_mov = self.reg_obj.imread(self.img_mov_dir)
                self.scene_mov.update_image(self.img_mov)
                self.graphicsView_Mov.fitInView(self.scene_mov.sceneRect(),
                                                Qt.KeepAspectRatio)

        # load 3D volumes
        elif self.radioButton_3D.isChecked():
            files = self.openFileNameDialog(url, 'video')
            if not files:
                print('cannot load')
                self.pop_up_alert('Load failed')
            else:
                # img_obj.read_stru_data(files)
                self.img_mov_dir = os.path.join(url, files)
                print(self.img_ref_dir, ' has been loaded')
                self.send_and_display_the_log(
                    'Load moving file: ' + files)  # send the message
                self.pop_up_alert('Load success')
                # self.radioButton_DisplayStru.setEnabled(True)  # enable the radio button

                self.filename = self.get_file_name(files)

                cwd = str(Path(os.path.dirname(files)))
                # rewrite the cwd
                self.lineEdit_OpenPath.setText(cwd)
                self.lineEdit_SavePath.setText(cwd)

                # update display
                self.img_mov = self.reg_obj.img_obj.load_3d_data(
                    self.img_mov_dir)
                # self.scene_mov.update_image(self.img_mov)
                # self.graphicsView_Mov.fitInView(self.scene_mov.sceneRect(), Qt.KeepAspectRatio)

    def get_file_name(self, files):
        '''Get the file name without extension and the type 'stru'

        :param file:
        :return:
        '''
        base = os.path.basename(files)
        return os.path.splitext(base)[0]

    def pop_up_alert(self, message='Load success'):
        '''Pop up the message box
        '''
        alert = QMessageBox()
        alert.setText(message)
        alert.exec_()

    def send_and_display_the_log(self, text):
        '''Display the logs
        '''
        time = datetime.now().strftime('%m-%d %H:%M:%S -- ')
        str_send = time + text
        self.plainTextEdit_log.appendPlainText(str_send)

    def display_img_groups(self):
        """
        Display the img groups
        :return:
        """

        try:
            num = self.spinBox_ImageGroup.value()
            mov_img_url = self.reg_obj.mov_img_paths[num - 1]
            self.img_mov = self.reg_obj.imread(mov_img_url)
            self.scene_mov.update_image(self.img_mov)
            self.graphicsView_Mov.fitInView(self.scene_mov.sceneRect(),
                                            Qt.KeepAspectRatio)
            try:
                error_img_url = self.reg_obj.diff_img_paths[num - 1]
                self.img_error = self.reg_obj.imread(error_img_url)
                self.scene_error.update_image(self.img_error)
                self.graphicsView_Error.fitInView(self.scene_error.sceneRect(),
                                                  Qt.KeepAspectRatio)
                self.send_and_display_the_log(
                    'Display the image group: ' + str(num))
            except Exception as e:
                print(e)
                self.send_and_display_the_log(
                    'Cannot display error images: ' + error_img_url)
        except Exception as e:
            print(e)

    def on_button_clicked_scale_slice_zoom_in(self):
        '''Zoom in
        '''
        scale = 1.3
        self.update_scale_of_view(scale)
        self.global_scale *= scale

    def on_button_clicked_scale_slice_zoom_out(self):
        '''Zoom out
        '''
        scale = 0.7
        self.update_scale_of_view(scale)
        self.global_scale *= scale

    def on_button_clicked_scale_to_fit(self):
        '''Scale to fit
        '''
        scale = 1.0 / self.global_scale
        self.global_scale = 1.0
        self.update_scale_of_view(scale)

    def update_scale_of_view(self, scale):
        ''' Updates all graphics view
        '''
        self.graphicsView_Ref.scale(scale, scale)
        self.graphicsView_Mov.scale(scale, scale)
        self.graphicsView_Ave.scale(scale, scale)
        self.graphicsView_Error.scale(scale, scale)

    def connect_to_scene(self):
        '''

        :return:
        '''
        # self.scene.update_image(img)

    def run_registration(self):
        """
        click the Run button
        :return:
        """
        try:
            self.send_and_display_the_log('Running...please wait')

            # single file
            if self.radioButton_SingleFile.isChecked():
                img_fix = self.img_ref_dir
                mov_img_paths = list([self.img_mov_dir])
                nV = self.reg_obj.nV
                self.reg_obj.regi_2d_N_imgs(img_fix, mov_img_paths,
                                            nV)  # Path(self.ori_file).stem)
                self.reg_obj.make_average(img_fix, nV)

            # for multi file
            elif self.radioButton_MultiFile.isChecked():
                img_fix = self.img_ref_dir
                mov_img_paths = self.reg_obj.mov_img_paths
                print('img_fix:', img_fix)
                print('mov_img_paths:', mov_img_paths)
                nV = self.reg_obj.nV
                self.reg_obj.regi_2d_N_imgs(img_fix, mov_img_paths, nV)
                self.reg_obj.make_average(img_fix, nV, Path(self.ori_file).stem)

            if self.reg_obj.regi_type == '2D':
                # display the average
                dir_name = os.path.dirname(self.img_ref_dir)
                average_dir = os.path.join(dir_name, 'averaged.png')
                self.img_ave = imread(average_dir)
                self.scene_ave.update_image(self.img_ave)
                self.graphicsView_Ave.fitInView(self.scene_ave.sceneRect(),
                                                Qt.KeepAspectRatio)
                self.send_and_display_the_log('Calculate the averaged image')

                # display the error
                self.img_error = imread(self.reg_obj.diff_img_paths[0])
                self.scene_error.update_image(self.img_error)
                self.graphicsView_Error.fitInView(self.scene_error.sceneRect(),
                                                  Qt.KeepAspectRatio)
                self.send_and_display_the_log(
                    'Display the residual error image')


        except Exception as e:
            print(e)
            self.send_and_display_the_log('Registration failed')

    # %% run batch registration
    def on_button_clicked_open_batch(self):
        """ Open the folder for batch processing
        :return:
        """
        url = self.lineEdit_OpenPath.text()
        try:
            url = self.open_folder(url)

            # put the url to the line edit
            url = str(Path(url))
            self.lineEdit_OpenPath.setText(url)
            self.lineEdit_SavePath.setText(url)

            print('open batch folder = ', url)

            file_df = self.search_files_in_folder(url)
            self.file_df = file_df  # put this to the class
            self.batch_url = url

            # clean previous logs
            self.listWidget_BatchList.clear()

            # display
            self.display_batch_files_to_list(file_df)
        except Exception as e:
            print(e)

    def search_files_in_folder(self, url):
        """ get files by file pattern
        Args:
            url: the url of folders
        Return:
            dataframe
        """
        '''
        XXX_Stru (folder)
            - projs_stru_0.png (with 2 or more scans) 
            - projs_stru_1.png
            - projs_stru_2.png
        -
        url = '../data'
        '''
        try:
            folder_pattern = self.lineEdit_StruPattern.text()
            imgfile_pattern = self.lineEdit_FlowPattern.text()

            file_df = pd.DataFrame(columns=['folder', 'img', 'num'])

            # file_list = os.listdir(url)
            # list all folders in the url
            file_list = glob(url + '/*/')

            for file in file_list:
                # print(file)
                folder_name = os.path.basename(os.path.dirname(file))
                loc = folder_name.find(folder_pattern)
                # file is the stru
                if loc != -1:
                    file_df = file_df.append({'folder': file},
                                             ignore_index=True)
                    img_name = os.path.join(file,
                                            imgfile_pattern)  # get the image file name

                    # put the info to dataframe
                    row_index = file_df.shape[0] - 1
                    file_df.iloc[row_index, 0] = folder_name
                    if os.path.exists(img_name):
                        file_df.iloc[row_index, 1] = img_name
                    file_df.iloc[row_index, 2] = self.spinBox_nV.value()

            print(file_df)
        except Exception as e:
            print(e)
        return file_df

    def display_batch_files_to_list(self, file_df):
        """ display batch files to list
        """
        for index, row in file_df.iterrows():
            self.listWidget_BatchList.addItem('--- Batch ---  ' + str(index))
            self.listWidget_BatchList.addItem(
                'Folder: ' + os.path.basename(str(row['folder'])))
            self.listWidget_BatchList.addItem(
                'Image: ' + os.path.basename(str(row['img'])))
            self.listWidget_BatchList.addItem(
                'N data: ' + os.path.basename(str(row['num'])))

    def on_button_clicked_run_single(self):
        """Run the select files from the list
        :return:
        """
        try:
            # get the file names from list
            row = self.listWidget_BatchList.currentRow()

            row_num = row // 4
            folder = self.file_df.iloc[row_num, 0]
            img_file = self.file_df.iloc[row_num, 1]
            nV = self.file_df.iloc[row_num, 2]

            self.process_single_file(folder, img_file, nV)
        except Exception as e:
            print(e)

    def on_button_clicked_run_batch(self):
        """Run batch files
        :return:
        """
        try:
            for index, row in self.file_df.iterrows():
                row_num = index
                folder = self.file_df.iloc[row_num, 0]
                img_file = self.file_df.iloc[row_num, 1]
                nV = self.file_df.iloc[row_num, 2]

                self.process_single_file(folder, img_file, nV)
        except Exception as e:
            print(e)

    def process_single_file(self, folder, img_file, nV):
        """ Run single fils
        :return:
        """
        print('processing the file: ', folder)
        self.send_and_display_the_log('processing the file: ' + folder)

        # single file
        # if self.radioButton_SingleFile.isChecked():
        #     img_fix = self.img_ref_dir
        #     mov_img_paths = list([self.img_mov_dir])
        #     nV = self.reg_obj.nV
        #     self.reg_obj.regi_2d_N_imgs(img_fix, mov_img_paths, nV)
        #     self.reg_obj.make_average(img_fix, nV)

        if self.radioButton_MultiFile.isChecked():

            # splict the files

            self.reg_obj = ImageRegistration()
            self.reg_obj.set_nV(nV)
            self.radio_style_check()

            if os.path.exists(img_file):
                self.img_ref_dir = img_file
                img = self.reg_obj.imread(self.img_ref_dir)
                cwd = os.path.dirname(img_file)
                self.reg_obj.data_url = cwd
                self.reg_obj.split_file(img, nV)  # split the images
                if self.reg_obj.regi_type == '2D':
                    self.img_ref_dir = os.path.join(cwd, 'split_image_0.png')
                elif self.reg_obj.regi_type == '3D':
                    self.img_ref_dir = os.path.join(cwd, 'split_image_0.dcm')

                # self.img_ref = imread(self.img_ref_dir)
                self.send_and_display_the_log('Split the file')

                img_fix = self.img_ref_dir
                mov_img_paths = self.reg_obj.mov_img_paths
                print('img_fix:', img_fix)
                print('mov_img_paths:', mov_img_paths)
                self.reg_obj.regi_2d_N_imgs(img_fix, mov_img_paths, nV)
                self.reg_obj.make_average(img_fix, nV, Path(img_file).stem)

            else:
                self.send_and_display_the_log('Image file does not exist')
        else:
            self.send_and_display_the_log('Please use the muti file mode')

    def get_average_img(self):
        self.send_and_display_the_log('Calculate the average images')

    def display_ave_img(self):
        self.graphicsView_Ave.fitInView(self.scene_ave.sceneRect(),
                                        Qt.KeepAspectRatio)

    def display_error_img(self):
        """

        :return:
        """

    def on_button_clicked_display_loss(self):
        """ Open a new window for manual segmentation
        """
        try:
            self.Loss = RegiWindow()  # %%
            self.Loss.show()
        except Exception as e:
            print(e)

    # logs
    def send_and_display_the_log(self, text):
        '''Display the logs
        '''
        time = datetime.now().strftime('%m-%d %H:%M:%S -- ')
        str_send = time + text
        self.plainTextEdit_log.appendPlainText(str_send)

