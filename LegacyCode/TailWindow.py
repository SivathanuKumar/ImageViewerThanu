class TailWindow(QtWidgets.QMainWindow, TailWindowUI.Ui_MainWindow):
    '''
    The tail removel window
    '''

    def __init__(self, stru=None, flow=None):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self.stru = stru
        self.flow = flow

        # display
        self.scene_XY = GraphicsSceneTail()
        self.graphicsView_XY.setScene(self.scene_XY)
        self.scene_YZ = GraphicsSceneTail()
        self.graphicsView_YZ.setScene(self.scene_YZ)
        self.scene_XZ = GraphicsSceneTail()
        self.graphicsView_XZ.setScene(self.scene_XZ)
        self.scene_TailOri = GraphicsSceneTail()
        self.graphicsView_TailOri.setScene(self.scene_TailOri)
        self.scene_TailRec = GraphicsSceneTail()
        self.graphicsView_TailRec.setScene(self.scene_TailRec)
        #
        # # button
        self.pushButton_Update.clicked.connect(self.on_click_update)
        self.pushButton_Preview.clicked.connect(self.on_click_preview)
        self.pushButton_Save.clicked.connect(self.on_click_save_parameters)
        self.pushButton_BatchOpen.clicked.connect(
            self.on_button_clicked_open_batch)
        self.pushButton_LoadSingle.clicked.connect(
            self.on_button_clicked_load_single)
        # self.pushButton_OpenSaveFolder

        self.pushButton_RunSingle.clicked.connect(
            self.on_button_clicked_run_single)
        self.pushButton_RunBatch.clicked.connect(
            self.on_button_clicked_run_batch)

        # init
        if stru is not None:
            self.init_parameters()

        self.on_click_update()

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
            # self.lineEdit_SavePath.setText(url)

            print('open batch folder = ', url)

            file_df = self.search_files_in_folder(url)
            self.file_df = file_df  # put this to the class
            self.batch_url = url

            # clean previous logs
            self.listWidget_BatchList.clear()

            # display
            self.display_batch_files_to_list(file_df)
        except Exception as e:
            print('Load failed:', e)

    def search_files_in_folder(self, url):
        """ get files by file pattern
        Args:
            url: the url of folders
        Return:
            dataframe
        """
        '''
        XXX_Stru (folder)
            - stru_flatten.dcm
            - flow_flatten.dcm
        -
        url = '../data'
        '''
        try:
            folder_pattern = self.lineEdit_FolderPattern.text()
            stru_pattern = self.lineEdit_StruPattern.text()
            flow_pattern = self.lineEdit_FlowPattern.text()

            file_df = pd.DataFrame(columns=['folder', 'stru', 'flow'])

            # file_list = os.listdir(url)
            # list all folders in the url
            file_list = glob(url + '/*/')

            for file in file_list:
                # print(file)
                folder_name = os.path.basename(
                    os.path.dirname(file))  # get the folder name
                loc = folder_name.find(folder_pattern)
                # file is the stru
                if loc != -1:
                    file_df = file_df.append({'folder': file},
                                             ignore_index=True)
                    stru_name = os.path.join(file,
                                             stru_pattern)  # get the image file name
                    flow_name = os.path.join(file, flow_pattern)

                    # put the info to dataframe
                    row_index = file_df.shape[0] - 1
                    file_df.iloc[row_index, 0] = folder_name
                    if os.path.exists(stru_name):
                        file_df.iloc[row_index, 1] = stru_name
                    if os.path.exists(flow_name):
                        file_df.iloc[row_index, 2] = flow_name

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
                'Stru: ' + os.path.basename(str(row['stru'])))
            self.listWidget_BatchList.addItem(
                'Flow: ' + os.path.basename(str(row['flow'])))

    # %%
    def init_parameters(self):
        """
        set the init parameters, full range
        :return:
        """
        try:
            [img_depth, img_w, img_f] = self.stru.shape
            self.lineEdit_XHigh.setText(str(img_w))
            self.lineEdit_YHigh.setText(str(img_f))
            self.lineEdit_ZHigh.setText(str(img_depth))

            self.spinBox_Slice.setMinimum(0)
            self.spinBox_Slice.setMaximum(img_f - 1)
        except Exception as e:
            print('Init failed: ', e)

    def get_parameters(self):
        """
        get the parameters from UI
        :return:
        """
        self.XrangeL = int(self.lineEdit_XLow.text())
        self.XrangeH = int(self.lineEdit_XHigh.text())
        self.YrangeL = int(self.lineEdit_YLow.text())
        self.YrangeH = int(self.lineEdit_YHigh.text())
        self.ZrangeL = int(self.lineEdit_ZLow.text())
        self.ZrangeH = int(self.lineEdit_ZHigh.text())

        self.Amp = float(self.lineEdit_Amp.text())
        self.Dec = float(self.lineEdit_Dec.text())
        self.BitDepth = float(self.lineEdit_BitDepth.text())
        self.rg_L = float(self.lineEdit_Rg_L.text())
        self.rg_H = float(self.lineEdit_Rg_H.text())

        self.slice_num = int(self.spinBox_Slice.value())

    def update_display_slices(self):
        """
        update the scene, XY, YZ, XZ
        :return:
        """
        try:
            if (self.stru is not None) and (self.flow is not None):
                # XY
                slice_XY = 300
                slice_YZ = 10
                slice_XZ = self.slice_num
                # img_XY = np.squeeze(self.stru[slice_XY,...])
                img_XY = np.mean(self.stru, axis=0)

                img_YZ = np.squeeze(self.stru[:, slice_YZ, :])

                img_XZ = np.squeeze(self.stru[:, :, slice_XZ])

                # img_tail_flow = np.squeeze(img_obj.flow3d[:, :, slice_XZ])
                # img_tail_stru = np.squeeze(img_obj.stru3d[:, :, slice_XZ])

                # img_tail = np.squeeze(self.preview_tail(img_XZ_stru, img_XZ_flow))

                # self.scene_XY.update_image(img_XY)
                self.scene_XY.draw_rect(img_XY, self.YrangeL, self.XrangeL,
                                        self.YrangeH, self.XrangeH)
                self.graphicsView_XY.fitInView(self.scene_XY.sceneRect(),
                                               Qt.KeepAspectRatio)

                self.scene_YZ.draw_rect(img_YZ, self.YrangeL, self.ZrangeL,
                                        self.YrangeH, self.ZrangeH)
                self.graphicsView_YZ.fitInView(self.scene_YZ.sceneRect(),
                                               Qt.KeepAspectRatio)

                self.scene_XZ.draw_rect(img_XZ, self.XrangeL, self.ZrangeL,
                                        self.XrangeH, self.ZrangeH)
                self.graphicsView_XZ.fitInView(self.scene_XZ.sceneRect(),
                                               Qt.KeepAspectRatio)

                # self.scene_Tail.update_image(img_tail)
                # self.graphicsView_Tail.fitInView(self.scene_Tail.sceneRect(), Qt.KeepAspectRatio)
        except Exception as e:
            print('Cannot update display: ', e)

    def update_display_tails(self):
        """
        update the scene, tailOri, tailRec
        :return:
        """
        try:
            slice_XZ = self.slice_num
            img_tail_stru = np.squeeze(
                self.stru[self.ZrangeL:self.ZrangeH, self.XrangeL: self.XrangeH,
                slice_XZ])
            img_tail_flow = np.squeeze(
                self.flow[self.ZrangeL:self.ZrangeH, self.XrangeL: self.XrangeH,
                slice_XZ])
            img_tail_rec = np.squeeze(
                self.preview_tail(img_tail_stru, img_tail_flow))

            self.scene_TailOri.update_image(img_tail_flow)
            self.graphicsView_TailOri.fitInView(self.scene_TailOri.sceneRect(),
                                                Qt.KeepAspectRatio)
            self.scene_TailRec.update_image(img_tail_rec)
            self.graphicsView_TailRec.fitInView(self.scene_TailRec.sceneRect(),
                                                Qt.KeepAspectRatio)
        except Exception as e:
            print('Cannot update display tail: ', e)

    def preview_tail(self, img_stru, img_flow):
        """
        call the tail reduction funciton
        :param img:
        :return:
        """
        img_tail = tail_reduction_parallel(img_flow, BitDepth=self.BitDepth,
                                           amp=self.Amp, dec=self.Dec,
                                           rg_L=self.rg_L,
                                           rg_H=self.rg_H)
        return img_tail

    def on_click_update(self):
        """
        update the trim boundry
        :return:
        """
        self.get_parameters()
        self.update_display_slices()

    def on_click_preview(self):
        """
        display the preview image
        :return:
        """
        try:
            self.get_parameters()
            self.update_display_slices()
            self.update_display_tails()
        except Exception as e:
            print('Preview failed: ', e)

    def on_click_save_parameters(self):
        """
        save the parameters to main panel
        :return:
        """
        self.get_parameters()
        img_obj.tail_xL = self.XrangeL
        img_obj.tail_xH = self.XrangeH
        img_obj.tail_yL = self.YrangeL
        img_obj.tail_yH = self.YrangeH
        img_obj.tail_zL = self.ZrangeL
        img_obj.tail_zH = self.ZrangeH

        img_obj.tail_amp = self.Amp
        img_obj.tail_dec = self.Dec
        img_obj.tail_bitdepth = self.BitDepth

        print('parameters saved')

    def on_button_clicked_load_single(self):
        """
        Load single ground files to the interface
        :return:
        """
        try:
            # get the file names from list
            row = self.listWidget_BatchList.currentRow()

            row_num = row // 4
            folder = self.file_df.iloc[row_num, 0]
            stru_url = self.file_df.iloc[row_num, 1]
            flow_url = self.file_df.iloc[row_num, 2]

            self.load_stru_flow(stru_url, flow_url)
            self.init_parameters()
            self.update_display_slices()
            self.update_display_tails()

        except Exception as e:
            print(e)

    def on_button_clicked_run_single(self):
        """Run the select files from the list
        :return:
        """
        try:
            # get the file names from list
            row = self.listWidget_BatchList.currentRow()

            row_num = row // 4
            folder = self.file_df.iloc[row_num, 0]
            stru_url = self.file_df.iloc[row_num, 1]
            flow_url = self.file_df.iloc[row_num, 2]

            self.process_single_file(stru_url, flow_url)
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
                stru_url = self.file_df.iloc[row_num, 1]
                flow_url = self.file_df.iloc[row_num, 2]

                self.process_single_file(stru_url, flow_url)
        except Exception as e:
            print(e)

    def load_stru_flow(self, url_s, url_f):

        self.stru = img_obj.load_3d_data(url_s)
        self.flow = img_obj.load_3d_data(url_f)

    def process_single_file(self, stru_url, flow_url):
        """
        run the algorithm
        :return:
        """
        try:
            self.load_stru_flow(stru_url, flow_url)
            # self.init_parameters()

            self.get_parameters()
            self.update_display_slices()
            self.update_display_tails()

            img_tail_stru = np.squeeze(
                self.stru[self.ZrangeL:self.ZrangeH, self.XrangeL: self.XrangeH,
                self.YrangeL:self.YrangeH])
            img_tail_flow = np.squeeze(
                self.flow[self.ZrangeL:self.ZrangeH, self.XrangeL: self.XrangeH,
                self.YrangeL:self.YrangeH])

            # run single volume
            img_tail_rec = self.preview_tail(img_tail_stru, img_tail_flow)

            # url = os.path.basename()
            save_url = os.path.join(os.path.dirname(stru_url),
                                    'tail_remove.dcm')
            self.save_volumes(img_tail_rec, save_url)
        except Exception as e:
            print('Can not process file: ', e)
        return img_tail_rec

    def save_volumes(self, img, url):
        """
        save the tail reduciton results to the url
        :return:
        """
        img_obj.save_video(img, url)

