#

from PyQt5 import QtCore, QtGui, QtWidgets

# setx path "%path%;c:\directoryPath"
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import pydicom
from OCTpy import Oct3Dimage


class ImageRegistration:
    """
    """
    def __init__(self):
        self.regi_type = '2D'
        self.nV = 0
        self.rigid_regi = False
        self.affine_regi = False
        self.bspine_regi = False
        self.diff_img_paths = []
        self.mov_img_paths = []
        self.img_obj = Oct3Dimage()


    #%% set global status
    def set_nV(self, num):
        self.nV = num

    def set_rigid_regi(self, status):
        assert(type(status)==bool)
        self.rigid_regi = status

    def set_affine_regi(self, status):
        assert(type(status)==bool)
        self.affine_regi = status

    def set_bspine_regi(self, status):
        assert(type(status)==bool)
        self.bspine_regi = status

    #%%
    '''
    def get_information(self):
        """ Get the image infromation
        :return:
        """
        if self.regi_type == '2d':
     '''

    #%%
    def imread(self, url):
        if self.regi_type == '2D':
            img = imageio.imread(url)
        elif self.regi_type == '3D':
            img = self.img_obj.load_3d_data(url)
        return img



    #%%
    def run_registration(self, fix_img_path, mov_img_path, out_path, para_file_path):
        """ Run the registration command
        :return:
        """
        '''
        fix_img_path = ''
        mov_img_path = ''
        out_path = ''
        para_file_path = ''
        '''
        #chenk the path of parameter files
        if os.path.exists(para_file_path):
            regi_command = 'elastix -f ' + fix_img_path + ' -m ' + mov_img_path + ' -out ' + out_path + ' -p ' + para_file_path
        else:
            print('Use default parameters:  ')
            cwd = os.getcwd()
            para_file_path = os.path.join(cwd, os.path.basename(para_file_path))
            regi_command = 'elastix -f ' + fix_img_path + ' -m ' + mov_img_path + ' -out ' + out_path + ' -p ' + para_file_path

        print('run command:', regi_command)
        os.system(regi_command)

    def make_dirs(self, img_path):
        """
        Make dirs for the registration
        """
        dir_name = os.path.dirname(img_path)
        # translation
        trans_dir = os.path.join(dir_name, 'trans')
        if not os.path.exists(trans_dir):
            os.mkdir(trans_dir)

        # affine
        affine_dir = os.path.join(dir_name, 'affine')
        if not os.path.exists(affine_dir):
            os.mkdir(affine_dir)

        # bspine
        bspine_dir = os.path.join(dir_name, 'bspine')
        if not os.path.exists(bspine_dir):
            os.mkdir(bspine_dir)

        # registered
        regi_dir = os.path.join(dir_name, 'registered')
        if not os.path.exists(regi_dir):
            os.mkdir(regi_dir)

        # averaged
        avg_dir = os.path.join(dir_name, 'average')
        if not os.path.exists(avg_dir):
            os.mkdir(avg_dir)

    def split_file(self, img, num):
        """ Split the image to single files for the repeat volume protocols
        :return:
        """
        # The size of the image is X*(N*Y)
        if self.regi_type == '2D':
            print('split the 2D files')
            self.img_size_slow = int(img.shape[1]/num)
            # if the data not fully divide by the size of single image

            for i in range(0, num):
                img_cut = img[:, i*self.img_size_slow: (i+1)*self.img_size_slow]
                url = os.path.join(self.data_url, 'split_image_'+str(i)+'.png')
                imageio.imwrite(url, img_cut)

                if i>=1:
                    self.mov_img_paths.append(url)

        elif self.regi_type == '3D':
            self.img_size_slow = int(img.shape[2]/num)
            print('split 3D files with shape', img.shape, ' cut frames=', self.img_size_slow)

            for i in range(0, num):
                img_cut = img[:, :, i*self.img_size_slow: (i+1)*self.img_size_slow]
                url = os.path.join(self.data_url, 'split_image_'+str(i)+'.dcm')
                img_cut = self.img_obj.save_video(np.uint16(img_cut / 2), url) # change the range from uint16 to 0-32767

                if i>=1:
                    self.mov_img_paths.append(url)




    def regi_2d_2img(self, fix_img_path, mov_img_path):
        """
        :param fix_img_path:
        :param mov_img_path:
        :return:
        """
        if self.regi_type == '2D':
            # make dirctoarty
            self.make_dirs(fix_img_path)
            imgs_dir = os.path.dirname(fix_img_path)

            # trans
            para_file_path = os.path.join(imgs_dir, 'parameters_Translation.txt')
            para_file_path = self.check_para_files(para_file_path)
            out_path = os.path.join(imgs_dir, 'trans')
            self.run_registration(fix_img_path, mov_img_path, out_path, para_file_path)

            # affine
            para_file_path = os.path.join(imgs_dir, 'parameters_Affine.txt')
            para_file_path = self.check_para_files(para_file_path)
            mov_img_path = os.path.join(out_path, 'result.0.mhd')
            out_path = os.path.join(imgs_dir, 'affine')
            self.run_registration(fix_img_path, mov_img_path, out_path, para_file_path)

            # bspine
            mov_img_path = os.path.join(out_path, 'result.0.mhd')
            para_file_path = os.path.join(imgs_dir, 'parameters_Bspline.txt')
            para_file_path = self.check_para_files(para_file_path)
            out_path = os.path.join(imgs_dir, 'bspine')
            self.run_registration(fix_img_path, mov_img_path, out_path, para_file_path)
            bspine_img = os.path.join(out_path, 'result.0.dcm')

        elif self.regi_type == '3D':
            # make dirctoarty
            self.make_dirs(fix_img_path)
            imgs_dir = os.path.dirname(fix_img_path)

            # trans
            para_file_path = os.path.join(imgs_dir, 'parameters_Translation_3D.txt')
            para_file_path = self.check_para_files(para_file_path)
            out_path = os.path.join(imgs_dir, 'trans')
            self.run_registration(fix_img_path, mov_img_path, out_path, para_file_path)

            # affine
            para_file_path = os.path.join(imgs_dir, 'parameters_Affine_3D.txt')
            para_file_path = self.check_para_files(para_file_path)
            mov_img_path = os.path.join(out_path, 'result.0.mhd')
            out_path = os.path.join(imgs_dir, 'affine')
            self.run_registration(fix_img_path, mov_img_path, out_path, para_file_path)

            # bspine
            mov_img_path = os.path.join(out_path, 'result.0.mhd')
            para_file_path = os.path.join(imgs_dir, 'parameters_Bspline_3D.txt')
            para_file_path = self.check_para_files(para_file_path)
            out_path = os.path.join(imgs_dir, 'bspine')
            self.run_registration(fix_img_path, mov_img_path, out_path, para_file_path)
            bspine_img = os.path.join(out_path, 'result.0.dcm')


    def regi_2d_N_imgs(self, fix_img_path, mov_img_paths, N):
        """ There N images in the folder, with the sequence image Type_image_0.png
        and the parameter files are in same folder
        fix_img_path:
        mov_img_paths:
        N: number of average

        :return:
        """
        # loop registration
        if self.regi_type == '2D':

            for i in range(1, N):
                # make dir
                paras_dir = os.path.dirname(fix_img_path)
                imgs_dir = os.path.join(paras_dir, 'out'+str(i))
                if not os.path.exists(imgs_dir):
                    os.mkdir(imgs_dir)
                self.make_dirs(imgs_dir+'/')

                # rigid
                para_file_path = os.path.join(paras_dir, 'parameters_Translation.txt')
                para_file_path = self.check_para_files(para_file_path)
                out_path = os.path.join(imgs_dir, 'trans')
                mov_img_path = mov_img_paths[i-1]
                self.run_registration(fix_img_path, mov_img_path, out_path, para_file_path)

                # affine
                mov_img_path = os.path.join(out_path, 'result.0.mhd')
                para_file_path = os.path.join(paras_dir, 'parameters_Affine.txt')
                para_file_path = self.check_para_files(para_file_path)
                out_path = os.path.join(imgs_dir, 'affine')
                self.run_registration(fix_img_path, mov_img_path, out_path, para_file_path)

                # bspine
                mov_img_path = os.path.join(out_path, 'result.0.mhd')
                para_file_path = os.path.join(paras_dir, 'parameters_Bspline.txt')
                para_file_path = self.check_para_files(para_file_path)
                out_path = os.path.join(imgs_dir, 'bspine')
                self.run_registration(fix_img_path, mov_img_path, out_path, para_file_path)


        elif self.regi_type == '3D':
            for i in range(1, N):
                # make dir
                paras_dir = os.path.dirname(fix_img_path)
                imgs_dir = os.path.join(paras_dir, '3Dout' + str(i))    #for 3D, the folder name is 3DoutN
                if not os.path.exists(imgs_dir):
                    os.mkdir(imgs_dir)
                self.make_dirs(imgs_dir + '/')

                # rigid
                para_file_path = os.path.join(paras_dir, 'parameters_Translation_3D.txt')
                para_file_path = self.check_para_files(para_file_path)
                out_path = os.path.join(imgs_dir, 'trans')
                mov_img_path = mov_img_paths[i - 1]
                self.run_registration(fix_img_path, mov_img_path, out_path, para_file_path)

                # affine
                mov_img_path = os.path.join(out_path, 'result.0.mhd')
                para_file_path = os.path.join(paras_dir, 'parameters_Affine_3D.txt')
                para_file_path = self.check_para_files(para_file_path)
                out_path = os.path.join(imgs_dir, 'affine')
                self.run_registration(fix_img_path, mov_img_path, out_path, para_file_path)

                # bspline
                mov_img_path = os.path.join(out_path, 'result.0.mhd')
                para_file_path = os.path.join(paras_dir, 'parameters_Bspline_3D.txt')
                para_file_path = self.check_para_files(para_file_path)
                out_path = os.path.join(imgs_dir, 'bspine')
                self.run_registration(fix_img_path, mov_img_path, out_path, para_file_path)

        self.result_dir = paras_dir


    def move_imgs(self):
        """move the images from
        :return:
        """
        # move the registered image to the folder



    def make_average(self, fix_img_path, N, save_name=''):
        """ Make the avergae results
        :param fix_img_path: the fix images path
        :param N: total number of repeat volumes
        :return:
        """


        if self.regi_type == '2D':
            # read fix images path
            img_ref = imageio.imread(fix_img_path).astype('float')

            # read all registered images
            img_sum = np.copy(img_ref)
            for i in range(1, N):
                regi_img_path = os.path.join(self.result_dir, 'out'+str(i))
                bspine_path = os.path.join(regi_img_path, 'bspine')
                bspine_img = os.path.join(bspine_path, 'result.0.dcm')

                img_regi = pydicom.dcmread(bspine_img).pixel_array.astype('float')
                img_sum = img_sum + img_regi

                # save the error image
                img_diff = self.check_diff(img_ref, img_regi)
                diff_img_path = os.path.join(regi_img_path, 'registered', 'error.png')
                self.diff_img_paths.append(diff_img_path)
                imageio.imwrite(diff_img_path, img_diff)


            # make the average result
            img_ave = img_sum/N
            # reset the dynamic range

            rg_l = np.percentile(img_ave, 0.1)
            rg_h = np.percentile(img_ave, 99.99)

            img_ave[img_ave < rg_l] = rg_l
            img_ave[img_ave > rg_h] = rg_h

            img_unit = (img_ave - rg_l)/(rg_h - rg_l) # range 0-1
            img_ave_save = np.uint8(img_unit*255)
            ave_file_name = save_name + '_averaged.png'
            ave_img_path = os.path.join(self.result_dir, ave_file_name)
            imageio.imwrite(ave_img_path, img_ave_save)

        # for 3D
        elif self.regi_type == '3D':
            # read fix images path
            img_ref = self.img_obj.load_dicom_file(fix_img_path).astype('float')

            # read all registered images
            img_sum = np.copy(img_ref)
            for i in range(1, N):
                regi_img_path = os.path.join(self.result_dir, '3Dout' + str(i))
                bspine_path = os.path.join(regi_img_path, 'bspine')
                bspine_img = os.path.join(bspine_path, 'result.0.dcm')

                img_regi = self.img_obj.load_3d_data(bspine_img).astype('float')
                print('img_regi shape,', img_regi.shape)
                img_sum = img_sum + img_regi


                # save the error image
                # img_diff = self.check_diff(img_ref, img_regi)
                # diff_img_path = os.path.join(regi_img_path, 'registered', 'error.png')
                # self.diff_img_paths.append(diff_img_path)
                # imageio.imwrite(diff_img_path, img_diff)

            # make the average result
            img_ave = img_sum / N
            # reset the dynamic range

            rg_l = np.percentile(img_ave, 0.1)
            rg_h = np.percentile(img_ave, 99.99)

            img_ave[img_ave < rg_l] = rg_l
            img_ave[img_ave > rg_h] = rg_h

            img_unit = (img_ave - rg_l) / (rg_h - rg_l)  # range 0-1
            img_ave_save = np.uint16(img_unit * 65535)


            ave_file_name1 = save_name + '_averaged.avi'
            ave_img_path = os.path.join(self.result_dir, ave_file_name1)
            self.img_obj.save_video(img_ave_save, ave_img_path)


            ave_file_name = save_name + '_averaged.dcm'
            ave_img_path = os.path.join(self.result_dir, ave_file_name)

            self.img_obj.save_video(img_ave_save, ave_img_path)
            # imageio.imwrite(ave_img_path, img_ave_save)



        print('averaging ', N, ' images')

    def check_diff(self, img_ref, img_reg):
        """

        :return:
        """
        img_diff = img_ref.astype('float') - img_reg.astype('float')
        return np.abs(img_diff)
        print('check the difference of the imagse')

    def check_the_demension(file_name):
        """get the size of the image

        :return:
        """
        if file_name.endwith('.jpg'):
            print('2d')

    def check_para_files(self, url):
        """
        check the parameter files
        :param url:
        :return:
        """
        if os.path.exists(url):
            return url
        else:
            file_name = os.path.basename(url)
            self.base_url = os.getcwd()
            new_url = os.path.join(self.base_url, file_name)
            print('cannot find the parameter files in local folder, use the: ', new_url)
            return new_url


    def download_parameter_files(self):
        '''
        :return:
        '''
        print('download from')

























