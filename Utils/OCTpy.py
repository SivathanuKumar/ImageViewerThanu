# -*- coding: utf-8 -*-
"""
Copyright: University of Washington
@author: Yuxuan Cheng
yxcheng at uw.edu
"""
import numpy as np
import re
from imageio import imread, imwrite, mimread, get_writer
from scipy.ndimage import gaussian_filter, median_filter
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt
import matplotlib
from LayerSeg import seg_video
from scipy.io import loadmat, savemat
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ImplicitVRLittleEndian
import warnings
import os
import SimpleITK as sitk
from tqdm import tqdm
import math

from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift

import logging
logger = logging.getLogger('ftpuploader')
# %%
class Oct3Dimage:
    '''
    Oct3Dimage is a library that provides an easy interface to read and write
    a wide range of oct image data and the segmentation files. It also contains
    simple image processing algorithms and easy to make more extensions.
    '''
    def __init__(self):
        ''' init
        '''
        # set the flag of existance
        self.exist_flow = False
        self.exist_stru = False
        self.exist_seg = False
        self.save_path = ''

        self.set_tail_vars()

        self.flatten_loc = 200
        
        print('current path = ', os.getcwd())

# %%
    # read the head of file, get the number of layers
    def set_tail_vars(self):
        """
        set the tail variables for the class
        :return:
        """
        self.tail_xL = 0
        self.tail_xH = 0
        self.tail_yL = 0
        self.tail_yH = 0
        self.tail_zL = 0
        self.tail_zH = 0

        self.tail_amp = 0
        self.tail_dec = 0
        self.tail_bitdepth = 16


    def _get_seg_file_info(self, file_name):
        '''Get the image size information from first line and layer information
        Args:
            file_name: the path of file
        Return:
            img_width, img_height, frame_num, layer_num: information read from
            the segmentaion files. From first line and the total lines
        Example:
            img_width,img_height,frame_num, layer_num = _get_seg_file_info(...
            "Zstruct_CFinalX300Y300Z1000.txt")
        '''
        ind = 0
        fid = open(file_name, 'r')
        first_line = fid.readline()

        try:
            found = re.search(r"(imgWidth=)(\d+)", first_line)
            img_width_str = found.group(0)[9:]
            found = re.search(r"(FrameNum=)(\d+)", first_line)
            frame_num_str = found.group(0)[9:]
            found = re.search(r"(imgHeight=)(\d+)", first_line)
            img_height_str = found.group(0)[10:]
        except AttributeError:
            print('image size information not found')
            fid.close()
            return -1, -1, -1, -1

        for line in fid:
            words = line.split()
            if words[0] == 'P':
                ind = ind + 1
        img_width = int(img_width_str)
        img_height = int(img_height_str)
        frame_num = int(frame_num_str)
        layer_num = int(ind/(img_width*frame_num))

        fid.close()
        return img_width, img_height, frame_num, layer_num


    def _read_calibrate(self, file_name, img_width, frame_num):
        '''Read the calibrate file, if needed
        Args:
            file_name: url
            img_width: image width read from segmentation file
            frame_num: frame number read from segmentation file
        Return:
            cali_layer: 2D array, calibartation matriax
        '''
        fid = open(file_name, 'r')
        cali_layer = np.zeros((img_width, frame_num))
        num = 0
        for line in fid:
            words = line.split()
            Y = int(num/img_width)
            X = num - Y*img_width
            num = num + 1
            cali_layer[X, Y] = float(words[0])
        fid.close()
        return cali_layer
    
   
    def _get_seg_lines(self, file_name, exist_cali=-1, scale=-1):
        '''Read the lines form the file, return the layers matrices
        Args:
            file_name: file_path
            exist_cali: set -1 for no calibration file, if exist calibration
                set it as the calibartion url to read the file
            scale: the scale of image, -1 for no scale
        Returns:
            new_layers: array shape segmentation files
        Example:
            layers,img_width,img_height,frame_num, line_num= _get_seg_lines(file_name)
        '''
        img_width, img_height, frame_num, layer_num = self._get_seg_file_info(file_name)
        fid = open(file_name, 'r') 

        # skip the first line
        first_line = fid.readline()

        print(first_line)
        ind=0   # the pixel index in one layer
        line_num = 0    # the number of layers
        layers = np.zeros((img_width, frame_num, layer_num))
        layer = np.zeros((img_width, frame_num))
        pix_per_layer = img_width*frame_num     # size of pixel per layer

        # read lines in the file 
        for line in fid:           
            words = line.split()
            if words[0] == 'P':     # the layer position strated with 'P'
                 
                # count the layers
                if ind>pix_per_layer:
                    # remove the empty lines
                    if np.mean(layer)==1:
                        break
                    layers[:,:,line_num] = layer 
                   # print(line_num)
                    line_num = line_num + 1
                    ind = 0
                layer[int(float(words[1])), int(float(words[2]))] = float(words[3])
                ind = ind+1
            else:
                break

        # if not empty, put the last line to the layers
        if np.mean(layer)!=1:
            layers[:,:,line_num] = layer
            line_num = line_num + 1

        # if exist calibarte file, add calibrate information  
        if exist_cali!=-1:
            try:
                cali_layer = self._read_calibrate(exist_cali, img_width, frame_num)
                print('read the calibrate file')
            except:
                print('can not read the calibrate file')    
                
            # apply the cali_file to the layers   
            for i in range(0, line_num):
                layers[:,:,i] = layers[:,:,i] - cali_layer
        
        fid.close()
        new_layers = layers[:,:,0:line_num]
        if scale!=-1:
            new_layers = new_layers/scale
                        
        return new_layers, img_width, img_height, frame_num, line_num
    #%%
    # auto segmentation

    def _get_auto_range(self):
        '''Get the depth range of images
        '''
        depth_profile = self.stru3d[0:-5, :, :].mean(axis=1).mean(axis=1)
        
        # get the maxinum index of depth
        max_depth = depth_profile.argmax()
        depth_start = 0
        depth_end = self.img_depth-5

        depth_profile = gaussian_filter(depth_profile, 10)
        depth_start = int(np.argmax(depth_profile >20) * 0.8)
        depth_end = int(len(depth_profile) - np.argmax((depth_profile[::-1] > 40))+30)
        # get the range
        # if (max_depth-600) >= 0:
        #     depth_start = max_depth - 600
        #
        # if (max_depth+800) < depth_end:
        #     depth_end = max_depth + 800

        print('The depth ranges are from: ', depth_start,' to ', depth_end, ', max depth=', max_depth)
        return depth_start, depth_end
        
    def auto_seg_stru(self, z_start=0, z_end=1000, auto_range=False, retina_lower_offset=50, retina_upper_offset=150, mode='eye', better_BM=False):
        '''Auto seg the ILM, RPE, BM of structure
        Args:
            z_start: to speed up the processing, only use the pixel within 
                this range to perform segmentation
            z_end: the end of range
            auto_range: use the auto range
            retina_lower_offset: offsets before maxinum intensity location
            retina_upper_offset: offsets before maxinum intensity location
        '''
        assert(self.exist_stru), 'structure file does not exist'
        
        if auto_range:
            z_start, z_end = self._get_auto_range()
        img = self.stru3d[z_start:z_end,...]
        print('better BM is', )
        if better_BM:
            new_layers = seg_video(img, self.flow3d[z_start:z_end,...], retina_lower_offset, retina_upper_offset, moving_avg=True, mode=mode, better_BM=True)
        else:
            new_layers = seg_video(img, retina_lower_offset, retina_upper_offset, moving_avg=True, mode=mode, better_BM=False)
        
        # compensate the cut range
        new_layers = new_layers + z_start
        self.layers = new_layers
        self.layer_num = new_layers.shape[2]
        self.exist_seg = True
    
    # read outside segmentation 
    def read_outside_seg(self, file_path):
        '''Load the outside segmentation file to the object
            (still need to be imporoved)
        Args:
            file_path: the filepath of .mat file
        '''
        names = file_path.split('.')
        file_format = names[-1]
        
        # save layers
        if file_format == 'npy':
            self.layers = np.load(file_path)
            self.layer_num = self.layers.shape[2]
            self.exist_seg = True
        elif file_format == 'mat':
            surface_data = loadmat(file_path)
            layers_variable = None
            the_only_key = None
            if 'layers' in surface_data:
                layers_variable = surface_data['layers']
            else:
                potential_data_keys = [k for k in surface_data.keys() if not k.startswith('__')]
                if len(potential_data_keys) == 1:
                    the_only_key = potential_data_keys[0]
                    layers_variable = surface_data[the_only_key]
                    print(
                        f"Info: Key 'layers' not found in {file_path}. Using the only available key: '{the_only_key}'.")
            if layers_variable is not None:
                self.layers = layers_variable
                if self.layers.ndim >= 3:
                    self.layer_num = self.layers.shape[2]
                    self.exist_seg = True
                else:
                    # Handle case where the found data doesn't have enough dimensions
                    raise ValueError(
                        f"Data found in {file_path} (key used: {'layers' if 'layers' in surface_data else the_only_key}) does not have at least 3 dimensions.")
            else:
                # Handle case where neither 'layers' nor a single alternative was found
                raise KeyError(
                    f"Could not find 'layers' key or a single unique data key in {file_path}. Available keys: {list(surface_data.keys())}")
        else:
            print('Cannot read the file, the file name should end with .npy or .mat')


#%%        
    def load_avi_file(self, file_name):
        '''Load the avi file and return the uint8 nparray
        Args:
            file_name: machine output avi files
        Return:
            img: 3D numpy array, with shape (depth, width, framenum)
        Example
            img = load_avi_file('Zstruct_C.avi')
            plt.imshow(img[:, :, 100])
        '''
        try:
            img = mimread(file_name, 'ffmpeg', memtest=False,)
        except Exception as e:
            logger.error(str(e))
            print ('can not open file: ', file_name)
            return Exception
        img = np.asarray(img, dtype=np.uint8)
    
        # for RGB video, only select first channel
        if img.ndim==4:
            img = img[..., 0]
            
        # reshape the dimension    
        img = np.rollaxis(img, 0, 3)
        return img
    
    def load_dicom_file(self, file_path):
        '''Load the dicom file and return the unit8 array
        Args:
            file_path: the file path of dcm file  
        '''
        try:
            ds = pydicom.dcmread(file_path) # shape(frame_number, row, column)
            img = np.rollaxis(ds.pixel_array, 0, 3)  # shape(row, column, frame_number)
            if img.max()>300:
                img = np.uint8(img/255)
            else:
                img = np.uint8(img)
        except Exception as e:
            print('Cannot load dicom file: ', e)
        return img

    def load_img_file(self, file_path):
        '''Load Zeiss IMG format
        Args:
            file_path: the file path of the IMG file
        '''
        try:
            # params = self._inferArgs(file_path)
            a = self._readfile(file_path)
        except Exception as e:
            print('Cannot load img file:', e)
        return a

    # Get cube size from filename; this is the "official" way to do it
    def _inferArgs(self, s, size):
        print('size=',size)
        s = s.split("/")[-1]
        p = re.search(r'\((\d+)mmx(\d+)mm\)', s, re.IGNORECASE)
        try:
            # for SD OCT
            if size == 125440000:
                return (350, 1024, 350)
            if size == 40960000:
                return (200, 1024, 200)
            if size == 67108864:
                return (128, 1024, 512)

            # for SS OCT
            w = p.group(1)
            h = p.group(2)
            if w == h and w == '3':
                return (300, 1536, 300)
            if w == h and (w == '6' or w == '9' or w == '12'):
                return (500, 1536, 500)
            if w == '15' and h == '9':
                return (834, 1536, 500)


        except Exception as e:
            print(e, 'SSOCT file name format exception: *(Nmm x Nmm) *.img, N=3,6,9,12')

        p = re.search(r'P(\d+)x(\d+)x(\d+)', s, re.IGNORECASE) # P A_lines x B_scans x Z-depth
        try:
            x = int(p.group(1))
            y = int(p.group(2))
            z = int(p.group(3))

            return (y, z, x)
        except Exception as e:
            print(e, 'SDOCT file name format exception: * P A_lines x B_scans x Z-depth *.img')



        return (128, 1024, 512) # sd oct has 3x3 6x6
        # print(f"unknown format from filename, please specify manually; read w = {w} h = {h}")
        # sys.exit(1)

    # Axis order is (height, width, number of B-scans)
    def _readfile(self, fn, params=None):
        a = np.fromfile(fn, np.uint8)
        if params is None:
            params = self._inferArgs(fn, size=len(a))
        a = np.reshape(a, params)
        a = np.swapaxes(a, 0, 1)
        a = np.swapaxes(a, 1, 2)
        a = np.flip(a, axis=0)
        a = np.flip(a, axis=1)
        return a

    def load_mhd_file(self, file_path):
        """
        load mhd file with simple itk
        :param file_path:
        :return:
        """
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(file_path)

        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        img = sitk.GetArrayFromImage(itkimage)
        img_new = np.rollaxis(img, 0, 3)
        return img_new



    def load_3d_data(self, file_path):
        '''Load the 3d data by file type
        '''
        names = file_path.split('.')
        file_format = names[-1]
        
        if file_format == 'avi':
            img = self.load_avi_file(file_path)
        elif file_format == 'dcm':
            img = self.load_dicom_file(file_path)
        elif file_format == 'img':
            img = self.load_img_file(file_path)
        elif file_format == 'mhd':
            img = self.load_mhd_file(file_path)
        else:
            print('File name does not correct, please input dcm or avi file')
        return img
        
    def read_stru_data(self, file_path):
        '''Read the flow data from file path to the self.stru3d
        Args:
            file_path: the file path of avi file
        '''
        self.stru3d = self.load_3d_data(file_path)
        print(type(self.stru3d))
        (self.img_depth, self.img_width, self.img_framenum) = self.stru3d.shape
        
        # set the flag as true 
        self.exist_stru = True
        self.url_stru = file_path
        print('structural data load. ', ' self.exist_stru = ', self.exist_stru)
        
    def read_flow_data(self, file_path):
        '''Read the flow data from file path to self.flow3d
        Args:
            file_path: the file path of avi file
        '''
        self.flow3d = self.load_3d_data(file_path)
        (self.img_depth, self.img_width, self.img_framenum) = self.flow3d.shape
        
        # set the flag as true 
        self.exist_flow = True
        self.url_flow = file_path
        print('flow data load. ', ' self.exist_flow = ', self.exist_flow)
        
    def read_seg_layers(self, seg_path, exist_cali=-1, scale=-1):
        '''Read the segmentation file from seg_path to self.layers
        Args:
            seg_path: file path of the segmentation txt file
            exist_cali: -1 for not use outside calibrate file
            scale: -1 for not use scale factor
        '''
        names = seg_path.split('.')
        file_format = names[-1]
        
        # save layers
        if file_format == 'txt':
            try:
                new_layers, img_width,img_height,frame_num, line_num = self._get_seg_lines(seg_path, exist_cali, scale)
            except Exception: 
                print('Cannot read segmentation file')
                return Exception
            
            # smooth
            for i in range(0, new_layers.shape[2]):
                new_layers[:,:,i] = median_filter(gaussian_filter(new_layers[:,:,i],sigma=(1,5)), (1,3)) 

            self.layers = new_layers
            self.layer_num = line_num
            self.exist_seg = True
            self.url_seg = seg_path
        else:
            self.read_outside_seg(seg_path)
        
        print('segmentation file load. ', ' self.exist_seg = ', self.exist_seg)
#%% saveing       
        
    def save_video(self, img, file_name='out.avi'):
        '''Save the video by te filename, avi or dicom
        Args:
            img: 3d numpy array
            file_name: url to save, endwith 'avi' or 'dicom'
        '''
        names = file_name.split('.')
        file_format = names[-1]
        
        if file_format == 'avi':
            img = self._save_avi_file(img, file_name)
        elif file_format == 'dcm':
            img = self._save_dicom_file_temp(img, file_name)
        else:
            print('File name does not correct, please input dcm or avi file')

        
    def _save_dicom_file(self, img, filename):
        '''save 3d dicom file to disk
        Args:
            img: 3d data with shape [row column frame_number]
            filename: the url to save, end with dcm
        '''
        file_meta = Dataset()
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    
        # Create the FileDataset instance (initially no data elements, but file_meta
        # supplied)
        ds = FileDataset(filename, {},
                         file_meta=file_meta, preamble=b"\0" * 128)
        
        # Add the data elements -- not trying to set all required here. Check DICOM
        # standard
        
        # Set the transfer syntax
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        
        # set pixel array    
        pixel_array = np.copy(img)
        # before [row column frame_number], after [frame_number, row, column ]
        pixel_array = np.rollaxis(pixel_array, 2, 0)    
        pixel_array.shape
        
        ## These are the necessary imaging components of the FileDataset object.
        ds.BitsAllocated = 16
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        
        ds.Rows = pixel_array.shape[1]
        ds.Columns = pixel_array.shape[2]
        ds.NumberOfFrames = pixel_array.shape[0]
        
        # uint16
        if pixel_array.dtype != np.uint16:
                pixel_array = pixel_array.astype(np.uint16)
        
        ds.PixelData = pixel_array.tostring()
        ds.save_as(filename)
        print('The file ', filename, ' has been saved.')

    # -------------------------------------------------------------------------
    # OPTIMIZED OAC CALCULATION (Optical Attenuation Coefficient)
    # -------------------------------------------------------------------------
    def OAC_calculation(self, img=None, chunk_size=20):
        """
        Calculate OAC (Optical Attenuation Coefficient) volume using Structural Data.
        Optimized for memory usage by processing in chunks (Frames).
        Formula: JLin = OCTL[:-1] / (CumulativeTrapz(Flip(OCTL)) * 2 * delta_z)
        """
        import scipy.integrate as integrate

        # Default to structural data if nothing passed
        if img is None:
            if self.exist_stru:
                img = self.stru3d
            else:
                print("Error: OAC requires structural data.")
                return None

        # Dimensions
        depth, width, frames = img.shape

        # Result container (Depth-1 because integration reduces dimension by 1)
        JLin_vol = np.zeros((depth - 1, width, frames), dtype=np.float32)

        print(f"Calculating OAC Volume in chunks of {chunk_size} frames...")

        # Process in chunks to save RAM
        for i in range(0, frames, chunk_size):
            # Define chunk boundaries
            f_start = i
            f_end = min(i + chunk_size, frames)

            # 1. Extract Chunk & Convert to Float32
            # Shape: [Depth, Width, Chunk_Frames]
            chunk = img[:, :, f_start:f_end].astype(np.float32)

            # 2. Calculate OCTL (Linear OCT)
            # OCTL = ((10 ** (img / 80)) - 1) / 7.1
            # We use in-place operations to keep memory low
            chunk /= 80.0
            np.power(10.0, chunk, out=chunk)
            chunk -= 1.0
            chunk /= 7.1
            # 'chunk' is now 'OCTL'

            # 3. Calculate M (Cumulative Trapezoid Integration)
            # Flip dimensions for integration from bottom-up
            # Note: cumtrapz reduces axis 0 size by 1
            M = integrate.cumulative_trapezoid(chunk[::-1, ...], axis=0)
            M = M[::-1, ...]  # Flip back

            # 4. Calculate JLin
            # Formula: OCTL / (M * 2 * (3 / Depth))
            # Note: OCTL must be sliced [:-1] to match M's new shape

            pixel_size_z = 3.0 / depth
            denominator = M * 2.0 * pixel_size_z

            # Avoid divide by zero
            denominator[denominator == 0] = 1e-9

            JLin_chunk = chunk[:-1, ...] / denominator

            # Store result
            JLin_vol[:, :, f_start:f_end] = JLin_chunk

            # Cleanup
            del chunk
            del M
            del JLin_chunk

        print("OAC Calculation Complete.")
        return JLin_vol

    def get_oac_projection(self, start_layer, end_layer):
        """
        Helper to get the 2D OAC Map (Mean Projection) between layers.
        """
        # 1. Calculate the full OAC Volume
        oac_vol = self.OAC_calculation()  # Uses self.stru3d by default

        if oac_vol is None: return None

        # 2. Extract Slab (Adjusting for the fact OAC vol is 1 pixel shorter)
        # We reuse the existing slab extraction logic logic, but applied to OAC vol
        # Note: We must be careful about indices since oac_vol is depth-1

        # Simple slab extraction for OAC (Custom loop to handle the depth difference)
        max_d, w, f = oac_vol.shape
        oac_map = np.zeros((w, f), dtype=np.float32)

        layers = self.layers

        for i in range(w):
            for j in range(f):
                # Get layer boundaries
                z1 = int(layers[i, j, start_layer])
                z2 = int(layers[i, j, end_layer])

                # Clip to volume bounds
                if z1 < 0: z1 = 0
                if z2 > max_d: z2 = max_d
                if z1 >= z2: z1 = z2 - 1

                if z2 > z1:
                    col = oac_vol[z1:z2, i, j]
                    oac_map[i, j] = np.mean(col)

        # Normalize map for visualization
        if np.max(oac_map) > 0:
            oac_map = oac_map / np.max(oac_map)

        return oac_map
    def _save_dicom_file_temp(self, data_save, file_name_out='out.dcm'):
        """
        data_save: [width, depth, frame]
        Uses a template 'dicom_format.dcm' to save data.
        FIXED: Looks for template in the same folder as this script.
        """
        # Look for the template in the same directory as THIS python file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(script_dir, 'dicom_format.dcm')

        if not os.path.exists(filename):
            print(f"ERROR: Template {filename} not found!")
            return

        ds = pydicom.dcmread(filename)

        # get the pixel information into a numpy array
        # data = ds.pixel_array # Not strictly needed if we overwrite

        # Force convert to u16 bit and adjust axes if necessary
        # Assuming input is [Depth, Width, Frames] -> Dicom usually [Frames, Depth, Width]
        # Adjust logic below to match your specific orientation needs
        data_save = np.uint16(np.rollaxis(data_save, 2, 0))

        # copy the data back to the original data set
        ds.PixelData = data_save.tobytes()  # 'F' order removed for compatibility, standard C order usually safer

        # update the information regarding the shape of the data array
        ds.NumberOfFrames, ds.Rows, ds.Columns = data_save.shape

        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.PatientID = 'None'

        pydicom.filewriter.dcmwrite(file_name_out, ds)
        print('The file ', file_name_out, ' has been saved.')



    def _save_avi_file(self, img, file_name_out='out.avi'):
        ''' Save the video to local disk
        Args:
            img: 3D image
            file_name_out: saved file(current format ffmpeg rawvideo avi.)
        '''
        fps=30
        import imageio
        writer = imageio.get_writer(file_name_out, fps=fps,
                        quality=10, codec='rawvideo')

        # normalization needed: float64 to uint8 issue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(0, img.shape[2]):
                writer.append_data(img[:, :, i])
            writer.close()
            print(file_name_out, ' video saved')

    def call_flatten_video(self, img, file_name='flatten.dcm', ref_layer_num=2, new_loc=400, saved=True):
        '''helper function to produce the flatten image, return a 3d array 
        Args:
            img: 3d image
            file_name: output avi filename
            ref_layer_num: image number for reference
        Return:
            img_faltten: the flatten image
        '''
        # use the RPE as reference to flatten the image
        rpe_layer = np.round(self.layers[..., ref_layer_num])        
        img_flatten = np.zeros((self.img_depth, self.img_width, self.img_framenum), dtype=img.dtype) #float
        # loop         
        for i in range(0, self.img_width):
            for j in range(0, self.img_framenum):
                offset_z = np.int32(new_loc - rpe_layer[i, j])    #400 can be changed
                try:
                    if offset_z>=0:
                        vec_length = np.int32(self.img_depth - offset_z)
                        img_flatten[np.abs(offset_z):, i, j] = img[0:vec_length, i, j]
                        
                    else:
                        vec_length = np.int32(self.img_depth - np.abs(offset_z))
                        img_flatten[0:vec_length, i, j] = img[np.abs(offset_z):, i, j]
                except Exception:
                    print('save fail at ', offset_z, vec_length ,i, j)
        # format issue
        if saved:
            self.save_video(img_flatten, file_name)

        return img_flatten

    def reverse_flatten_video(self, img, file_name='flatten_original.dcm', ref_layer_num=2, new_loc=400, saved=True):
        '''helper function to produce the flatten image, return a 3d array
        Args:
            img: 3d image
            file_name: output avi filename
            ref_layer_num: image number for reference
        Return:
            img_faltten: the flatten image
        '''
        # use the RPE as reference to flatten the image
        rpe_layer = np.round(self.layers[..., ref_layer_num])
        max_depth = np.int32(np.max(rpe_layer) + img.shape[0]+100)
        img_flatten = np.zeros((max_depth, self.img_width, self.img_framenum), dtype=img.dtype)  # float
        # loop
        for i in range(0, self.img_width):
            for j in range(0, self.img_framenum):
                offset_z = np.int32(rpe_layer[i, j] - new_loc)  # 400 can be changed
                try:
                    if offset_z >= 0:
                        vec_length = np.int32(img.shape[0])

                        img_flatten[np.abs(offset_z):np.abs(offset_z)+vec_length, i, j] = img[0:vec_length, i, j]

                    else:
                        vec_length = np.int32(self.img_depth - np.abs(offset_z))
                        img_flatten[0:vec_length, i, j] = img[np.abs(offset_z):, i, j]
                except Exception:
                    print('save fail at ', offset_z, vec_length, i, j)
        # format issue
        if saved:
            self.save_video(img_flatten, file_name)

        return img_flatten

    def save_reverse_flatten_video(self, file_name='flatten_original.dcm', video_type='stru', ref_layer_num=0, new_loc=400, saved=True):
        '''Save the flatten layers based on the segmentation line
        Args:
            file_name: the flatten file to save
            video_type: stru or flow
            ref_layer_num: the reference layer for flatten image
            new_loc: the new depth of the reference layer
        Return:
            img_flatten: numpy array style flatten image
        '''
        # assert (self.exist_flow==True), "flow not exist"
        assert (self.exist_seg), "seg file not exist"
        if video_type == 'stru':
            assert (self.exist_stru), "stru3d not exist"
            img_flatten = self.reverse_flatten_video(self.stru3d, file_name, ref_layer_num, new_loc, saved)
        elif video_type == 'flow':
            assert (self.exist_flow), "flow3d not exist"
            img_flatten = self.reverse_flatten_video(self.flow3d, file_name, ref_layer_num, new_loc, saved)

        return img_flatten

    def save_flatten_video(self, file_name='flatten.dcm', video_type='stru', ref_layer_num=0, new_loc=400, saved=True):
        '''Save the flatten layers based on the segmentation line
        Args:
            file_name: the flatten file to save
            video_type: stru or flow
            ref_layer_num: the reference layer for flatten image
            new_loc: the new depth of the reference layer
        Return:
            img_flatten: numpy array style flatten image
        '''
        #assert (self.exist_flow==True), "flow not exist"
        assert (self.exist_seg), "seg file not exist"
        if video_type == 'stru':
            assert (self.exist_stru), "stru3d not exist"
            img_flatten = self.call_flatten_video(self.stru3d, file_name, ref_layer_num, new_loc, saved)
        elif video_type == 'flow':
            assert (self.exist_flow), "flow3d not exist"
            img_flatten = self.call_flatten_video(self.flow3d, file_name, ref_layer_num, new_loc, saved)
        
        return img_flatten
    
    def save_layers(self, url='layers.npy'):
        '''Save the segmentation file to the disk
        Args:
            file_name: the url of file, should end with .mat or .npy
        '''
        assert(self.exist_seg), "seg file not exist"
        
        # get the formats of the file
        names = url.split('.')
        file_format = names[-1]
        layers = self.layers
        
        # save layers
        if file_format == 'npy':
            np.save(url, layers)
        elif file_format == 'mat':
            layer_dict = {}
            layer_dict['layers'] = layers
            savemat(url, layer_dict)
        else:
            print('Cannot save the file, the file name should end with .npy or .mat')
            
#%% image subpixel registration
    def sub_pixel_regi(self, img, flow=None):
        """
        do the subpixel registration
        :param img: shape [ x, y, frame]
        :return:
        """
        # get the first image as the reference image
        new_img = np.zeros_like(img)
        ref_img = img[:, :, 0]
        new_img[:, :, 0] = ref_img
        new_img_flow = None

        if flow is not None:
            new_img_flow = np.zeros_like(img)
            new_img_flow[:, :, 0] = flow[:, :, 0]

        errors = []

        print('Subpixel registration:')
        # loop registration to the first frame
        for i in tqdm(range(1, img.shape[2])):
            offset_image = img[:, :, i]
            shift, error, diffphase = phase_cross_correlation(ref_img, offset_image,
                                                              upsample_factor=10, overlap_ratio=0.5)
            errors.append(error)
            #print('slice ', i, ', shift ', shift)

            # reconstruct
            #re_image = fourier_shift(np.fft.fftn(offset_image), (shift[0], shift[1]))
            re_image = fourier_shift(np.fft.fftn(offset_image), (shift[0], 0))  # do not shift laterally

            re_image = np.fft.ifftn(re_image)

            # cut the range
            re_image = self.cut_range(re_image.real, img[:, :, 0])
            fix_img = re_image.real.astype(img.dtype)  # change the dtype

            new_img[:, :, i] = fix_img

            if flow is not None:
                re_image_flow = fourier_shift(np.fft.fftn(flow[:, :, i]), (shift[0], shift[1]))
                re_image_flow = np.fft.ifftn(re_image_flow)
                re_image_flow = self.cut_range(re_image_flow.real, img[:, :, 0])
                new_img_flow[:, :, i] = re_image_flow.real.astype(img.dtype)
                return new_img, new_img_flow, errors
            # update the reference image

            ref_img = np.copy(fix_img)

        return new_img, errors

    def cut_range(self, img, ref):
        """
        cut the range of img by the data type of ref
        """
        # img = img.astype(float)
        max_val = np.iinfo(ref.dtype).max
        min_val = np.iinfo(ref.dtype).min
        img[img > max_val] = max_val
        img[img < min_val] = min_val
        return img

    #%%
    def plot_slice(self, slice_num):
        '''Plot the slice of stru and flow image, if exist
        Args:
            slice_num: the number of slice
        '''
        assert ((slice_num >= 0) and (slice_num < self.img_framenum)), "slice number incorrect"
        if self.exist_flow:
            plt.figure()
            plt.imshow(self.flow3d[:, :, slice_num], cmap='gray')
            plt.title('flow' + str(slice_num))
            plt.close()
        if self.exist_stru:
            plt.figure()
            plt.imshow(self.stru3d[:, :, slice_num], cmap='gray')
            plt.title('stru' + str(slice_num))
            plt.close()
    
    def __plot_slice_layers(self, img, slice_num, orientation):
        '''Plot the layers and images
        Args:
            img: 3d array image
            slice_num: slice number
            orientation: 1 for fast scan preview, 2 for slow scan preview
        '''
        range_z = 900
        fig1 = plt.figure(figsize = (12,6))
        ax1 = fig1.add_subplot(111)
        
        # plot the YZ frame
        if orientation==2:
            assert ((slice_num >= 0) and (slice_num < self.img_width)), "slice number incorrect"
            ax1.imshow(img[0: range_z, slice_num, :], cmap='gray')
            x = np.arange(0, self.layers.shape[1])
            cmap = matplotlib.cm.get_cmap('prism')
            for i in range(0, self.layer_num):    
                ax1.plot(x, self.layers[slice_num, :, i], linewidth=2, color=cmap(i*10), label=str(i))  #plt.cm.gist_ncar(np.random.random())

        # plot the XZ frame        
        elif orientation==1:
            assert ((slice_num >= 0) and (slice_num < self.img_framenum)), "slice number incorrect"
            ax1.imshow(img[0: range_z, :, slice_num], cmap='gray')
            x = np.arange(0, self.layers.shape[0])
            cmap = matplotlib.cm.get_cmap('prism')
            for i in range(0, self.layer_num):    
                ax1.plot(x, self.layers[:, slice_num, i], linewidth=2, color=cmap(i*10), label=str(i))  #plt.cm.gist_ncar(np.random.random())
        
        ax1.legend()
        plt.show()
        plt.close()
        
    def plot_stru_layers(self, slice_num, orientation=1):
        ''' Plot the OCT images with segmentation lines
        Args:
            slice_num: slice number
            orientation: 1 for fast scan preview, 2 for slow scan preview
        '''
        self.__plot_slice_layers(self.stru3d, slice_num, orientation)
        
    def plot_flow_layers(self, slice_num, orientation=1):
        ''' Plot the OCT-A images with segmentation lines
        Args:
            slice_num: slice number
            orientation: 1 for fast scan preview, 2 for slow scan preview
        '''
        self.__plot_slice_layers(self.flow3d, slice_num, orientation)

    def save_seg_video(self, step=10, file_name='lines.avi'):
        '''Save the video with lines (Fixed for Matplotlib 3.8+)'''
        assert (self.exist_stru), 'stru image does not exist'
        assert (self.exist_seg), 'seg file does not exist'

        # Use imageio writer
        import imageio
        writer = imageio.get_writer(file_name, fps=30, macro_block_size=None,
                                    quality=10, codec='rawvideo')

        # Calculate aspect ratio
        img_h = 6
        img_w = (img_h * self.stru3d.shape[1] / self.stru3d.shape[0])

        print(f"Saving Seg Video to {file_name}...")

        # Loop plot
        for i in range(0, self.img_framenum, step):
            fig = plt.figure(figsize=(img_w, img_h), dpi=200)
            plt.title('FrameNum=' + str(i))
            plt.imshow(self.stru3d[:, :, i], cmap='gray', aspect='equal')
            plt.axis('off')

            # Plot all layers
            for j in range(0, self.layers.shape[2]):
                plt.plot(self.layers[:, i, j], label=str(j))
            plt.legend()

            # --- FIX: Use buffer_rgba ---
            fig.canvas.draw()
            # Get RGBA buffer
            data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()

            # Reshape and drop Alpha channel to get RGB (h, w, 3)
            im_array = data.reshape((h, w, 4))[:, :, :3]

            writer.append_data(im_array)
            plt.close(fig)  # Important: Close fig to save memory

        writer.close()
        print("Seg Video Saved.")
    
    #%%
    def thickness_map(self, start, end, smooth=True):
        ''' Get the thickness map of layers, if start=end, return depth location
        Args:
            strat: the start layer number
            end: the end of layer nunber
            smooth: do smooth on slow scan 
        Retrun: The thickness map
        '''
        assert(end >= start), 'please correct the layer number'
        if end>start:
            thick_map = self.layers[:, :, end] - self.layers[:, :, start]
        else:
            thick_map = self.layers[:, :, end]
        if smooth:
            thick_map = median_filter(thick_map, (1, 5))
        plt.figure()
        plt.imshow(thick_map)
        plt.colorbar()
        plt.close()
        return thick_map

    def get_mean_thickness(self, start, end):
        """
        Get the mean thickness between start layer and end layer
        :param start:
        :param end:
        :return:
        """
        temp = self.layers[:, :, end] - self.layers[:, :, start]
        thick = np.mean(temp)
        return thick

    def get_fovea_location(self, rg_low=0.4, rg_high=0.6):
        """
        Get the fovea location based on segmentation data, check the center part of the layers
        :param rg_low:
        :param rg_high:
        :return: X_max[column], Y_max[raw]
        """
        layer_diff = gaussian_filter(self.layers[:, :, 2], (3, 3)) - gaussian_filter(self.layers[:, :, 0], (3, 3))

        layer_diff_smooth = gaussian_filter(layer_diff, (25, 25))

        surface = layer_diff_smooth - layer_diff

        [Y_rg, X_rg] = surface.shape
        mask = np.zeros_like(surface)
        mask[int(Y_rg * rg_low):int(Y_rg * rg_high), int(X_rg * rg_low):int(X_rg * rg_high)] = 1

        if surface.min() < 0:
            surface = surface - surface.min()
        surface = surface * mask

        X_max = np.argmax(np.mean(surface[int(Y_rg * rg_low):int(Y_rg * 0.6), :], axis=0))
        Y_max = np.argmax(np.mean(surface[:, int(X_rg * rg_low):int(X_rg * 0.6)], axis=1))

        # plt.imshow(layer_diff)
        # plt.plot(X_max, Y_max,'*r')
        # plt.show()

        return X_max, Y_max
            
    #%%
    def _get_one_layer_image(self, img, start, end, start_offset=0, end_offset=0):
        '''Get the image within one layer
        Args:
            img: the 3D image
            start: number of strating layer
            end: number of end layer
            strat_offset: offset of the starting layer
            end_offset: offset of the end layer
        Return: the new image only contains the information inside the layers
        '''
        layer_int = self.layers.astype(np.int16)
        '''Below here is the old indexing method.
        #proj image struc, flow
        select_img_stru = np.zeros(img.shape)
        
        for i in range(0, img.shape[1]):
            for j in range(0, img.shape[2]):
               # label[layer_int[i, j, 2]:layer_int[i, j, 4],i,j] = 1
                select_img_stru[layer_int[i, j, start]+start_offset: \
                                layer_int[i, j, end]+end_offset, i, j] = \
                                img[layer_int[i, j, start]+start_offset: \
                                    layer_int[i, j, end]+end_offset, i, j]
        '''
        layer_start=layer_int[...,start]+start_offset
        layer_end =layer_int[...,end] +end_offset
        layer_start = self._check_boundry(img.shape[0], layer_start)
        layer_end = self._check_boundry(img.shape[0], layer_end)

        layer_thickness=layer_end - layer_start
        max_thickness=np.amax(layer_thickness)
        nx,ny=np.shape(img[1,...])
        O,P,Q=np.meshgrid(np.arange(max_thickness),np.arange(nx),np.arange(ny),sparse=False, indexing='ij')
        
        O=O+layer_start[np.newaxis,...]
        
        ind_outof_bottom = O>np.size(img,0)-1;
        ind_outof_top    = O<0;
        
        Omask=np.zeros(O.shape)
        Omask[O-layer_end>0]=1 #mask pixels outside layer_end
        Omask[ind_outof_bottom]=1 #mask pixels outside bottom of image
        Omask[ind_outof_top]=1 #mask pixels outside top of image
        
        O[ind_outof_bottom]=0 
        O[ind_outof_top]=0
        select_img=img[O,P,Q]
        
        select_img_msk=np.ma.array(select_img, mask=Omask)
        return select_img_msk

    def _check_boundry(self, img_depth, layer):
        '''
        Args:
            img_depth: the
            layer:
        :return:
        '''
        layer[layer > img_depth-1] = img_depth-1
        layer[layer < 0] = 0
        return layer

    def _sum_proj_layers(self, image):
        '''Make the sum projection of the layers of image, it sums the intensity
        of pixels and weighted by the thickness of 
        Args:
            image: image with shape (depth, width, frames_num)
        Return: projection image
        '''
        # 1. Calculate the raw sum (This creates huge numbers, e.g., 5000+)
        projs = np.nansum(image, axis=0)

        # 2. Normalize to 0.0 - 1.0 range
        if projs.max() > 0:
            proj_img = projs / projs.max()
        else:
            proj_img = projs

        # 3. RETURN THE NORMALIZED IMAGE (Fix: Return proj_img, not projs)
        return proj_img
    
    def _sum_mean_proj_layers(self, image):
        '''Make the mean projection of the layers of image
        Args:
            image: image with shape (depth, width, frames_num)
        Return: projection image
        '''      
        # make the sum projection image of layers
        projs = np.zeros((image.shape[1], image.shape[2]))
        projs = image.mean(axis=0)
        #file_name_out = file_name  + '_sum.png' 
        proj_img = projs/projs.max()
    #    imwrite(file_name_out, proj_img)            
        #print('Sum projection complete')
        return projs
        
    
    def _max_proj_layers(self, image):
        '''Make the maxinum projection of the layers of image
        Args:
            image: image with shape (depth, width, frames_num)
        Return: projection image
        '''
        # make the maxinum projection image of layers
        projs = np.zeros((image.shape[1], image.shape[2])) 
        projs = np.nanmax(image, axis=0)
        #file_name_out = file_name  + '_sum.png' 
        proj_img = projs/projs.max()
    #    imwrite(file_name_out, proj_img)
        #print('Sum projection complete')
        return projs

    def _max_depth_proj_layers(self, image):
        '''Make the maxinum projection of the layers of image and its depth location
        Args:
            image: image with shape (depth, width, frames_num)
        Return: projection image, depth index
        '''
        # make the maxinum projection image of layers
        projs = np.zeros((image.shape[1], image.shape[2])) 
        projs = image.max(axis=0)
        inds = image.argmax(axis=0)
        #file_name_out = file_name  + '_sum.png' 
        proj_img = projs/projs.max()
    #    imwrite(file_name_out, proj_img)
        #print('Sum projection complete')
        return projs, inds
    
    def _max_mean_proj_layers(self, image, select_num=5):
        '''Make the mean of 5 maxinum projection of the layers of image
        Args:
            image: image with shape (depth, width, frames_num)
            select_num: calculate the mean first N maxinum, default=5
        Return: projection images
        '''
        # make the maxinum projection image of layers
        projs = np.zeros((image.shape[1], image.shape[2]))

        # number of average should smaller than the thickness of slab
        if select_num>image.shape[0]:
            select_num = image.shape[0]
    
        # sort with descending order
       # img_sort = np.sort(image, axis=0)[::-1]
        image.sort(endwith=False, axis=0)
        #np.save('sort.npy', img_sort)

        projs = np.mean(image[-select_num:, :, :], axis=0)
        #print('Really do the maxmean')
        
        #file_name_out = file_name  + '_sum.png' 
        proj_img = projs/projs.max()
    #    imwrite(file_name_out, proj_img)
        #print('Sum projection complete')
        return projs

    def _min_proj_layers(self, image):
        '''Make the maxinum projection of the layers of image
        Args:
            image: image with shape (depth, width, frames_num)
        Return: projection image
        '''
        # make the maxinum projection image of layers
        projs = np.zeros((image.shape[1], image.shape[2]))
        projs = image.min(axis=0)
        # file_name_out = file_name  + '_sum.png'
        proj_img = projs / projs.max()
        #    imwrite(file_name_out, proj_img)
        # print('Sum projection complete')
        return projs

    def _min_mean_proj_layers(self, image, select_num=5):
        '''Make the mean of 5 maxinum projection of the layers of image
        Args:
            image: image with shape (depth, width, frames_num)
            select_num: calculate the mean first N maxinum, default=5
        Return: projection images
        '''
        # make the maxinum projection image of layers
        projs = np.zeros((image.shape[1], image.shape[2]))

        # sort with descending order
        img_sort = np.sort(image, axis=0)
        projs = np.mean(img_sort[0:select_num, :, :], axis=0)

        # file_name_out = file_name  + '_sum.png'
        proj_img = projs / projs.max()
        #    imwrite(file_name_out, proj_img)
        # print('Sum projection complete')
        return projs
    
    def img_constrat_change(self, img, low_rg=2, high_rg=99.5):
        """ Change the image contrast with the percentile 
        Return:
            img_cst: image float [0-1]
        """
        int_low = np.percentile(img, low_rg)
        int_high = np.percentile(img, high_rg)
        
        
        img_limit = 1
        img_cst = (img - int_low)*img_limit/(int_high-int_low) #[0, 1]
        img_cst[img_cst<0] = 0
        img_cst[img_cst>img_limit] = img_limit
        
        return img_cst
        
    
    # change the contrast and intensity of image
    def img_enhance_2d(self, img, low_rg=2, high_rg=99.7):
        """ Change the global contrast and enhance the local contrast by CHILE
        img: 2D image
        low_rg: the low perctile of the image [0-100]
        high_rg: the high perctile of the image [0-100]
        Return:
            img_new: The enhanced 2d image with uint8 type
        """
        # noise reduction
        img = median_filter(img, size=(2,2))
        img_cst = self.img_constrat_change(img,low_rg, high_rg)
        # img_new = equalize_adapthist(img_cst, clip_limit=0.008) 
        img_new = img_cst
        img_new = img_new*255
        return img_new.astype('uint8')
        
    
    def volume_filter(self, img, low_rg=2, high_rg=99.9):
        """3d volume filter for noise reduction
        img: 3D image
        low_rg: the low perctile of the image [0-100]
        high_rg: the high perctile of the image [0-100]
        Return:
            img_new: The enhanced 3d image with same type
        """
        
        img = median_filter(img, size=(4, 2, 2))
        img_cst = self.img_constrat_change(img, low_rg, high_rg)
        return img_cst


    def plot_proj(self, start, end, datatype='stru', projmethod='max', 
                  start_offset=0, end_offset=0, display_slice=10, display=True,
                  enhance=False, rotate=False):
        '''make the enface projection of the layer
        Args:
            start: the number of strat layer, 0,1,2..
            end: the number of end layer, start+0,1,2....
            datatype: 'stru' or 'flow'
            projmethod: 'max', 'sum', 'mean', 'maxmean'
            start_offset: the strating pixel before the first layer, 
                e.g. start=-5, start=7
            end_offset: the ending pixel beyond the last layer, 
                e.g. end=-4, end=2
            display_slice: display the slab on the slice, default=10
        Return:
            2d-array of en face projection, not normalized
        '''
        assert (end>=start), 'please input end>=start'
        
        # select data type
        if datatype == 'stru':
            assert (self.exist_stru), 'stru image does not exist'
            img = self.stru3d    
        elif datatype=='flow':
            assert (self.exist_flow), 'flow image does not exist'
            img = self.flow3d
        img_volume = self._get_one_layer_image(img, start, end, start_offset, end_offset)

        # select the projection method
        projection_methods = {
            'max': self._max_proj_layers,
            'sum': self._sum_proj_layers,
            'mean': self._sum_mean_proj_layers,
            'maxmean': lambda x: self._max_mean_proj_layers(x, select_num=5),
            'min': self._min_proj_layers,
            'minmean': lambda x: self._min_mean_proj_layers(x, select_num=5)
        }

        if projmethod not in projection_methods:
            raise ValueError(
                f"Invalid projmethod. Must be one of {', '.join(projection_methods.keys())}.")

        proj_imgs = projection_methods[projmethod](img_volume)

        # re-arrange the dynamic range
        if enhance:
            proj_imgs = self.img_enhance_2d(proj_imgs, low_rg=2, high_rg=99.5)

        # add the rotation and flip for eye data
        if rotate:
            proj_imgs_rot = np.rot90(proj_imgs, 3)
            proj_imgs_rot = np.flip(proj_imgs_rot, 1)
        else:
            proj_imgs_rot = proj_imgs

        # if display the plot
        if display:
            plt.figure()
            plt.imshow(proj_imgs, cmap='gray')
            plt.colorbar()
            plt.show()
            plt.close()

            #plot a b_scan
            line_up = self.layers[:, display_slice, start] + start_offset
            line_low = self.layers[:, display_slice, end] + end_offset
            plt.figure(figsize=(6, 8))
            plt.imshow(img[:, :, display_slice], cmap='gray')
            plt.plot(line_up, '--', linewidth=2)
            plt.plot(line_low, '--', linewidth=2)
            plt.show()
            plt.close()
        
        return proj_imgs_rot
        
                
    def save_single_layer_volume(self, img, start_layer, end_layer, start_offset, end_offset):
        """ return the one layer 3d data
        :return:
        """
        #start_ind = math.ceil(self.layers[:, :, start_layer].min()) - 1
        #end_ind = math.ceil(self.layers[:, :, start_layer+1].max())

        img_volume = self._get_one_layer_image(img, start_layer, end_layer, start_offset, end_offset)
        return img_volume

