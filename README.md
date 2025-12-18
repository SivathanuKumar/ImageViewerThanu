# OCT image processing library

Oct3Dimage  is a library that provides an easy interface to read and write a wide range of oct image data and the segmentation files. It also contains simple image processing algorithms and easy to make more extensions. 

## Folder Structure:
### 1.Analysis:
 - Contains main processing codes, called from the main image viewer to work on the inputs to the software
### 2.Networks:
 - List of networks used to segment files. Modifications to network structure and other AI enabled processes happen here
### 3.UI:
 - UI for all the software and additional software saved here. Use QTdesigner to modify the UI files
### 4.Utils:
 - Contains the Utilities for the code. OCTpy is stored here, along with other helper functions used by the code.
### 5.Weights:
 - Weights saved for the segmentation codes.



## Example:

Here are 2 minimal examples of how to use Oct3Dimage. Please [download](https://drive.google.com/drive/folders/1fZpO-Yk9Ta2yVNNObfJ7qWvbU21X_NTB?usp=sharing) the video data and segmenting file to the folder. See the docs for [more examples](https://github.com/UWBAIL/OCTpy/blob/master/examples.ipynb).

```python
# Example 1, exist outside segmentation, make sum projection of superficial layer on flow video, flatten the video
from OCTpy import Oct3Dimage    # import the library
from imageio import imwrite
img_obj = Oct3Dimage()    # create the instance
img_obj.read_flow_data('SubZeissFlow.avi')    # read the flow data
img_obj.read_seg_layers('SubZeissStructFinalX500Y500Z800.txt')    # read the segmentation file
img_obj.plot_flow_layers(250, orientation=1)    #plot the layers, slicing on fast scan #250
print(img_obj)    #print information
img_proj = img_obj.plot_proj(0, 1, 'flow', 'max')    # plot enface images between layer 1 and 2
img_proj_CC = img_obj.plot_proj(2, 2, 'flow', 'max', start_offset=6, end_offset=13) # plot enface images of CC, below BM(layer 2), form pixel 6-13
img_flatten = img_obj.save_flatten_video('flatten_flow.avi', video_type='flow', ref_layer_num=0)    # save the flatten video to current folder, flatten by ILM
imwrite('max_flow_superficial_proj.png', img_proj) # write the enface image to current folder
del img_obj    # delete the object


#%%
# Example 2, use auto segmentation to get ILM, RPE, BM, and project choriod vessel on structural images, display the thickness map, flatten the video
from OCTpy import Oct3Dimage    # import the library
img_obj = Oct3Dimage()    # create the instance
img_obj.read_stru_data('SubZeissStruct.avi')    # read the structural data
img_obj.auto_seg_stru(z_start=10, z_end=1000, auto_range=False, retina_th=50)    #auto segmentation within the range from 10 to 1000, the minimum thickness of retina is 50 
img_obj.plot_stru_layers(300, orientation=2)    #plot the layers, slicing on slow scan #230
img_obj.save_seg_video(step=10, file_name='lines.avi') #save segmentaion result as a movie
print(img_obj)    #print information

img_proj = img_obj.plot_proj(2, 2, 'stru', 'sum', start_offset=10, end_offset=60)    # use sum projection plot choriod vessel, slab between 10 to 60 below layer 2
img_flatten = img_obj.save_flatten_video('flatten_stru.avi', ref_layer_num=2)    # save the flatten video to current folder, flatten by BM(layer 2)
img_obj.plot_proj(0, 0, 'stru', 'max',start_offset=-5, end_offset=10)   # maxinum projection of the RNFL below ILM(layer 0)
img_thick = img_obj.thickness_map(1, 2)     # displey the thickness map between layer 1-2
imwrite('sum_stru_proj.png', img_proj)
del img_obj    # delete the object

```

The results of example 1 and 2:

![Result](result.png)

## API in a nutshell

As a user, you just have to remember a handful of functions: 

- `read_stru_data()` and `read_flow_data()` - load avi and dicom video
- `read_seg_layers()` - load manual segmentation file or use `auto_seg_stru()` to automatically segment the ILM,  RPE and BM
- `save_seg_video()`- save the segmentation result as movie
- `plot_stru_layers()` and `plot_flow_layers()` - plot B-frame images with segmentation lines
- `plot_proj()` - plot projection images
- The segmentation, structural, and flow data are stored in, `InstanceName.layers`, `InstanceName.stru3d`,  `InstanceName.flow` 

## Features

- A python based image processing library for eye imaging group developers

- Cross platform, runs on Windows, Linux, OS X 
- Object-oriented, polymorphism, encapsulation
- Functionalized by various API for multitasking, error handling, message passing 
- Easy to extend with plugins

## Dependencies

- Python 3.5+
- Numpy
- Imageio
- Scipy
- Matplotlib
- ffmpeg
- Pydicom



______________________________

# Getting started

## Installation

1. If Python is not installed on your machine, get it here:

https://anaconda.org/anaconda/python

We recommend install python through Anaconda. Anaconda is a free and open source distribution of the Python and R programming languages for data science and machine learning related applications, that aims to simplify package management and deployment.  

2. Most of the dependencies are automatically included, if you installed python through Anaconda. The only one might be missing is "imageio", as well as ffmpeg. Install them by the following command: 

```shell
python -m pip install --upgrade pip #upgrade pip (package manager) to latest.
pip install imageio
conda install ffmpeg -c conda-forge
pip install -U pydicom

or pip3 install -r requirements.txt
```
3. Clone this repository to your computer.
4. Add this repository to python path by the following command:
```shell
sys.path.insert(0,"\Path\to\OCTpy") # replace "\Path\to\OCTpy" by your the path of this repository
```

## Usage examples

See more examples at the [Jupyter notebook files](https://github.com/UWBAIL/OCTpy/blob/master/examples.ipynb). 



## User API

To see the function information, please use the `help` function as below.

```
>>>help(Oct3Dimage.save_flatten_video)
Help on function save_flatten_video in module OCTpy:

save_flatten_video(self, file_name='flatten.avi', video_type='stru', ref_layer_num=2)
    Save the flatten layers based on the segmentation line
    Args:
        file_name: the flatten file to save
        video_type: stru or flow
        ref_layer_num: the reference layer for flatten image
    Return:
        img_flatten: numpy array style flatten image

```

#### Functions for reading:

- `read_stru_data(file_path)`
- `read_flow_data(file_path)`
- `read_seg_layers(file_path)`
- `read_outside_seg(file_path)`

#### Functions for saving

- `save_flatten_video(file_name, video_type, ref_layer_num)`
- `save_video(img, file_name)`
- `save_seg_video(step, file_name)`
- `save_layers(url)`

#### Image preview 

- `plot_stru_layers(slice_num, orientation)`
- `plot_flow_layers(slice_num, orientation)`
- `plot_slice(slice_num)`

#### Image processing

- `auto_seg_stru(z_start, z_end, auto_range, retina_th)`
- `plot_proj(start, end, datatype, projmethod, start_offset, end_offset)`
- `thickness_map(start, end, smooth)`



```
Help on Oct3Dimage in module OCTpy object:

class Oct3Dimage(builtins.object)
 |  Oct3Dimage is a library that provides an easy interface to read and write
 |  a wide range of oct image data and the segmentation files. It also contains
 |  simple image processing algorithms and easy to make more extensions.
 |  
 |  Methods defined here:
 |  
 |  __init__(self)
 |      init
 |  
 |  __str__(self)
 |      for the print functuon
 |  
 |  auto_seg_stru(self, z_start=0, z_end=1000, auto_range=False, retina_th=50)
 |      Auto seg the ILM, RPE, BM of structure
 |      Args:
 |          z_start: to speed up the processing, only use the pixel within 
 |              this range to perform segmentaiton
 |          z_end: the end of range
 |          auto_range: use the auto range
 |  
 |  call_flatten_video(self, img, file_name='flatten.avi', ref_layer_num=2)
 |      helper function to produce the flatten image, return a 3d array 
 |      Args:
 |          img: 3d image
 |          file_name: output avi filename
 |          ref_layer_num: image number for reference
 |      Return:
 |          img_faltten: the flatten image
 |  
 |  load_3d_data(self, file_path)
 |      Load the 3d data by file type
 |  
 |  load_avi_file(self, file_name)
 |      Load the avi file and return the uint8 nparray
 |      Args:
 |          file_name: machine output avi files
 |      Return:
 |          img: 3D numpy array, with shape (depth, width, framenum)
 |      Example
 |          img = load_avi_file('Zstruct_C.avi')
 |          plt.imshow(img[:, :, 100])
 |  
 |  load_dicom_file(self, file_path)
 |      Load teh dicom file and return the unit8 array 
 |      Args:
 |          file_path: the file path of dcm file
 |  
 |  plot_flow_layers(self, slice_num, orientation=1)
 |      Plot the OCT-A images with segmentation lines
 |      Args:
 |          slice_num: slice number
 |          orientation: 1 for fast scan preview, 2 for slow scan preview
 |  
 |  plot_proj(self, start, end, datatype='stru', projmethod='max', start_offset=0, end_offset=0, display_slice=10)
 |      make the enface projection of the layer
 |      Args:
 |          start: the number of strat layer, 0,1,2..
 |          end: the number of end layer, start+0,1,2....
 |          datatype: 'stru' or 'flow'
 |          projmethod: 'max', 'sum', 'mean', 'maxmean'
 |          start_offset: the strating pixel before the first layer, 
 |              e.g. start=-5, start=7
 |          end_offset: the ending pixel beyond the last layer, 
 |              e.g. end=-4, end=2
 |          display_slice: display the slab on the slice, default=10
 |      Return:
 |          2d-array of en face projection, not normalized
 |  
 |  plot_slice(self, slice_num)
 |      Plot the slice of stru and flow image, if exist
 |      Args:
 |          slice_num: the number of slice
 |  
 |  plot_stru_layers(self, slice_num, orientation=1)
 |      Plot the OCT images with segmentation lines
 |      Args:
 |          slice_num: slice number
 |          orientation: 1 for fast scan preview, 2 for slow scan preview
 |  
 |  read_flow_data(self, file_path)
 |      Read the flow data from file path to self.flow3d
 |      Args:
 |          file_path: the file path of avi file
 |  
 |  read_outside_seg(self, file_path)
 |      Load the outside segmentation file to the object
 |          (still need to be imporoved)
 |      Args:
 |          file_path: the filepath of .mat file
 |  
 |  read_seg_layers(self, seg_path, exist_cali=-1, scale=-1)
 |      Read the segmentation file from seg_path to self.layers
 |      Args:
 |          seg_path: file path of the segmentation txt file
 |          exist_cali: -1 for not use outside calibrate file
 |          scale: -1 for not use scale factor
 |  
 |  read_stru_data(self, file_path)
 |      Read the flow data from file path to the self.stru3d
 |      Args:
 |          file_path: the file path of avi file
 |  
 |  save_flatten_video(self, file_name='flatten.avi', video_type='stru', ref_layer_num=0)
 |      Save the flatten layers based on the segmentation line
 |      Args:
 |          file_name: the flatten file to save
 |          video_type: stru or flow
 |          ref_layer_num: the reference layer for flatten image
 |      Return:
 |          img_flatten: numpy array style flatten image
 |  
 |  save_layers(self, url='layers.npy')
 |      Save the segmentation file to the disk
 |      Args:
 |          file_name: the url of file, should end with .mat or .npy
 |  
 |  save_seg_video(self, step=10, file_name='lines.avi')
 |      Save the video with lines
 |      Args:
 |          step: the step to plot the B-scans
 |          file_name = name of the video
 |  
 |  save_video(self, img, file_name='out.avi')
 |      Save the video by te filename, avi or dicom
 |      Args:
 |          img: 3d numpy array
 |          file_name: url to save, endwith 'avi' or 'dicom'
 |  
 |  thickness_map(self, start, end, smooth=True)
 |      Get the thickness map of layers
 |      Args:
 |          strat: the start layer number
 |          end: the end of layer nunber
 |          smooth: do smooth on slow scan 
 |      Retrun: The thickness map
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
```

## Developer API

The object’s attributes:

Oct3Dimage.\__dict__

```
mappingproxy({'_Oct3Dimage__plot_slice_layers': <function OCTpy.Oct3Dimage.__plot_slice_layers>,
              '__dict__': <attribute '__dict__' of 'Oct3Dimage' objects>,
              '__doc__': '\n    Oct3Dimage is a library that provides an easy interface to read and write\n    a wide range of oct image data and the segmentation files. It also contains\n    simple image processing algorithms and easy to make more extensions.\n    ',
              '__init__': <function OCTpy.Oct3Dimage.__init__>,
              '__module__': 'OCTpy',
              '__str__': <function OCTpy.Oct3Dimage.__str__>,
              '__weakref__': <attribute '__weakref__' of 'Oct3Dimage' objects>,
              '_get_auto_range': <function OCTpy.Oct3Dimage._get_auto_range>,
              '_get_one_layer_image': <function OCTpy.Oct3Dimage._get_one_layer_image>,
              '_get_seg_file_info': <function OCTpy.Oct3Dimage._get_seg_file_info>,
              '_get_seg_lines': <function OCTpy.Oct3Dimage._get_seg_lines>,
              '_max_mean_proj_layers': <function OCTpy.Oct3Dimage._max_mean_proj_layers>,
              '_max_proj_layers': <function OCTpy.Oct3Dimage._max_proj_layers>,
              '_min_mean_proj_layers': <function OCTpy.Oct3Dimage._min_mean_proj_layers>,
              '_min_proj_layers': <function OCTpy.Oct3Dimage._min_proj_layers>,
              '_read_calibrate': <function OCTpy.Oct3Dimage._read_calibrate>,
              '_save_avi_file': <function OCTpy.Oct3Dimage._save_avi_file>,
              '_save_dicom_file': <function OCTpy.Oct3Dimage._save_dicom_file>,
              '_sum_mean_proj_layers': <function OCTpy.Oct3Dimage._sum_mean_proj_layers>,
              '_sum_proj_layers': <function OCTpy.Oct3Dimage._sum_proj_layers>,
              'auto_seg_stru': <function OCTpy.Oct3Dimage.auto_seg_stru>,
              'call_flatten_video': <function OCTpy.Oct3Dimage.call_flatten_video>,
              'load_3d_data': <function OCTpy.Oct3Dimage.load_3d_data>,
              'load_avi_file': <function OCTpy.Oct3Dimage.load_avi_file>,
              'load_dicom_file': <function OCTpy.Oct3Dimage.load_dicom_file>,
              'plot_flow_layers': <function OCTpy.Oct3Dimage.plot_flow_layers>,
              'plot_proj': <function OCTpy.Oct3Dimage.plot_proj>,
              'plot_slice': <function OCTpy.Oct3Dimage.plot_slice>,
              'plot_stru_layers': <function OCTpy.Oct3Dimage.plot_stru_layers>,
              'read_flow_data': <function OCTpy.Oct3Dimage.read_flow_data>,
              'read_outside_seg': <function OCTpy.Oct3Dimage.read_outside_seg>,
              'read_seg_layers': <function OCTpy.Oct3Dimage.read_seg_layers>,
              'read_stru_data': <function OCTpy.Oct3Dimage.read_stru_data>,
              'save_flatten_video': <function OCTpy.Oct3Dimage.save_flatten_video>,
              'save_layers': <function OCTpy.Oct3Dimage.save_layers>,
              'save_seg_video': <function OCTpy.Oct3Dimage.save_seg_video>,
              'save_video': <function OCTpy.Oct3Dimage.save_video>,
              'thickness_map': <function OCTpy.Oct3Dimage.thickness_map>})
```

