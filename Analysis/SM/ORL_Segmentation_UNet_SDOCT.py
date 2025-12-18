# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 03:03:30 2106

@author: Thanu
"""

import numpy as np
from scipy.ndimage import median_filter
import tensorflow as tf
import logging
import time
import os
import keras.layers as layers
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
from OCTpy import Oct3Dimage
import tensorflow.keras.layers as layers
from scipy import integrate
from tqdm import tqdm
from skimage import io
from skimage.registration import phase_cross_correlation
from skimage.morphology import remove_small_objects
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import medfilt2d
import numpy as np
import cv2
from skimage import io
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from skimage import measure, exposure


add = os.getcwd()
img_obj = Oct3Dimage()
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class TransposeConv(tf.keras.layers.Layer):
    """ Self-define 2D Transpose Convolution Layer with kernel initializer  """

    def __init__(self, fs=64, ks=3, s=2, use_bias=False, padding="same",
                 **kwargs):
        """
        Args:
            fs: int, filter size of the Transpose Conv2D layer
            ks: int, kernel size of the Transpose Conv2D layer
            s: int, strides of the Transpose Conv2D layer
            use_bias: bool, decide use the bias or not inside the layer
            padding: string, optional setting: "same" or "valid"
        """
        super(TransposeConv, self).__init__(**kwargs)
        self.he_initializer = tf.keras.initializers.HeNormal()
        self.random_initializer = tf.random_normal_initializer(0., 0.02)
        self.fs = fs
        self.ks = ks
        self.s = s
        self.use_bias = use_bias
        self.padding = padding

        self.transpose_conv2d_layer = tf.keras.layers.Conv2DTranspose(
            filters=fs, kernel_size=(ks, ks), strides=(s, s), padding=padding,
            use_bias=use_bias, kernel_initializer=self.random_initializer,
        )

    def call(self, inputs):
        x = self.transpose_conv2d_layer(inputs)
        return x

    def get_config(self):
        config = super(TransposeConv, self).get_config()
        config.update(
            {
                "fs": self.fs,
                "ks": self.ks,
                "s": self.s,
                "use_bias": self.use_bias,
                "padding": self.padding,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Conv(tf.keras.layers.Layer):
    """ Self-define 2D Convolution Neural Layer with kernel initializer """

    def __init__(self, fs=64, ks=3, s=1, use_bias=False, padding="same",
                 **kwargs):
        """
        Args:
            fs: int, filter size of the Conv2D layer
            ks: int, kernel size of the Conv2D layer
            s: int, strides of the Conv2D layer
            use_bias: bool, decide use the bias or not inside the layer
            padding: string, optional setting: "same" or "valid"
        """
        super(Conv, self).__init__(**kwargs)
        self.he_initializer = tf.keras.initializers.HeNormal()
        self.random_initializer = tf.random_normal_initializer(0., 0.02)
        self.fs = fs
        self.ks = ks
        self.s = s
        self.use_bias = use_bias
        self.padding = padding

        self.conv2d_layer = tf.keras.layers.Conv2D(
            filters=fs, kernel_size=(ks, ks), strides=(s, s), padding=padding,
            use_bias=use_bias, kernel_initializer=self.random_initializer,
        )

    def call(self, inputs):
        x = self.conv2d_layer(inputs)
        return x

    def get_config(self):
        config = super(Conv, self).get_config()
        config.update(
            {
                "fs": self.fs,
                "ks": self.ks,
                "s": self.s,
                "use_bias": self.use_bias,
                "padding": self.padding,
            }
        )
        return config
class ConvBlock(tf.keras.layers.Layer):
    """ Very basic conv+bn+activation blocks, build this block for convenience
    """

    def __init__(self, fs=64, ks=3, s=1, use_bias=False, use_BN=True,
                 acti="relu", **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.fs = fs
        self.ks = ks
        self.s = s
        self.use_bias = use_bias
        self.use_BN = use_BN

        self.conv2d_layer = Conv(fs=fs, ks=ks, s=s, use_bias=use_bias)
        self.bn_layer = layers.BatchNormalization()
        self.activation = get_activation_layer(activation_type=acti)

    def call(self, inputs):
        x = self.conv2d_layer(inputs)
        if self.use_BN:
            x = self.bn_layer(x)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update({
            "fs": self.fs,
            "ks": self.ks,
            "s": self.s,
            "use_bias": self.use_bias,
            "use_BN": self.use_BN,
        })

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Variables:
    def __init__(self):
        # TODO: parameters for keras.fit function
        self.fitParas = {
            "bs": 4,  # bs: batch size
            "epoch_sv": 12,  # sv: supervised
        }

        # TODO: number of train/valid datasets
        self.num_of_ds = 2300

        # TODO: shape of the datasets and labels
        self.width = 512
        self.height = 512
        self.channel = 1
        self.image_shape = (self.width, self.height, self.channel)

        # TODO: number of segmentation class
        self.seg_num = 2  # number of segmentation class you want. please input
        # 2 if you want to do binary segmentation/classification.

        self.optimParas = {
            "beta1": 0.9,
            "beta2": 0.999,
            "learn_rate": 1e-3,
            "decay_rate": 0.98,
            "decay_step": 40000,
        }

        # additional information.
        self.ts = time.localtime()
        self.time_date = str(self.ts[0]) + '.' + str(self.ts[1]) + '.' + str(
            self.ts[2]) + ' -- ' + str(self.ts[3]) + ':' + str(self.ts[4])

class UNet:
    def __init__(self, image_shape, out_cls):
        self.image_shape = image_shape
        self.out_cls = out_cls

    def __call__(self, include_top=True):
        Inputs = tf.keras.Input(self.image_shape)
        n_filters = [32, 32, 64, 64, 128]

        x = ConvBlock(use_BN=False)(Inputs)

        # 512 * 512 *64
        x = ConvBlock(fs=n_filters[0])(x)
        x = ConvBlock(fs=n_filters[0])(x)
        concat_1 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        # 256 * 256 * 128
        x = ConvBlock(fs=n_filters[1])(x)
        x = ConvBlock(fs=n_filters[1])(x)
        concat_2 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        # 128 * 128 * 256
        x = ConvBlock(fs=n_filters[2])(x)
        x = ConvBlock(fs=n_filters[2])(x)
        concat_3 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        # 64 * 64 * 512
        x = ConvBlock(fs=n_filters[3])(x)
        x = ConvBlock(fs=n_filters[3])(x)
        concat_4 = x
        x = tf.keras.layers.MaxPooling2D()(x)

        # 32 * 32 * 1024
        x = ConvBlock(fs=n_filters[4])(x)
        x = ConvBlock(fs=n_filters[4])(x)

        x = TransposeConv(fs=n_filters[3])(x)  # 64 * 64 * 512
        x = tf.keras.layers.Concatenate()([x, concat_4])
        x = ConvBlock(fs=n_filters[3])(x)
        x = ConvBlock(fs=n_filters[3])(x)

        x = TransposeConv(fs=n_filters[2])(x)  # 128 * 128 * 256
        x = tf.keras.layers.Concatenate()([x, concat_3])
        x = ConvBlock(fs=n_filters[2])(x)
        x = ConvBlock(fs=n_filters[2])(x)

        x = TransposeConv(fs=n_filters[1])(x)  # 256 * 256 *128
        x = tf.keras.layers.Concatenate()([x, concat_2])
        x = ConvBlock(fs=n_filters[1])(x)
        x = ConvBlock(fs=n_filters[1])(x)

        x = TransposeConv(fs=n_filters[0])(x)  # 512 * 512 * 64
        x = tf.keras.layers.Concatenate()([x, concat_1])
        x = ConvBlock(fs=n_filters[0])(x)
        x = ConvBlock(fs=n_filters[0])(x)

        x = Conv(fs=self.out_cls, ks=1, s=1)(x)

        if include_top:
            if self.out_cls == 1:
                x = get_activation_layer("sigmoid")(x)
            elif self.out_cls > 1:
                x = get_activation_layer("softmax")(x)

        models = tf.keras.models.Model(inputs=Inputs, outputs=x, name="UNet")
        return models


def OAC_calculation(img):
    """
    calculate the OAC volume through the structural data
    Returns:
    OAC volume # TODO: this is very slow and cannot handle a very large matrix
    """
    try:
        OCTL = ((10 ** (np.single(img.astype(float)) / 80) - 1) / 7.1)  # linear oct
        M = integrate.cumulative_trapezoid(np.flipud(OCTL), axis=0)  # M size(H-1, W, frame)
        M = np.flipud(M)
        eps = 1e-10
        JLin = (OCTL[:-1, :, :] / (M * 2 * (3 / img.shape[0] + eps))) # pixel_size_z = 3mm/pixel_num_Z
    except Exception as e:
        print(e)


    return JLin

def get_activation_layer(activation_type="relu"):
    flags = activation_type.lower()

    activation_library = {
        "relu": tf.keras.layers.ReLU(),
        "leakyrelu": tf.keras.layers.LeakyReLU(alpha=0.2),
        "prelu": tf.keras.layers.PReLU(shared_axes=[1, 2]),
        "softmax": tf.keras.layers.Softmax(),
        "sigmoid": tf.keras.activations.sigmoid,
        "gelu": tf.keras.activations.gelu,
    }

    if flags in activation_type:
        return activation_library[flags]
    else:
        raise ValueError("@Jinpeng: ERROR INPUT OF THE ACTIVATION LAYER TYPE")

def count_files(folder_path, extension):
    count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(extension):
            count += 1
    return count


def read_tensor(tar_fp):
    img = tar_fp
    img = img / 255.
    img = tf.clip_by_value(img, 0, 1)
    return img


def resize_and_pad_image_top(image_path):
    image = Image.fromarray(image_path)
    resized_image = image.resize((500, 1200), Image.LANCZOS)
    padded_image = np.uint8(resized_image)
    return padded_image

def imread_binary(path):
    image = path
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    binary_image = binary_image // 255
    return binary_image

def align_images_intensity_based_vertical(image, reference):
    # after cropping, aligning the main part of the image to the center
    npimage = np.array(image)
    npref = np.array(reference)
    shift, error, diffphase = phase_cross_correlation(npref, npimage, upsample_factor=100)
    vertical_shift = int(round(shift[0]))
    #print(vertical_shift)
    return npimage, vertical_shift

def shiftz(image, shift_value):
    # alignment of the images according to the shift values in array
    npimage = np.array(image)
    shifted_image = np.zeros_like(npimage)

    if shift_value > 0:
        shifted_image[shift_value:, :] = npimage[:-shift_value, :]
    elif shift_value < 0:
        # Shifting up
        shift_value = -shift_value  # Make the shift value positive
        shifted_image[:-shift_value, :] = npimage[shift_value:, :]
    else:
        # No shift
        shifted_image = npimage

    return np.uint8(shifted_image)

def reverse_vertical_shifts_spatially_for_masks(mask_array, shifts):
    reversed_masks = np.zeros_like(mask_array)
    for i in range(mask_array.shape[2]):
        individual_mask = mask_array[:, :, i]
        vertical_shift = - shifts[i]
        if vertical_shift > 0:
            reversed_masks[vertical_shift:, :, i] = individual_mask[:-vertical_shift]
        elif vertical_shift < 0:
            reversed_masks[:vertical_shift, :, i] = individual_mask[-vertical_shift:]
        else:
            reversed_masks[:, :, i] = individual_mask
    return reversed_masks


def clean_segmentation_mask(mask, min_area=100, max_bottom_height=50):
    # Label connected components
    labeled, num_features = ndimage.label(mask)

    # Measure properties of labeled regions
    props = measure.regionprops(labeled)

    # Sort regions by area, descending
    props.sort(key=lambda x: x.area, reverse=True)

    # Create a new mask with only the largest region
    clean_mask = np.zeros_like(mask)
    if props:
        largest_region = props[0]
        clean_mask[labeled == largest_region.label] = 1

        # Remove any part of the largest region that's in the bottom section
        bottom_limit = mask.shape[0] - max_bottom_height
        clean_mask[bottom_limit:, :] = 0

    return clean_mask


def correct_layer_with_safeguards(layer1, layer2, window_size=5,
                                  threshold=2):
    # Apply median filter to both layers
    layer1_filtered = medfilt2d(layer1, kernel_size=window_size)
    layer2_filtered = medfilt2d(layer2, kernel_size=window_size)

    # Calculate the difference between original and filtered layer2
    diff = np.abs(layer2 - layer2_filtered)

    # Identify outliers in layer2
    mad = np.median(np.abs(diff - np.median(diff)))
    outliers = diff > (threshold * mad)

    # Create a mask where layer1 should not be corrected
    no_correction_mask = outliers | (layer1 <= layer2)

    # Apply correction only where needed
    corrected_layer1 = np.where(no_correction_mask, layer1, layer2)
    # print("hi")

    return corrected_layer1


def correct_2d_array(arr1, arr2, flag):
    if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
        raise ValueError("Input arrays must have the same dimensions")

    corrected_array = [row[:] for row in arr1]

    for i in range(len(arr1)):
        for j in range(len(arr1[0] - 1)):
            if flag == 1:
                if corrected_array[i][j] > arr2[i][j]:
                    if j > 0:
                        corrected_array[i][j] = corrected_array[i][
                            j - 1]  # Use previous element in the same row
                    elif i > 0:
                        corrected_array[i][j] = corrected_array[i - 1][
                            j]  # Use previous element in the same column
                    else:
                        corrected_array[i][j] = arr2[i][
                            j]  # If no previous element, use value from arr2
            elif flag == 0:
                if corrected_array[i][j] < arr2[i][j]:
                    if j > 0:
                        corrected_array[i][j] = corrected_array[i][
                            j - 1]  # Use previous element in the same row
                    elif i > 0:
                        corrected_array[i][j] = corrected_array[i - 1][
                            j]  # Use previous element in the same column
                    else:
                        corrected_array[i][j] = arr2[i][
                            j]  # If no previous element, use value from arr2
    return corrected_array


def interpolate_empty_columns(array):
    """
    Interpolate values for empty columns (where all values are 0) based on nearby valid columns.
    """
    # Create a copy of the input array
    result = array.copy()

    # Find columns that are all zeros
    empty_cols = np.all(array == 0, axis=0)

    if np.all(empty_cols):
        return array  # Return original if all columns are empty

    # Get indices of valid columns
    valid_cols = np.where(~empty_cols)[0]

    # Process each empty column
    for col in range(len(empty_cols)):
        if empty_cols[col]:
            # Find nearest valid columns to the left and right
            left_valid = valid_cols[valid_cols < col]
            right_valid = valid_cols[valid_cols > col]

            if len(left_valid) == 0:
                # If no valid columns to the left, use nearest right
                nearest_valid = right_valid[0]
                result[:, col] = array[:, nearest_valid]
            elif len(right_valid) == 0:
                # If no valid columns to the right, use nearest left
                nearest_valid = left_valid[-1]
                result[:, col] = array[:, nearest_valid]
            else:
                # Interpolate between nearest valid columns
                left_col = left_valid[-1]
                right_col = right_valid[0]
                weight_right = (col - left_col) / (right_col - left_col)
                weight_left = 1 - weight_right
                result[:, col] = (array[:, left_col] * weight_left +
                                  array[:, right_col] * weight_right)

    return result




def ORL_segmentation_UNet_SDOCT(img_obj, flag):

    # Part 1: Converting to OAC and defining variables
    logging.debug("Entering ORL_segmentation_UNet_ch")
    try:
        v = Variables()
        size_x = size_y = v.width
        n_class = v.seg_num

        saved_weight_list = {
            'UNet': 'UNet\\UNet',
        }
        dicom_data = img_obj.stru3d


        OAC_conv = OAC_calculation(dicom_data)
        max_val = np.nanpercentile(OAC_conv, 99)
        OAC_conv[OAC_conv > max_val] = max_val
        OAC_conv = OAC_conv / max_val * 255

        #if (flag ==1):
        # uncomment for non-OAC data
            #img_ipl = dicom_data
            #img_ipl = np.squeeze(img)

        img = OAC_conv
        Y, X, Z = img.shape
        Y = Y + 1
        res_mask_size = 1200
        to_test = np.empty((512, 512, Z), dtype=np.uint8)
        mask_op = np.empty((512, 512, Z), dtype=np.uint8)
        img_current = np.empty((512, 512, Z), dtype=np.uint8)
        dfo = "Reference_Resized.png"
        dfolder = os.path.join(add, dfo)
        reference = io.imread(dfolder)
        shifty = []
        aligned_image = np.empty((512, 512, Z), dtype=np.uint8)
        #aligned_image = np.empty((Y, X, Z), dtype=np.uint8)


        #%% Part 2: Initial cropping and Aligning the layers to reference
        #init values
        equal_flag = 0
        blank_row = np.zeros((1, X), dtype=np.uint8)
        base_img = img[:, :, 3].astype(np.uint8)
        curr_img = np.vstack((blank_row, np.array(base_img)))
        crop_im = Image.fromarray(curr_img).convert('L')
        width, height = crop_im.size
        bottom = res_mask_size
        top = 0
        left = 0
        right = width
        refe = Image.fromarray(reference).convert('L')
        refe_crop = refe.crop((left, top, right, bottom))
        refe_crop = refe_crop.resize((512, 512), Image.LANCZOS)

        for q in tqdm(range(Z), desc="Layer Alignment and Resizing (1/2)"):
            base_img = img[:, :, q].astype(np.uint8)
            curr_img = np.vstack((blank_row, np.array(base_img)))
            crop_im = Image.fromarray(curr_img).convert('L')
            #width, height = crop_im.size
            if width == 512 and height == 512:
                #print("hi")
                img_crop1 = crop_im.resize((512,512), Image.LANCZOS)
                equal_flag = 1
            else:
                img_crop1 = crop_im.crop((left, top, right, bottom))
                img_crop1 = img_crop1.resize((512,512), Image.LANCZOS)
            npimage, shift_amt = align_images_intensity_based_vertical(np.array(img_crop1), np.array(refe_crop))
            shifty.append(shift_amt)
            aligned_image[:, :, q] = npimage
            time.sleep(0.1)

        shifty = gaussian_filter(shifty, sigma=4)

        for y in range(Z):
            aligned_image_shrt = aligned_image[:, :, y].astype(np.uint8)
            shiftfornow = shifty[y]
            shiftx = shiftz(aligned_image_shrt, shiftfornow)
            to_test[:, :, y] = shiftx.T




        #%% Part 3: Cropping and resizing the images to 512x512

        for i in range(Z):
            img_curr = to_test[:, :, i]
            img_matched = exposure.match_histograms(np.array(img_curr),np.array(refe_crop))
            img_matched[img_matched < 10] = 0
            img_res = Image.fromarray(img_matched).rotate(-90, expand=True)
            #img_res = im.resize((512, 512), Image.LANCZOS)
            if i == 2:  # First slice
                img_current[:, :, 0] = img_current[:, :, 1]
                img_current[:, :, i] = np.array(img_res)
            elif i == Z - 1:  # Last slice
                img_current[:, :, i] = img_current[:, :,
                                       i - 1]  # Copy from previous slice
            else:  # Middle slices
                img_current[:, :, i] = np.array(img_res)



        #result_path = os.path.join(debugfolder, f'processed_img_i_{1}.png')
        #io.imsave(result_path, np.uint8(img_current[:, :, 1]))


        #%% Part 4: Using resized images and generating segmentation layers




        for marker in saved_weight_list.keys():
            model = UNet((size_x, size_y, 1), n_class)()
            model.load_weights(saved_weight_list[marker]).expect_partial()

            for img_nums in tqdm(range(Z), desc="Layer Segmentation progress (2/2)"):
                img_fp = img_current[:, :, img_nums]
                # Saving and debugging images
                #debugfolder = "D:\\Thanu_Testing_Seg_smol\\Checkseg"
                #if not os.path.exists(debugfolder):
                #    os.makedirs(debugfolder)
                #result_path = os.path.join(debugfolder,
                #                           f'processed_img_i_{img_nums + 1}.png')
                #io.imsave(result_path, np.uint8(img_fp))
                input_img = tf.expand_dims(read_tensor(img_fp), axis=0)
                pred_mask = model(input_img)
                pred_mask *= 255
                pred_mask_argmax = np.argmax(np.array(pred_mask[0]), axis=-1)
                j = np.uint8(pred_mask_argmax * 255)
                #jj = resize_and_pad_image_top(j)
                mask_cleaned = remove_small_objects(j, min_size=400)
                ff_flip = np.fliplr(mask_cleaned)
                #result_path = os.path.join(debugfolder,
                #                           f'processed_mask_i_{img_nums + 1}.png')
                #io.imsave(result_path, np.uint8(ff_flip))
                mask_op[:, :, img_nums] = ff_flip

                time.sleep(0.01)

        #%% Reversing the shifts to match original image

        #mask_op[:, :, 0] = mask_op[:, :, 4]
        #mask_op[:, :, 1] = mask_op[:, :, 4]
        #mask_op[:, :, 2] = mask_op[:, :, 4]
        #mask_op[:, :, 3] = mask_op[:, :, 4]

        original_rev = reverse_vertical_shifts_spatially_for_masks(mask_op, shifty)
        original = np.zeros((Y, X, Z))

        for n in range(1, Z):
            resizing_image = Image.fromarray(original_rev[:, :, n])
            resizing_image_result = resizing_image.resize((X, Y), Image.LANCZOS)
            original[:, :, n] = np.uint8(resizing_image_result)


        #%% Creating layers files

        layers = np.zeros((X, Z, 5))
        layers_ch = np.zeros((X, Z, 5))
        #maybe X or Y above

        for i in range(Z):
            mask = imread_binary(original[:, :, i]) > 0
            if i == 0:  # First slice
                mask = imread_binary(original[:, :, 1]) > 0
            elif i == Z-1:  # Last slice
                mask = imread_binary(original[:, :, Z-2]) > 0
            mask_uint = np.uint8(mask)
            if equal_flag == 0:
                kernel = np.ones((10, 10), np.uint8)
                mask_cl = cv2.morphologyEx(mask_uint, cv2.MORPH_OPEN, kernel)
                mask_cl = clean_segmentation_mask(mask_cl)
            else:
                kernel = np.ones((3, 3), np.uint8)
                mask_cl = cv2.morphologyEx(mask_uint, cv2.MORPH_OPEN, kernel)
                mask_cl = np.uint8(mask_uint)

            # uncomment the following to save the individual images formed at this step

            #debugfolder = "D:\\Thanu_Testing_Seg_smol\\Checkseg"
            #if not os.path.exists(debugfolder):
            #    os.makedirs(debugfolder)
            #result_path = os.path.join(debugfolder,
            #                           f'mask_processed_img_i_{i + 1}.png')
            #mask_cl_array = np.array(mask_cl * 255,
            #                         dtype=np.uint8)  # Convert to uint8 for 0-255 range
            #Image.fromarray(mask_cl_array).save(result_path)



            opl = np.argmax(mask_cl, axis=0)
            opl[mask_cl[opl, np.arange(mask_cl.shape[1])] == 0] = 0
            opl = interpolate_empty_columns(opl.reshape(1, -1))[0]

            flipped_mask_cl = np.flipud(mask_cl)
            choi = np.argmax(flipped_mask_cl, axis=0)
            choi[flipped_mask_cl[choi, np.arange(flipped_mask_cl.shape[1])] == 0] = 0
            choi = flipped_mask_cl.shape[0] - choi - 1
            choi = interpolate_empty_columns(choi.reshape(1, -1))[0]

            layers[:, i, 0] = opl - 3
            layers[:, i, 1] = choi + 13
            layers[:, i, 2] = choi + 25
            layers[:, i, 3] = opl - 11
            layers[:, i, 4] = choi + 21 #BM

        layers -= 1

        #layers[:, :, 0] = gaussian_filter(layers[:, :, 0], sigma=1)
        layers[:, :, 2] = gaussian_filter(layers[:, :, 2], sigma=8)
        layers[:, :, 3] = gaussian_filter(layers[:, :, 3], sigma=8)
        iao3 = gaussian_filter(layers[:, :, 1], sigma=1.5)
        excheck = np.transpose(iao3, (1, 0))
        medcheck = median_filter(excheck, size=2)
        layers_ch[:, :, 1] = np.transpose(medcheck, (1, 0))
        layers_ch[:, :, 0] = layers[:, :, 0]

        layers_debug = np.copy(layers_ch)



        #%% Trying to remove outliers

        channel_2 = layers[:, :, 2]
        #channel_2 = median_filter(channel_2, size=5)
        channel_3 = layers[:, :, 3]
        #channel_3 = median_filter(channel_3, size=5)

        #debug
        #ch2 = Image.fromarray(np.uint8(channel_2)).save(os.path.join(debugfolder, f'Channel2afterfilter.png'))
        #ch3 = Image.fromarray(np.uint8(channel_3)).save(os.path.join(debugfolder, f'Channel3afterfilter.png'))
        #io.imsave(os.path.join(debugfolder, f'Channel2afterfilter.png'), ch2)
        #io.imsave(os.path.join(debugfolder, f'Channel3afterfilter.png'), ch3)



        layer1 = layers[:, :, 1]
        layer1 = correct_2d_array(layer1, channel_2, 1)
        layer0 = layers[:, :, 0]
        layer0 = correct_2d_array(layer0, channel_3, 0)
        layers[:, :, 0] = layer0
        layers[:, :, 1] = layer1


        #%% Final Layers to send to software
        layers_send = np.zeros((X, Z, 3))

        #for qo in range(0, 12):
        #    layers[:, qo, 1] = layers[:, 12, 1]
        #    #print("interchange on zero")

        layers_send[:, :, 0] = layers[:, :, 0]
        layers_send[:, :, 1] = layers[:, :, 1]
        layers_send[:, :, 2] = layers[:, :, 4] #BM
        #ia_bm = np.transpose(layers_send[:, :, 2])
        #ia_bm = median_filter(ia_bm, size = 4)
        #layers_send[:, :, 2] = np.transpose(ia_bm, (1,0))

        #oopl = gaussian_filter(layers[:, :, 0], sigma=1)
        #cchoi = gaussian_filter(layers[:, :, 1], sigma=1)
        oopl = layers[:, :, 0]
        cchoi = layers[:, :, 1]

        interchanged_arr_3d = np.transpose(cchoi, (1, 0))
        interchanged_arr_opl_3d = np.transpose(oopl, (1, 0))
        interchanged_arr_3d = median_filter(interchanged_arr_3d, size=2)
        interchanged_arr_opl_3d = median_filter(interchanged_arr_opl_3d, size=2)
        layers_send[:, :, 1] = np.transpose(interchanged_arr_3d, (1, 0))
        layers_send[:, :, 0] = np.transpose(interchanged_arr_opl_3d, (1, 0))
        print("Segmentation done. Layers file exported")
        #layers_send[:, Z, 1] = layers_send[:, Z - 3, 1]
        layers_send[:, Z - 1, 1] = layers_send[:, Z - 3, 1]
        layers_send[:, Z - 2, 1] = layers_send[:, Z - 3, 1]

        return layers_send
    except Exception as e:
        logging.error(f"Error in subprogram: {str(e)}")
    logging.debug("Exiting subprogram")


        #%% Uncomment to run program on its own

#add= os.getcwd()
#img_obj = Oct3Dimage()
#url = "D:\\Thanu_Testing_Seg_smol\\"
#files = os.path.join(url, "Doheny_struct_smol_350_1024_100.avi")
#img_obj.read_stru_data(files)
#print("running sample dataset")
#paras = {'retina_lower_offset': 5, 'retina_upper_offset': 5, 'upper_bound': 5, 'lower_bound': 5}
#p_layer = ORL_segmentation_UNet_SDOCT(img_obj,paras)
#p_layer = OAC_checksum(img_obj, url)