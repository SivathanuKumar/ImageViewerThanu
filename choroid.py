import numpy as np
import cv2
from imageio import imread
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import threshold_otsu
from skimage import morphology
from PIL import Image


# inputs = loadmat('D:/ChoroidPackage/test_cases/inputs.mat')
#
# vol = inputs['inputs']
# StrucAttnCub = vol[0, 0]['StrucAttnCub']
# BMSeg = vol[0, 0]['BMSeg']
# choroidSeg = vol[0, 0]['choroidSeg']
# ONHseg = vol[0, 0]['ONHseg']


def vol_mask(size, BMSeg, choroidSeg):
    """
    get the volume mask
    :param size:
    :param BMSeg:
    :param choroidSeg:
    :return:
    """
    VolMask = np.ones(size)
    # print(VolMask.shape)
    # print(BMSeg)
    # print(choroidSeg)
    VolMask[:] = np.NaN

    BMSeg = np.floor(BMSeg).astype(int)
    choroidSeg = np.floor(choroidSeg).astype(int)
    try:
        # print(BMSeg)
        # print(choroidSeg)
        for i in range(0, size[1]):
            for j in range(0, size[2]):
                for t in range(BMSeg[i, j], choroidSeg[i, j]):
                    VolMask[t, i, j] = 1
    except Exception as e:
        print(e)
        print('Thickness out of range, please check and edit the segmentation')

    return VolMask

def minimum_projection(cube, thickness):
    """
    minimum projeciton of the cube
    :param cube:
    :param thickness:
    :return:
    """
    cube[cube == 0] = 250
    num = (np.sort(cube, axis=0))
    k2 = np.floor(np.mean(thickness) / 7).astype(int)
    k = min(k2, 15)
    yend = num[-k::,:,:]
    yendmin = num[1:k, :, :]
    Min_Mean = np.squeeze(np.sum(yendmin, axis=0)) / k
    Min_Mean = Image.fromarray(Min_Mean).resize((cube.shape[1], cube.shape[2]), Image.BICUBIC)

    return Min_Mean

def Flatimage(origImage, colshifts, shiftsize):

    #% Shits origImage columns according to colshifts
    colshifts = colshifts.astype(int)
    nCols = origImage.shape[1]
    origImage = np.pad(origImage, ((shiftsize, shiftsize), (0, 0)), 'constant', constant_values=(0, 0))
    flatImage = np.zeros(origImage.shape)

    #% Shift col by col
    for j in range(nCols):
        flatImage[:, j] = np.roll(origImage[:, j], colshifts[j])

    flatImage = flatImage[shiftsize+1: -shiftsize, :]
    return flatImage

def choroid_BM_CSI(StrucAttnCub, BMSeg, choroidSeg, ONHseg):
    """  
    :param StrucAttnCub: 
    :param BMSeg: 
    :param choroidSeg: 
    :param ONHseg: 
    :return: 
    """


    (_, N_Ascan, N_Frame) = np.shape(StrucAttnCub)

    ONHmask = np.ones((N_Ascan, N_Frame))
    ONHmask_1 = 1 - (ONHseg)
    if ONHseg is not None:
        ONHmask[ONHseg==1] = np.NaN

    # Thickness map and MCT
    BMSeg = np.floor(BMSeg.astype(float))
    BMSeg[BMSeg < 1] = 1
    # choroidSeg = np.round(np.float(choroidSeg))
    thickness = choroidSeg - BMSeg
    thickness[thickness <= 0] = 1
    thickness[thickness >299 ] = 299


    in_temp = np.flipud(np.rot90(thickness * ONHmask_1))
    in_temp[in_temp > 250] = 250  # thickness range [0,250]
    # plt.imshow(in_temp)
    # plt.show()

    in_temp = gaussian_filter(in_temp, sigma=(4, 1))

    map_thickness = np.uint8(in_temp)
    # cv2.normalize(in_temp, map_thickness, 0, 255, cv2.NORM_MINMAX)

    mct = np.mean(np.mean(thickness)) * 1.95

    # %%Flat video at BM, then move BM to z-position 500, crop the whole
    #     %%OCT-Attn video from 251 to 800 at z-direction
    #     %%The reason we need flat is later adjascent 3 B-scans were averaged
    cropSTR_EC = np.zeros(( 550, N_Ascan, N_Frame))
    for iY in range(N_Frame):
        ims_EC = StrucAttnCub[:, :, iY]
        colShifts2 = 500 - BMSeg[:, iY]
        maxShift2 = np.max(np.abs(colShifts2)).astype(int)
        shiftedSTR_EC = Flatimage(ims_EC, colShifts2, maxShift2)
        cropSTR_EC[:, :, iY] = shiftedSTR_EC[250:800, :]

    # plt.imshow(cropSTR_EC[:,:,100])
    # plt.show()


    # % % enface vasculature image

    flatsuface = np.ones((cropSTR_EC.shape[1], cropSTR_EC.shape[2])) * 251

    volumeMask_enface = np.uint8(vol_mask(cropSTR_EC.shape, flatsuface, (flatsuface + thickness)))  # mask for choroid slab, all pixels outside choroid is set to 255.
    choroid_enface = cropSTR_EC.astype(float)
    choroid_enface[volumeMask_enface == 0] = 255

    choroid_enface = gaussian_filter(choroid_enface, sigma=0.5)  #choroid_enface = smooth3(choroid_enface, 'gaussian', 3, 0.5)

    Inv_choroid = minimum_projection(choroid_enface, thickness)  # mean of the smallest 15 pixels, if choroid thickness is less then 105 pixels, then use the smallest 15 % of the pixels.
    Inv_choroid = np.asarray(Inv_choroid)   #convert Image to np array
    Inv_choroid = (255 - Inv_choroid) / 255
    A = equalize_adapthist(Inv_choroid, clip_limit=0.02)

    v_min, v_max = np.percentile(A, (0.15, 95))  # imadjust J = imadjust(A, [0.15; 0.95], [0; 1]);
    J = rescale_intensity(A, in_range=(v_min, v_max), out_range=(0, 1))

    map_vessel = np.flipud(np.rot90(J))

    # plt.imshow(map_vessel)
    # plt.show()

    # use a slab from BM -50 to choroid + 50 for choroid vessel segmentation / binarization
    a = 50
    yy2 = np.floor(thickness) + 250 + a
    z0 = 250 - a
    z1 = np.floor(min(550, yy2.max())).astype(int)
    cropSTR_EC1 = cropSTR_EC[z0:z1, :, :].astype(float)

    cropSTR_CVI = np.zeros((cropSTR_EC1.shape[0] - a, cropSTR_EC1.shape[1], cropSTR_EC1.shape[2]))

    # % % Adjascent three B - scans were averaged
    for i in range(2, cropSTR_EC1.shape[2] - 3):
        Avg = cropSTR_EC1[:, :, i - 2: i]
        Junk = np.squeeze(np.mean(Avg, 2)); #% % Adjascent three B - scans were averaged

        # % % choroid mask
        MaskJunk = np.zeros(Junk.shape)
        MaskJunk[np.isnan(Junk)] = 1

        for j in range(N_Ascan):
            k = np.floor((yy2[j, i]) - 249).astype(int)
            MaskJunk[k::, j] = 1

        in_temp = gaussian_filter(Junk, sigma=0.5)

        Junkk = np.zeros_like(in_temp)
        cv2.normalize(in_temp, Junkk, 0, 0.99999, cv2.NORM_MINMAX)  # Junkk = mat2gray(imgaussfilt(Junk));
        A = equalize_adapthist(Junkk, clip_limit=0.01)    #  A = adapthisteq(Junkk, 'clipLimit', 0.01, 'Distribution', 'rayleigh');
        v_min, v_max = np.percentile(A, (0.1, 95))
        Junkk = rescale_intensity(A, in_range=(v_min, v_max), out_range=(0, 1))

        thres = threshold_otsu(Junkk)*1.04
        Junkkk = Junkk > thres
        Junkkk = morphology.remove_small_objects(Junkkk, min_size=20, connectivity=8)
        BW = morphology.remove_small_objects(np.invert(Junkkk), min_size=20, connectivity=8)    #
        BW = (1 - BW).astype(float)

        BW[MaskJunk == 1] = np.NaN

        # % % ONH regions were excluded, maybe exclude ONH after the loop is
        #  another option
        mask1 = ONHmask[:, i]
        mask2 = np.tile(mask1[:, np.newaxis], (1, cropSTR_EC1.shape[0]))
        mask2 = mask2.T
        BW = BW * mask2
        cropSTR_CVI[:, :, i] = BW[a: cropSTR_EC1.shape[0], :]


    cropSTR_CVI[:,:, 0] = cropSTR_CVI[:, :, 2]
    cropSTR_CVI[:,:, 1] = cropSTR_CVI[:, :, 2]


    # % % cropSTR_CVI is the binarized choroid(0 is choroid vessel, 1 is
    # % % stroma; and non - choroid regions are NaN) the VVD2D may be not necessary, please check if you can
    # % % simplify
    VVD2D = np.zeros((2, N_Ascan, N_Frame))
    for i in range(N_Ascan):
        for j in range(N_Frame):
            VVD2D[0, i, j] = np.sum(cropSTR_CVI[:, i, j] == 1)
            VVD2D[1, i, j] = np.sum(cropSTR_CVI[:, i, j] == 0)


    cvi = np.nansum(VVD2D[1,:,:]) / np.nansum(VVD2D) # % % cvi is calculated as (# of 0s)/(# of 0s and 1s) in cropSTR_CVI

    temp = np.nansum(VVD2D, axis=0) # divided by zero
    temp[temp == 0] = np.nan
    b = VVD2D[1, :, :] / temp
    b = np.squeeze(b) # % % % % cvi enface map(b) is calculated as (  # of 0s)/(# of 0s and 1s) in cropSTR_CVI from enface
    b[np.isnan(b)] = 0

    map_cvi = rescale_intensity(np.flipud(np.rot90(b)), out_range=(0, 255))

    return cvi, mct, map_thickness, map_vessel, map_cvi, cropSTR_CVI

# plt.imshow(map_cvi)
# plt.show()
#
# plt.imshow(np.squeeze(VVD2D[0,:,:]))
# plt.show()


