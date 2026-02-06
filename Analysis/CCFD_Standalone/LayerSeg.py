# -*- coding: utf-8 -*-

from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, median_filter
import numpy as np
from skimage.filters import scharr
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import convolve

def segRPE_BM(img_slice, BM_last, retina_lower_offset=50, retina_upper_offset=150):
    '''Get the segmentation line location of ILM RPE, BM
    Args:
        img_slice
    Return:
        ILM, RPE, BM 
    '''
    #smooth the image
    img_slice = gaussian_filter(img_slice, (1,1))

    img_width = img_slice.shape[1]
    
    # get the rough location of RPE
    RPE_rough = np.argmax(img_slice, axis=0)
    RPE_rough = median_filter(RPE_rough, (40), mode='nearest')
    RPE_rough = gaussian_filter(RPE_rough, (20), mode='nearest')
 
    # set the mask for RPE finding
    slice_ori = np.copy(img_slice)
    for j in range(0, img_width):
        img_slice[0:RPE_rough[j]-10, j] = 0
        img_slice[RPE_rough[j]+10:, j] = 0
    
    # get the maxinum intensity location
    RPE = np.argmax(img_slice, axis=0)
    
    # smooth
    # RPE_refine = median_filter(RPE, (15), mode='nearest')
    RPE_refine = gaussian_filter(RPE, (0.5), mode='nearest')
    # RPE_refine  = RPE
    
    x = np.arange(0, img_width)
    y = RPE_refine
    points = np.vstack(([x],[y]) )
    points = points.T
    
    #%% convex hull for BM
    try:
        hull = ConvexHull(points)
        min_ind = np.argmin(hull.vertices)
        max_ind = np.argmax(hull.vertices)
        
        # find the maxinum decending loop sequence
        if max_ind< min_ind:
            inds = np.arange(max_ind, min_ind+1)
        else:
            inds  = np.hstack((np.arange(0, min_ind+1), np.arange(max_ind, len(hull.vertices)-1)))
        points_select = points[hull.vertices[inds], :]
        
        # sort by cloumn 0
        points_select = points_select[points_select[:, 0].argsort()]
        
        if points_select.shape[0]<2:
            print(points_select)
        else:
            try:
                f1 = interp1d(points_select[:,0], points_select[:,1], kind='linear', fill_value="extrapolate")
            except:
                try:
                    f1 = interp1d(points_select[:,0], points_select[:,1], kind='linear', fill_value="extrapolate")
                    # print('less than 4 points, use linear interpolation')
                except:
                    BM = BM_last
                    # print('cannot fit the line, copy previous ones')
    #            print(points_select)
    #            plt.imshow(img_slice)
    #            plt.plot(points_select[:,0], points_select[:,1], 'o',points_select[:,0], f1(points_select[:,0]),'--')
    #            plt.show()

            
#    plt.plot(points_select[:,0], points_select[:,1], 'o', points_select[:,0], f(points_select[:,0]),'--')
        BM = f1(x)
        
    except:
        BM = BM_last
        print('Convex hull algorithm fail, copy previous ones')
    
    #%% find the ILM    
    img_border = np.copy(slice_ori)
    img_border = median_filter(img_border, (5, 5))
    img_border = gaussian_filter(img_border, (3, 3))
    
    # scharr edge
    edge_scharr = scharr(img_border)
#    plt.imshow(edge_scharr[0:500,...])
    
    # find the maxinum response before RPE
    RPE_vague = median_filter(RPE_refine, 20,  mode='nearest')
    for j in range(0, img_width):
        # set the range of image
        offset = RPE_vague[j] - retina_lower_offset
        if offset < 0:
            offset = 0
        edge_scharr[offset:, j] = 0  # 40 pixel before RPE
        offset = RPE_vague[j] - retina_upper_offset
        if offset < 0:
            offset = 0
        edge_scharr[0:offset, j] = 0  # set [0: RPE-upperoffset] as 0, remove the backgound noise
    
    ILM = np.argmax(edge_scharr, axis=0)
    ILM_refine = median_filter(ILM, (15), mode='nearest')
    
    return ILM_refine, RPE_refine, BM


def smooth_img(img):
    img_med = median_filter(img, size=(2, 10))
    img_gau = gaussian_filter(img_med, sigma=(.5, 7))
    return img_gau

def la_edge(img):
    def_weight = np.array([[-3,-8,-10,-8,-3], [-1,-5,-2,-5,-1], [0,0,0,0,0], [1,5,2,5,1], [3,8,10,8,3]])/92
    result = convolve(img, def_weight)
    return result

def get_BM(slice_stru, slice_flow, img_width, ILM_refine, RPE_refine, RPE_fit):
    """ Use the OCTA image to guide the segmentaion for better BM
    :param slice_stru:
    :param slice_flow:
    :param img_width:
    :param ILM_refine:
    :param RPE_refine:
    :param RPE_fit:
    :return: BM fit
    """

    # enhance the image
    # img_cst = smooth_img(slice_stru) - 1.0*smooth_img(slice_flow)
    img_cst = slice_stru - 1.0*smooth_img(slice_flow)
    img_bi = cv2.bilateralFilter(img_cst.astype('float32'), sigmaColor=150, sigmaSpace=150, d=8)

    # get the edge image
    edge_img = la_edge(img_bi)
    #%%
    # BM_last = np.zeros((1, img_width))
    # ILM_refine, RPE_refine, RPE_fit = segRPE_BM1(slice_stru,  BM_last)

    # get the BM
    # select certain range
    thick_ref = np.abs(RPE_fit - RPE_refine)
    thick_ref = thick_ref.astype('int64')
    upper_ref = np.copy(thick_ref)
    upper_ref[upper_ref>20] = 20    # cut the upper boundry
    thick_ref[thick_ref>20] += 10   # add the lower bondary
    RPE_fit = RPE_fit.astype('int64')

    for j in range(0, img_width):
        edge_img[0: RPE_fit[j]-upper_ref[j], j] = -400
        edge_img[RPE_fit[j]+thick_ref[j]+10:, j] = -400

    BM_test = np.argmax(edge_img, axis=0)
    # BM_test_med = median_filter(BM_test, (25))  # smooth
    # lines = np.stack((ILM_refine, RPE_refine, RPE_fit, BM_test, BM_test_med), axis=0)
    # plot_img_with_lines(img_cst, lines)
    # plot_img_with_lines(edge_img, lines)

    # select points with high gradients
    new_line = np.empty((img_width))
    new_line[:] = np.nan
    edge_vals = np.copy(new_line)

    sel_points = []
    for i in range(0, img_width):
        edge_vals[i] = edge_img[BM_test[i], i]  # get the line at

    thre = np.percentile(edge_vals, 60) # the percentage of selection 60
    for i in range(0, img_width):

        if (edge_vals[i]>thre) or (i<6) or (i>img_width-5):
            new_line[i] = BM_test[i]
            sel_points.append([i, BM_test[i]])

    # lines = np.stack((ILM_refine, RPE_refine, RPE_fit, BM_test, BM_test_med, new_line), axis=0)
    # plot_img_with_lines(img_cst, lines)
    # plot_img_with_lines(edge_img, lines)

    # fitting the line
    x = np.arange(0, img_width)
    f2 = interp1d(np.asarray(sel_points)[:, 0], np.asarray(sel_points)[:, 1], kind='slinear', fill_value="extrapolate")

    BM_fit = f2(x)
    BM_fit = median_filter(BM_fit, (15), mode='nearest')
    BM_fit = gaussian_filter(BM_fit, (15), mode='nearest')

    # lines = np.stack((ILM_refine, RPE_refine,RPE_fit, BM_test, BM_test_med, new_line, BM_fit), axis=0)
    # plot_img_with_lines(img_cst, lines)
    return BM_fit


def segRPE_BM_skin(img_slice, retina_lower_offset=50, retina_upper_offset=150):
    '''Get the segmentation line location of ILM RPE, BM
    Args:
        img_slice
    Return:
        ILM, RPE, BM
    '''
    # smooth the image
    img_slice = gaussian_filter(img_slice, (1, 1))

    img_width = img_slice.shape[1]

    # get the rough location of RPE
    RPE_rough = np.argmax(img_slice, axis=0)
    RPE_rough = median_filter(RPE_rough, (40))
    RPE_rough = gaussian_filter(RPE_rough, (20))

    # set the mask for RPE finding
    slice_ori = np.copy(img_slice)
    for j in range(0, img_width):
        img_slice[0:RPE_rough[j] - 10, j] = 0
        img_slice[RPE_rough[j] + 10:, j] = 0

    # get the maxinum intensity location
    RPE = np.argmax(img_slice, axis=0)

    # smooth
    # RPE_refine = median_filter(RPE, (15))
    RPE_refine = gaussian_filter(RPE, (2), mode='nearest')
    # RPE_refine = RPE

    x = np.arange(0, img_width)
    y = RPE_refine
    points = np.vstack(([x], [y]))
    points = points.T


    # %% find the ILM
    img_border = np.copy(slice_ori)
    img_border = median_filter(img_border, (5, 5), mode='nearest')
    img_border = gaussian_filter(img_border, (3, 3), mode='nearest')

    # scharr edge
    edge_scharr = scharr(img_border)
    edge_scharr_cp = np.copy(edge_scharr)
    #    plt.imshow(edge_scharr[0:500,...])

    # find the maxinum response before RPE
    RPE_vague = median_filter(RPE_refine, 20, mode='nearest')

    for j in range(0, img_width):
        edge_scharr[RPE_vague[j] - retina_lower_offset:, j] = 0  # 40 pixel before RPE

    ILM = np.argmax(edge_scharr, axis=0)
    ILM_refine = median_filter(ILM, (15), mode='nearest')

    # BM for skin
    for j in range(0, img_width):
        edge_scharr_cp[0: RPE_vague[j], j] = 0
        edge_scharr_cp[RPE_vague[j] + retina_upper_offset::, j] = 0  # 40 pixel before RPE

    BM = np.argmax(edge_scharr_cp, axis=0)
    BM_refine = median_filter(BM, (15), mode='nearest')

    return ILM_refine, RPE_refine, BM_refine


def seg_video(img, img_flow=None, retina_lower_offset=50, retina_upper_offset=150, moving_avg=True, mode='eye', better_BM=False):
    '''Segment the images
    Args:
        img: 3d img cube
    Return:
        new_layers: layers with ILM, RPE, BM
    '''
    if mode == 'skin':
        better_BM = False

    [img_width, frame_num] = img.shape[1], img.shape[2]
    print('Segment the videwo with size: ', img.shape)
    print('Segmentation progress: ')
    if better_BM:
        layers = np.zeros((img_width, frame_num, 4))
    else:
        layers = np.zeros((img_width, frame_num, 3))

    BM_last = np.zeros((img_width))
    for i in tqdm(range(0, frame_num)):
    # for i in range(0, frame_num):
        # progress
        # if (i%50 == 0):
        #     print('framenum=', i, '  progress=', i/frame_num*100)  # progress
            
        #moving average to get the slice
        if moving_avg:
            if (i>0) and (i<frame_num-1):
                img_slice = np.mean(img[:,:, i-1:i+1], axis=2)
            else:
                img_slice = img[:, :, i]
        else:
            img_slice = img[:, :, i]

        if better_BM:   # get the slice of flow
            if moving_avg:
                if (i>0) and (i<frame_num-1):
                    slice_flow = np.mean(img_flow[:,:, i-1:i+1], axis=2)
                else:
                    slice_flow = img_flow[:, :, i]
            else:
                slice_flow = img_flow[:, :, i]

        if mode == 'eye':
            # print('auto segmentation use eye mode, better BM alogrithm:')
            ILM, RPE, RPE_fit = segRPE_BM(img_slice, BM_last, retina_lower_offset, retina_upper_offset)
            if better_BM:
                BM = get_BM(img_slice, slice_flow, img_width, ILM, RPE, RPE_fit)
                # print('get BM shape', BM.shape)

        elif mode == 'skin':
            # print('auto segmentation use skin mode:')
            ILM, RPE, RPE_fit = segRPE_BM_skin(img_slice, retina_lower_offset, retina_upper_offset)

        layers[:, i, 0] = ILM
        layers[:, i, 1] = RPE
        layers[:, i, 2] = RPE_fit
        if better_BM:
            layers[:, i, 3] = BM
        
        BM_last = np.copy(RPE_fit)
        
    # smooth
    new_layers = np.zeros(layers.shape)    
    for i in range(0, layers.shape[2]):
        new_layers[..., i] = median_filter(layers[:, :, i], (1, 15))
#        plt.imshow(new_layers[..., i])
#        plt.show()
     
    #topological check 1,2
    thick_map = new_layers[:,:,2] - new_layers[:,:,1]
    thick_map[thick_map>=0] = 0
    new_layers[:,:,2] = new_layers[:,:,2] - thick_map    
        
#    fn = 330         
#    plt.figure(figsize=(6,4))
#    plt.imshow(img[0:500,:, fn], cmap='gray')    
#    plt.plot(new_layers[:,fn, 0])
#    plt.plot(new_layers[:,fn, 1])
#    plt.plot(new_layers[:,fn, 2])
    
    return new_layers





#%%
#img_sort = np.sort(img_obj.stru3d, axis=0)[::-1]

#img = img_obj.stru3d
#
#[img_width, frame_num] = img.shape[1], img.shape[2]
#
#layers = np.zeros((img_width, frame_num, 3))
#for i in range(100 , 101):
#    print(i)
#    #moving average
#    if (i>0) and (i<frame_num-1):
#        img_slice = np.mean(img[:,:, i-1:i+1], axis=2)
#    else:
#        img_slice = img[:, :, i];
##    RPE, BM = segRPE_BM(img_slice)
#
#    #%% get the rought location, constrain the area
#    RPE_rough = np.argmax(img_slice, axis=0)
#    RPE_rough = median_filter(RPE_rough, (40))
#    RPE_rough = gaussian_filter(RPE_rough, (20))
# 
#    # set the mask for RPE finding
#    slice_ori = np.copy(img_slice)
#    for j in range(0, img_width):
#        img_slice[0:RPE_rough[j]-50, j] = 0
#        img_slice[RPE_rough[j]+30:, j] = 0
#    
#    # get the maxinum intensity location
#    RPE = np.argmax(img_slice, axis=0)
#    
#    # smooth
#    RPE_refine = median_filter(RPE, (20))
#    RPE_refine = gaussian_filter(RPE_refine, (2))
#    
#    x = np.arange(0, img_width)
#    y = RPE_refine
#    points = np.vstack(([x],[y]) )
#    points = points.T
#    
#    #%% convex hull for BM
#    hull = ConvexHull(points)
#    min_ind = np.argmin(hull.vertices)
#    max_ind = np.argmax(hull.vertices)
#    
#    if max_ind< min_ind:
#        inds = np.arange(max_ind, min_ind+1)
#    else:
#        inds  = np.hstack((np.arange(0, min_ind+1), np.arange(max_ind, len(hull.vertices)-1)))
#    points_select = points[hull.vertices[inds], :]
#    
#    # sort by cloumn 0
#    points_select = points_select[points_select[:, 0].argsort()]
#    
#    f1 = interp1d(points_select[:,0], points_select[:,1], kind='cubic', fill_value="extrapolate")
##    plt.plot(points_select[:,0], points_select[:,1], 'o', points_select[:,0], f(points_select[:,0]),'--')
#    BM = f1(x)
#    
#    #%% find the ILM    
#    img_border = np.copy(img_slice)
#    img_border = median_filter(img_border, (5,5))
#    img_border = gaussian_filter(img_border, (3,3))
#    
#    edge_scharr = scharr(img_border)
##    plt.imshow(edge_scharr[0:500,...])
#    
#    RPE_vague = median_filter(RPE_refine, 50)
#    for j in range(0, img_width):
#        edge_scharr[RPE_vague[j] - 20:, j] = 0  # 20 pixel befoe RPE
#    
#    ILM = np.argmax(edge_scharr, axis=0)
#    ILM_refine = median_filter(ILM, (15))
#    
#    #%%
#    layers[:, i, 0] = ILM_refine
#    layers[:, i, 1] = RPE_refine
#    layers[:, i, 2] = BM
##    # convex hull to fit the BM
##       
#    plt.figure(figsize=(20,10))
#    plt.imshow(img_slice[0:500,...], cmap='gray')
#    plt.plot(RPE)
#    plt.plot(RPE_refine)
#    plt.plot(points_select[:,0], points_select[:,1], 'o', points_select[:,0], f1(points_select[:,0]),'--')
#    plt.show()
#    
##    plt.figure(figsize=(10,5))
##    plt.imshow(img_slice[0:500,...], cmap='gray')
##    plt.plot(RPE_refine)
##    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1])
##    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o')
###    plt.plot(hull.simplices[:,1],hull.simplices[:,0], 'o')
##    plt.show()
##    
##from skimage import feature
##from skimage.filters import roberts
#from skimage.filters import threshold_otsu, sobel, scharr
##from skimage.restoration import denoise_bilateral
#    
#img_border = np.copy(slice_ori)
##img_border = gaussian_filter(img_border, (5,5))
#img_border = median_filter(img_border, (5,5))
#img_border = gaussian_filter(img_border, (3,3))
##img_border = denoise_bilateral(img_border, sigma_spatial=15,multichannel=False)
##for j in range(0, img_width):
##    img_border[RPE_rough[j]-40:,j] = 0
#    
##thresh = threshold_otsu(img_border)
##img_border[img_border < (thresh*0.5)] = 0
##edges = feature.canny(img_border, sigma=5)
#
##edge_roberts = roberts(img_border)
##plt.imshow(edge_roberts[0:500,...])
#
##edge_sobel = sobel(img_border)
##plt.imshow(edge_sobel[0:500,...])
#
#edge_scharr = scharr(img_border)
#plt.imshow(edge_scharr[0:500,...])
#
#for j in range(0, img_width):
#    edge_scharr[RPE_rough[j] - 40:, j] = 0
#
#ILM = np.argmax(edge_scharr, axis=0)
#ILM_refine = median_filter(ILM, (15))
#plt.imshow(slice_ori[0:500,...], cmap='gray')    
#plt.plot(ILM_refine)
#plt.show()
#
#plt.imshow(img_border[0:500,...])
#plt.imshow(slice_ori[0:500,...])
#plt.imshow(edge_scharr[0:500,...])
#plt.show()
##%%
#
#
#
