
from imageio import imwrite
import matplotlib.pyplot as plt
from scipy import integrate
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy import integrate
from skimage.filters import scharr
from skimage.exposure import equalize_adapthist
import cv2
from scipy.ndimage import convolve

import time
from joblib import Parallel, delayed
import multiprocessing
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.measure import label

import Analysis.SM.opl_utilities as oplut

num_cores = multiprocessing.cpu_count()-1


def get_layers_parallel(i, OCT_S, OCT_F, rpe, paras):
    """
    Parallel processing for the segmenation
    Args:
        i: frame number,
        OCT_S: OCT mean struct, (depth, width, frameNum)
        OCT_F: OCT flow
        OAC_S: OAC struct
        OAC_F: OAC flow
        RPE_rough: the loc of rpe
        paras: oct segmenation paras
    Returns:

    """
    rpe_ori = np.squeeze(rpe[:, i])
    rpe = gaussian_filter(rpe, (5, 1), mode='nearest')
    rpe = median_filter(rpe, (10, 2), mode='nearest')
    ub = paras['upper_bound']
    lb = paras['lower_bound']
    # img_slice = gaussian_filter(OAC_S[ub:lb, :, i], (1, 1)) - gaussian_filter(OAC_F[ub:lb, :, i], (2, 2))
    img_width = rpe.shape[1]
    # RPE_rough = np.argmax(gaussian_filter(img_slice, (1, 1)), axis=0)
    # RPE_rough = gaussian_filter(RPE_rough, (1))
    # RPE_rough = get_RPE(img_slice)
    RPE_rough = np.squeeze(rpe[:, i])
    RPE_fit = get_RPE_fit(RPE_rough)
    ILM = get_ILM(OCT_S[ub:lb, :, i], RPE_rough, paras)
    slice_stru = OCT_S[ub:lb, :, i]
    slice_flow = OCT_F[ub:lb, :, i]
    BM, BM_test = get_BM(slice_stru, slice_flow,
                         img_width, None, RPE_rough, RPE_fit)

    layers_slice = np.zeros((img_width, 4))
    layers_slice[:, 0] = ILM
    layers_slice[:, 1] = rpe_ori
    layers_slice[:, 2] = RPE_fit
    layers_slice[:, 3] = BM
    # layers_slice[:, 4] = BM_test

    return layers_slice


def get_RPE_parallel(i, OAC_S, OAC_F, paras):
    """
    Get the RPE layer only with the parallel loop
    Returns:
    """
    ub = paras['upper_bound']
    lb = paras['lower_bound']
    img_slice = gaussian_filter(
        OAC_S[ub:lb, :, i], (1, 1)) - gaussian_filter(OAC_F[ub:lb, :, i], (2, 2))
    img_width = img_slice.shape[1]
    # RPE_rough = np.argmax(gaussian_filter(img_slice, (1, 1)), axis=0)
    # RPE_rough = gaussian_filter(RPE_rough, (1))
    RPE_rough = get_RPE(img_slice)

    return RPE_rough


def get_RPE(img_slice):
    # img_slice = gaussian_filter(img_slice, (1, 1))

    img_width = img_slice.shape[1]

    # get the rough location of RPE
    RPE_rough = np.argmax(img_slice, axis=0)
    RPE_rough = median_filter(RPE_rough, (40), mode='nearest')
    RPE_rough = gaussian_filter(RPE_rough, (20), mode='nearest')

    # set the mask for RPE finding
    slice_ori = np.copy(img_slice)
    for j in range(0, img_width):
        # boundary check
        low_range = RPE_rough[j] - 30
        if low_range < 0:
            low_range = 0

        up_range = RPE_rough[j] + 20
        if up_range >= img_slice.shape[0]:
            up_range = img_slice.shape[0] - 1

        img_slice[0:low_range, j] = 0
        img_slice[up_range:, j] = 0

    # get the maxinum intensity location
    RPE = np.argmax(img_slice, axis=0)

    # smooth
    # RPE_refine = median_filter(RPE, (15))
    RPE_refine = gaussian_filter(RPE, (2))

    return RPE_refine


def get_RPE_fit(RPE_refine):
    """

    Args:
        RPE_refine:

    Returns:

    """
    img_width = RPE_refine.shape[0]
    x = np.arange(0, img_width)
    y = RPE_refine
    points = np.vstack(([x], [y]))
    points = points.T

    BM_last = median_filter(RPE_refine, 10, mode='nearest')

    # %% convex hull for BM
    try:
        hull = ConvexHull(points)
        min_ind = np.argmin(hull.vertices)
        max_ind = np.argmax(hull.vertices)

        # find the maxinum decending loop sequence
        if max_ind < min_ind:
            inds = np.arange(max_ind, min_ind + 1)
        else:
            inds = np.hstack((np.arange(0, min_ind + 1),
                              np.arange(max_ind, len(hull.vertices) - 1)))
        points_select = points[hull.vertices[inds], :]

        # sort by cloumn 0
        points_select = points_select[points_select[:, 0].argsort()]

        if points_select.shape[0] < 2:
            print(points_select)
        else:
            try:
                f1 = interp1d(
                    points_select[:, 0], points_select[:, 1], kind='slinear', fill_value="extrapolate")
            except:
                try:
                    f1 = interp1d(
                        points_select[:, 0], points_select[:, 1], kind='slinear', fill_value="extrapolate")
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
    return BM


def get_ILM(oct_slice, RPE_refine, paras):
    """
    Get the ILM layer before the RPE
    Args:
        oct_slice: oct slice of the
        RPE_refine:
        paras:

    Returns:

    """
    # find the ILM
    img_width = RPE_refine.shape[0]
    img_border = median_filter(oct_slice, (5, 5))
    img_border = gaussian_filter(img_border, (3, 3))

    # scharr edge
    edge_scharr = scharr(img_border)
    #    plt.imshow(edge_scharr[0:500,...])

    # find the maxinum response before RPE
    RPE_vague = median_filter(RPE_refine, 20)
    for j in range(0, img_width):
        # set the range of image
        offset = RPE_vague[j] - paras['retina_lower_offset']
        if offset < 0:
            offset = 0
        edge_scharr[offset:, j] = 0  # 40 pixel before RPE
        offset = RPE_vague[j] - paras['retina_upper_offset']
        if offset < 0:
            offset = 0
        # set [0: RPE-upperoffset] as 0, remove the backgound noise
        edge_scharr[0:offset, j] = 0

    ILM = np.argmax(edge_scharr, axis=0)
    ILM_refine = median_filter(ILM, (15))

    return ILM_refine


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
    img_cst = gaussian_filter(slice_stru, (3, 3)) - 1.3*smooth_img(slice_flow)
    img_bi = cv2.bilateralFilter(img_cst.astype(
        'float32'), sigmaColor=150, sigmaSpace=150, d=8)

    # get the edge image
    edge_img = la_edge(img_bi)
    # edge_img = scharr(img_bi)
    # %%
    # BM_last = np.zeros((1, img_width))
    # ILM_refine, RPE_refine, RPE_fit = segRPE_BM1(slice_stru,  BM_last)

    # get the BM
    # select certain range
    thick_ref = np.abs(RPE_fit - RPE_refine)
    thick_ref = thick_ref.astype('int64')
    upper_ref = np.copy(thick_ref)
    upper_ref[upper_ref > 20] = 20    # cut the upper boundry
    thick_ref[thick_ref > 35] = 35   # add the lower bondary in drusen region
    RPE_fit = RPE_fit.astype('int64')

    for j in range(0, img_width):
        edge_img[0: RPE_fit[j]-upper_ref[j], j] = -4000
        edge_img[RPE_fit[j]+thick_ref[j]+50:, j] = - \
            4000     # default low bound range

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

    thre = np.percentile(edge_vals, 10)  # the percentage of selection 60
    for i in range(0, img_width):

        if (edge_vals[i] > thre) or (i < 6) or (i > img_width-5):
            new_line[i] = BM_test[i]
            sel_points.append([i, BM_test[i]])

    # lines = np.stack((ILM_refine, RPE_refine, RPE_fit, BM_test, BM_test_med, new_line), axis=0)
    # plot_img_with_lines(img_cst, lines)
    # plot_img_with_lines(edge_img, lines)

    # fitting the line
    x = np.arange(0, img_width)
    f2 = interp1d(np.asarray(sel_points)[:, 0], np.asarray(sel_points)[
                  :, 1], kind='slinear', fill_value="extrapolate")

    BM_fit = f2(x)
    BM_fit = median_filter(BM_fit, (10), mode='nearest')
    BM_fit = gaussian_filter(BM_fit, (5), mode='nearest')

    # lines = np.stack((ILM_refine, RPE_refine,RPE_fit, BM_test, BM_test_med, new_line, BM_fit), axis=0)
    # plot_img_with_lines(img_cst, lines)
    # BM_test = gaussian_filter(BM_test, (2))

    return BM_fit, BM_test


def smooth_img(img):
    img_med = median_filter(img, size=(2, 10))
    img_gau = gaussian_filter(img_med, sigma=(.5, 7))
    return img_gau


def la_edge(img):
    def_weight = np.array([[-3, -8, -10, -8, -3], [-1, -5, -2, -5, -1],
                           [0, 0, 0, 0, 0], [1, 5, 2, 5, 1], [3, 8, 10, 8, 3]])/92
    result = convolve(img, def_weight)
    return result


def mean_volume_parallel(i, img, N):
    """
    Parallel function for the sliding window averaging
    Args:
        i: frame index
        img: 3d image volume
        N: window size
    Returns:

    """

    if i >= N+1 and i <= img.shape[2]-N-1:
        slice = np.mean(img[:, :, i-N-1:i+N], axis=2)
    else:
        slice = img[:, :, i]

    return slice


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
        JLin = (OCTL[:-1, :, :] / (M * 2 * (3 / img.shape[0]))) # pixel_size_z = 3mm/pixel_num_Z
    except Exception as e:
        print(e)


    return JLin


def smooth_layers(layers):
    """ smooth all layers
    """
    for i in range(layers.shape[2]):
        layers[:, :, i] = median_filter(
            layers[:, :, i], (2, 10), mode='nearest')
        layers[:, :, i] = gaussian_filter(
            layers[:, :, i], (1, 3), mode='nearest')

    return layers


def find_opl_parallel(i, frame_size, ILM, RPE, center, stru3d, oac, threshold, great_fovea_region, buffer_frame):
    rpe_offset = np.zeros((ILM.shape[0]))
    y_pos = np.zeros((ILM.shape[0]))
    upper = np.zeros((ILM.shape[0]))
    y_pos_up = np.zeros((ILM.shape[0]))
    opl_lowerBD = np.zeros((ILM.shape[0]))

    # int_distribution, med_or_mean, this_std = oplut.find_distribution_frame(
    #     i, stru3d, ILM, RPE)
    # threshold = oplut.get_int_threshold(
    #     int_distribution, med_or_mean, this_std)
    dist_threshold_lowbd, dist_threshold_highbd = oplut.short_dis_threshold_frame(
        i, ILM, RPE)

    img_slice = oplut.get_moving_slice(i, frame_size, stru3d)

    mean_RPE = oplut.get_RPE_fitpoly(i, RPE)

    # find fovea coordinates if in fovea frame
    fovea_index = None
    if i in great_fovea_region:
        fovea_index = oplut.local_fovea(i, ILM, RPE)
        fovea_center = oplut.find_fovea_center(fovea_index, i, ILM, RPE)
        fovea_index = oplut.rearange_fovea_x(fovea_index, fovea_center)
        buffer_x, great_fovea_x = oplut.find_buffer_x(fovea_index, i, ILM, RPE)
    else:
        fovea_index, great_fovea_x = None, None

    for num in range(img_slice.shape[1]):
        check_lowdist_threshold = (
            RPE[num, i] - ILM[num, i]) < dist_threshold_lowbd
        check_highdist_threshold = (
            RPE[num, i] - ILM[num, i]) > dist_threshold_highbd

        if fovea_index and num in fovea_index:
            rang_f = fovea_index[-1] - fovea_index[0]
            is_buffer_frame = i in buffer_frame
            if num in range(int(fovea_index[0] + rang_f/5), int(fovea_index[0] + rang_f*4/5 + 1)):
                ilm_dis, rpe_dis = oplut.get_distance(
                    mean_RPE, RPE, ILM, i, num, threshold, img_slice, fovea=True, frame_buffer=is_buffer_frame)
            elif num in buffer_x:
                ilm_dis, rpe_dis = oplut.get_distance(
                    mean_RPE, RPE, ILM, i, num, threshold, img_slice, buffer=True, frame_buffer=is_buffer_frame)
            else:
                ilm_dis, rpe_dis = oplut.get_distance(
                    mean_RPE, RPE, ILM, i, num, threshold, img_slice, edge=True, frame_buffer=is_buffer_frame)
            rpe_offset[num] = rpe_dis

            rand = img_slice[ILM[num, i].astype(
                int):RPE[num, i].astype(int)-rpe_dis, num]
            peaks, _ = find_peaks(rand, height=threshold)

            y_pos[num] = RPE[num, i].astype(int)-rpe_dis
            upper[num] = ILM[num, i].astype(int)
            if peaks.shape[0] > 0:
                y_pos_up[num] = peaks[-1]+ILM[num, i]
            else:
                y_pos_up[num] = np.nan
        else:
            ilm_dis, rpe_dis = oplut.get_distance(
                mean_RPE, RPE, ILM, i, num, threshold, img_slice, short=check_lowdist_threshold, long_=check_highdist_threshold)
            rpe_offset[num] = rpe_dis

            # Find lower boundary of OPL, find positive peak, choose last
            rand = img_slice[ILM[num, i].astype(
                int)+ilm_dis:RPE[num, i].astype(int)-rpe_dis, num]
            peaks, _ = find_peaks(rand, height=threshold)

            if peaks.shape[0] > 0:
                y_pos[num] = peaks[-1]+ilm_dis+ILM[num, i]
                y_pos_up[num] = peaks[-1]+ilm_dis+ILM[num, i]
                upper[num] = ILM[num, i].astype(int)+ilm_dis

                # refine opl by find negative peak, choose last
                y = oplut.peak_refine(img_slice,
                                      num,
                                      y_pos[num],
                                      upper[num])
                y_pos_up[num] = y + upper[num]

                # clean up points near ILM
                if abs(y_pos_up[num] - ILM[num, i]) <= 0.15*(RPE[num, i]-ILM[num, i]):
                    y_pos_up[num] = np.nan
            else:
                y_pos[num] = np.nan
                y_pos_up[num] = np.nan
                upper[num] = np.nan

    # clean up the outlier, fill in long Nan and empty edges
    y_pos_up[:] = oplut.clean_local_outliers(y_pos_up)

    # extract opl lower bound
    for num in range(img_slice.shape[1]):
        if fovea_index and num in fovea_index:
            lowerBD_temp = y_pos_up[num]
        else:
            lowerBD_temp = y_pos[num]

        if ~np.isnan(y_pos_up[num]) and ~np.isnan(y_pos[num]):
            y = oplut.peak_refine(img_slice,
                                  num,
                                  RPE[num, i],
                                  lowerBD_temp,
                                  lowerBD=True)
            opl_lowerBD[num] = y + lowerBD_temp
        else:
            opl_lowerBD[num] = np.nan

    opl_lowerBD[:] = oplut.clean_local_outliers(opl_lowerBD)
    y_pos_up[:] = oplut.find_fill_nan(y_pos_up, i, ILM[:, i], RPE[:, i], ILM)
    opl_lowerBD[:] = oplut.find_fill_nan(
        opl_lowerBD, i, y_pos_up, RPE[:, i], ILM)

    return y_pos_up, opl_lowerBD, fovea_index


def seg_video_parallel(img_obj, paras):

    [img_depth, img_width, img_framenum] = img_obj.stru3d.shape

    # moving average
    t = time.time()
    mean_img_s = Parallel(n_jobs=num_cores, prefer='threads')(
        delayed(mean_volume_parallel)(i, img_obj.stru3d, 1) for i in range(img_framenum))
    mean_img_s = np.array(mean_img_s)
    # gathered image dim (parrelFrame, depth, X)
    mean_img_s = np.rollaxis(mean_img_s, 0, 3)

    mean_img_f = Parallel(n_jobs=num_cores, prefer='threads')(
        delayed(mean_volume_parallel)(i, img_obj.flow3d, 1) for i in range(img_framenum))
    mean_img_f = np.array(mean_img_f)
    mean_img_f = np.rollaxis(mean_img_f, 0, 3)

    print('averaging done')
    #print(time.time() - t)
    # get OAC, slow
    OAC_S = OAC_calculation(mean_img_s)  # OAC struct
    OAC_F = OAC_calculation(mean_img_f)  # OAC flow
    OAC_F[np.isnan(OAC_F)] = 0
    print('OAC done')

    # set the segmentation parameters for the algorithm
    # paras = {'retina_lower_offset': 50, 'retina_upper_offset': 200, 'upper_bound': 250, 'lower_bound': 850}

    # run the rpe segmentation for fast and slow scans
    rpe_fast = Parallel(n_jobs=num_cores, prefer='threads')(
        delayed(get_RPE_parallel)(i, OAC_S, OAC_F, paras) for i in range(img_framenum))
    rpe_slow = Parallel(n_jobs=num_cores, prefer='threads')(
        delayed(get_RPE_parallel)(i, np.swapaxes(OAC_S, 1, 2), np.swapaxes(OAC_F, 1, 2), paras) for i in
        range(img_width))
    rpe_fast = np.array(rpe_fast)
    rpe_fast = rpe_fast.T
    rpe_slow = np.array(rpe_slow)
    # rpe_slow = np.swapaxes(rpe_slow, 0, 1)  # swap axis

    # clean the rpe segmentation

    seg_diff = rpe_fast - rpe_slow
    mask_fast_error = np.zeros_like(seg_diff)
    mask_fast_error[seg_diff < -50] = 1  # diff between two segs

    # plt.imshow(seg_diff, cmap='seismic')
    # plt.colorbar()
    # plt.show()
    # plt.imshow(mask_fast_error, cmap='jet')
    # plt.show()

    new_rpe = np.copy(rpe_fast)
    # replace the error of fast scan by slow scans
    new_rpe[mask_fast_error == 1] = rpe_slow[mask_fast_error == 1]
    new_rpe = gaussian_filter(new_rpe, (1, 3), mode='nearest')

    print('rpe done')
    # plt.imshow(new_rpe, cmap='jet')
    # plt.show()

    # run the segmentation for rest layers
    layers = Parallel(n_jobs=num_cores, prefer='threads')(
        delayed(get_layers_parallel)(i, mean_img_s, mean_img_f, new_rpe, paras) for i in range(img_framenum))
    layers = np.array(layers)
    layers = np.swapaxes(layers, 0, 1)
    layers = layers + paras['upper_bound']

    # run opl segmentation for all frames
    ILM = layers[:, :, 0]
    RPE = layers[:, :, 1]
    center, radius = oplut.find_fovea_center_frame(ILM, RPE)
    # print('Fovea center= ', center, ', radius=', radius)
    fovea_frame = range(center-radius, center+radius+1)
    great_fovea_region = oplut.refine_fovea_frame(
        fovea_frame, center, img_obj.stru3d)
    buffer_frame = oplut.get_buffer_frame(center, great_fovea_region)

    int_distribution, med_or_mean, this_std = oplut.find_distribution(
        img_obj.stru3d, ILM, RPE)
    threshold = oplut.get_int_threshold(
        int_distribution, med_or_mean, this_std)
    opl, opl_lowerBD, _ = zip(*Parallel(n_jobs=num_cores, prefer='threads')(
        delayed(find_opl_parallel)(i, img_framenum, ILM, RPE, center, img_obj.stru3d, OAC_S, threshold, great_fovea_region, buffer_frame) for i in range(img_framenum)
    ))
    opl = np.array(opl)
    opl_lowerBD = np.array(opl_lowerBD)
    opl = opl.T
    opl_lowerBD = opl_lowerBD.T

    opl = oplut.interp_fill(opl, ILM, RPE)
    opl_lowerBD = oplut.interp_fill(opl_lowerBD, ILM, RPE)
    print("OPL done")

    # stack opl and its lower boundary to layers
    layers = np.dstack((layers, opl))
    layers = np.dstack((layers, opl_lowerBD))

    # topological check 1,2
    thick_map = layers[:, :, 2] - layers[:, :, 1]
    thick_map[thick_map >= 0] = 0
    layers[:, :, 2] = layers[:, :, 2] - thick_map
    print(time.time() - t)

    layers = smooth_layers(layers)
    return layers


def seg_video_SD_parallel(img_obj, retina_th=50):
    """
    Zeiss SD OCT segmeantion, structural only

    Args:
        img_obj:
        paras:

    Returns:
    """
    [img_depth, img_width, img_framenum] = img_obj.stru3d.shape
    mean_img_s = Parallel(n_jobs=num_cores, prefer='threads')(
        delayed(mean_volume_parallel)(i, img_obj.stru3d, 1) for i in range(img_framenum))
    mean_img_s = np.array(mean_img_s)
    # gathered image dim (parrelFrame, depth, X)
    mean_img_s = np.rollaxis(mean_img_s, 0, 3)

    ILM = Parallel(n_jobs=num_cores, prefer='threads')(
        delayed(get_ILM_thre_parallel)(i, mean_img_s) for i in range(img_framenum))
    ILM = np.array(ILM)
    # gathered image dim (parrelFrame, depth, X)
    ILM = ILM.T

    ILM_medf = median_filter(ILM, (20, 20), mode='nearest')
    ILM_m = median_filter(ILM, (5, 5), mode='nearest')
    # plt.imshow(ILM_medf)

    select_img, gau_shift = select_slab(
        mean_img_s, ILM_medf, shift_down=retina_th, slab_thick=200, gau_height=int(retina_th*1.2))

    RPE = Parallel(n_jobs=num_cores, prefer='threads')(
        delayed(get_RPE_parallel_SD)(i, select_img) for i in range(img_framenum))
    RPE = np.array(RPE)
    RPE = RPE.T

    RPE_adjust = RPE + ILM_medf + retina_th + gau_shift    #

    layers = np.copy(ILM_m)
    layers = np.dstack((layers, RPE_adjust))

    # RPE_fit
    RPE_fit = Parallel(n_jobs=num_cores, prefer='threads')(
        delayed(get_RPE_fit_parallel)(i, RPE_adjust) for i in range(img_framenum))
    RPE_fit = np.array(RPE_fit)
    RPE_fit = RPE_fit.T

    layers = np.dstack((layers, RPE_fit))
    return layers


def get_RPE_fit_parallel(i, rpe):
    RPE_fit = np.squeeze(get_RPE_fit(rpe[:, i]))
    return RPE_fit


def get_RPE_parallel_SD(i, select_img):
    img_slice = gaussian_filter(select_img[:, :, i], (1, 0.5))
    rpe = get_RPE(img_slice)
    return rpe


def get_ILM_thre_parallel(i, OCT_S, paras=None):
    slice = OCT_S[:, :, i]

    img_blur = gaussian_filter(slice, (3, 3))
    img_blur = gaussian_filter(img_blur, (5, 5))
    # img_blur = equalize_adapthist(img_blur.astype(np.uint8), kernel_size=None, clip_limit=0.01)
    try:
        thresh = threshold_otsu(img_blur)*0.8
    except Exception as e:
        print(e)
        thresh = 0
    binary = img_blur > thresh
    binary_clean = remove_small_objects(binary, 2000)
    binary_clean = getLargestCC(binary_clean)
    loc = gaussian_filter(np.argmax(binary_clean, axis=0), (5))

    return loc


def getLargestCC(segmentation):
    labels = label(segmentation)
    # assert(labels.max() != 0) # assume at least 1 CC
    try:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        if labels.max()>2:
            largestCC2 = labels == np.argsort(np.bincount(labels.flat)[1:], axis=0)[-2]+1   # second-largest component
            largestCC = largestCC + largestCC2
    except Exception as e:
        print('Warning: Image has no connected component, ', e)
        largestCC = labels
    return largestCC


def select_slab(OCT_S, ILM, shift_down=70, slab_thick=200, gau_height=40):
    """
    Get the RPE slab from the OCT volume data,

    Args:
        OCT_S:
        shift_down:
        slab_thick:

    Returns:
    The slab in OCT_S, from ILM to shiftdwon
    """
    # shift_down = 70    # pixel from ILM
    # slab_thick = 200
    nz, nx, ny = OCT_S.shape
    layer_start = ILM + shift_down
    # mask the ILM for RPE
    O, P, Q = np.meshgrid(np.arange(slab_thick), np.arange(
        nx), np.arange(ny), sparse=False, indexing='ij')

    O = O + layer_start[np.newaxis, ...]

    # add a gaussian layer to fit the shape/depth of RNFL
    pad = get_2d_gaussain(sizeX=nx, sizeY=ny, height=gau_height)
    O = O + pad

    ind_outof_bottom = O > nz - 1
    ind_outof_top = O < 0

    # Omask = np.zeros(O.shape)
    # Omask[O - layer_end > 0] = 1  # mask pixels outside layer_end
    # Omask[ind_outof_bottom] = 1  # mask pixels outside bottom of image
    # Omask[ind_outof_top] = 1  # mask pixels outside top of image

    O[ind_outof_bottom] = 0
    O[ind_outof_top] = 0
    select_img = OCT_S[O, P, Q]

    return select_img, pad


def get_2d_gaussain(sizeX=512, sizeY=128, height=30):
    x, y = np.meshgrid(np.linspace(-1, 1, sizeY), np.linspace(-1, 1, sizeX))
    dst = np.sqrt(x * x + y * y)

    # Initializing sigma and muu
    sigma = 0.3
    muu = 0.000

    # Calculating Gaussian array
    gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2)))
    result = (1 - gauss)*height
    # plt.imshow(result)
    # plt.colorbar()
    # plt.show()

    # plt.figure()
    # plt.plot(result[250, :])
    # plt.show()
    return result.astype(int)
