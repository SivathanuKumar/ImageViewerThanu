# This is the utility functions for computing OPL
import numpy as np
from scipy.signal import find_peaks, peak_widths
from itertools import tee
from scipy.stats import skewtest
from PIL import Image
import imagehash
import cv2
from scipy.ndimage import gaussian_filter

# Function with important params


def get_distance(mean_RPE, RPE, ILM, frame_num, x_posit, threshold, img_slice, short=False, long_=False, fovea=False, edge=False, buffer=False, frame_buffer=False):
    total = RPE[x_posit, frame_num] - ILM[x_posit, frame_num]

    # normal case
    ilm_mult = 0.30
    rpe_mult = 0.3

    if short:
        ilm_mult = 0.33   # 0 -> 0.35
        rpe_mult = 0.35   # 0.5 -> 0.4

    if long_:
        ilm_mult = 0.15   # 0 -> 0.35
        rpe_mult = 0.28

    if not frame_buffer:
        if buffer:
            ilm_mult, rpe_mult = 0.35, 0.5
        elif edge:
            ilm_mult,  rpe_mult = 0, 0.60
        elif fovea:
            ilm_mult,  rpe_mult = 0, 0.75
    else:
        if buffer:
            ilm_mult, rpe_mult = 0.35, 0.40  # 0.4 before
        elif edge:
            ilm_mult,  rpe_mult = 0, 0.50  # 0.5 before
        elif fovea:
            ilm_mult,  rpe_mult = 0, 0.60

    if x_posit in range(RPE.shape[0]//25+1) or x_posit in range(RPE.shape[0]-RPE.shape[0]//25, RPE.shape[0]):
        ilm_mult = 0.15
        rpe_mult = 0.35

    to_ilm = total * ilm_mult
    to_rpe = total * rpe_mult

    if RPE[x_posit, frame_num] - to_rpe*0.2 > mean_RPE[x_posit]:
        to_rpe = to_rpe + RPE[x_posit, frame_num] - \
            to_rpe*0.2 - mean_RPE[x_posit]
        to_ilm = to_ilm - (RPE[x_posit, frame_num] - to_rpe*0.2 - mean_RPE[x_posit]) if to_ilm - (
            RPE[x_posit, frame_num] - to_rpe*0.2 - mean_RPE[x_posit]) > 0 else 0
        # while np.nanmean(img_slice[int(RPE[x_posit, frame_num]-to_rpe):int(RPE[x_posit, frame_num])+1]) >= threshold*1.5 and RPE[x_posit, frame_num] - to_rpe > ILM[x_posit, frame_num] + 5:
        #     to_rpe += 10

    if x_posit in range(RPE.shape[0]//10+1) or x_posit in range(RPE.shape[0]-RPE.shape[0]//10, RPE.shape[0]):
        if RPE[x_posit, frame_num] - mean_RPE[x_posit] >= 60:
            to_ilm = total * ilm_mult
            to_rpe = total * rpe_mult
        elif RPE[x_posit, frame_num] - mean_RPE[x_posit] >= 20 and RPE[x_posit, frame_num] - to_rpe*0.5 > mean_RPE[x_posit]:
            to_rpe = to_rpe + RPE[x_posit, frame_num] - \
                to_rpe*0.5 - mean_RPE[x_posit]
            to_ilm = to_ilm - (RPE[x_posit, frame_num] - to_rpe*0.5 - mean_RPE[x_posit]) if to_ilm - (
                RPE[x_posit, frame_num] - to_rpe*0.5 - mean_RPE[x_posit]) > 0 else 0

    return int(to_ilm), int(to_rpe)


# OPL find negative peaks
def peak_refine(img_slice, num, lower_bound, upper_bound, lowerBD=False):
    if ~np.isnan(lower_bound):
        # if not lowerBD:
        rand = img_slice[int(upper_bound):int(lower_bound)-5, num]
        # else:
        #     rand = img_slice[int(upper_bound):int(lower_bound)-5, num]

        if rand.shape[0] <= 0:
            return np.nan

        ave = np.mean(-rand)
        if not lowerBD:
            peaks, _ = find_peaks(-rand, height=(ave +
                                                 1.5*np.std(-rand), ave+2.5*np.std(-rand)))
        else:
            peaks, _ = find_peaks(-rand, height=(ave, ave+2.5*np.std(-rand)))

        if peaks.shape[0] > 0:
            if lowerBD:
                return peaks[0]
            else:
                return peaks[-1]
        else:
            return np.nan
    else:
        return np.nan


# 2d ILM to locate fovea center frame
def find_max_per_frame(ILM):
    ans = np.max(
        ILM[int(ILM.shape[1]/3):int(ILM.shape[1]*2/3), :], axis=0)
    return ans


def find_max_per_x(ILM):
    ans = np.max(ILM, axis=1)
    return ans


def apply_mask_ilm(ILM):
    masked_ILM = ILM.copy()
    masked_ILM = gaussian_filter(masked_ILM, (5))

    lbd = int(ILM.shape[1] / 2 - ILM.shape[1]//10)
    rbd = int(ILM.shape[1] / 2 + ILM.shape[1]//10)

    masked_ILM[:, :lbd] = 0
    masked_ILM[:, rbd+1:] = 0
    masked_ILM[:int(ILM.shape[0]/3), lbd:rbd+1] = 0
    masked_ILM[int(ILM.shape[0]*2/3):, lbd:rbd+1] = 0
    return masked_ILM, lbd, rbd


def find_fovea_center_frame(ILM, RPE):
    masked_ILM, lbd, rbd = apply_mask_ilm(ILM)

    ans = find_max_per_frame(masked_ILM)
    middle = ans[lbd:rbd+1]

    peaks, _ = find_peaks(middle, width=5)

    if peaks.shape[0] == 0:
        print("Can't find fovea frame!")
        return int(masked_ILM.shape[1]/2), masked_ILM.shape[1]//25
    else:
        peaks_shift = peaks+lbd

    frame_peak_ilm = np.argmax(ans[peaks_shift])

    results_half = peak_widths(middle,
                               [peaks[frame_peak_ilm]], rel_height=0.7)

    return peaks_shift[frame_peak_ilm], int(results_half[0][0]/2)

# new fovea detecting


def local_fovea(f, ILM, RPE):
    inverse_offset_orig = ILM[:, f] + 50 - gaussian_filter(RPE[:, f], (20))
    inverse_offset = inverse_offset_orig[int(
        ILM.shape[0]/3):int(ILM.shape[0]*2/3)+1]
    peaks, _ = find_peaks(inverse_offset, width=10)

    if peaks.shape[0] == 0:
        return list(range(int(ILM.shape[0]/2), int(ILM.shape[0]/2)+11))

    # currently use first peak
    index = np.argmax(inverse_offset[peaks])
    results_half = peak_widths(inverse_offset,
                               [peaks[index]], rel_height=0.3)

    return list(range(int(peaks[index]+ILM.shape[0]/3-results_half[0][0]/2), int(peaks[index]+ILM.shape[0]/3+results_half[0][0]/2)+1))


# Other utilities...
def get_RPE_fitpoly(i, RPE):
    y_fit = np.polyfit(np.arange(0, RPE.shape[0]), RPE[:, i], 3)
    p = np.poly1d(y_fit)
    fit = p(np.arange(0, RPE.shape[0]))
    return fit


def get_buffer_frame(center, great_fovea_region):
    if great_fovea_region[0] != center and great_fovea_region[-1] != center:
        buffer_frame = list(range(great_fovea_region[0],
                                  great_fovea_region[0]+int((center-great_fovea_region[0])/5)+1)) + list(range(great_fovea_region[-1]-int((great_fovea_region[-1]-center)/5), great_fovea_region[-1]+1))
    elif great_fovea_region[0] == center:
        buffer_frame = list(range(
            great_fovea_region[-1]-int((great_fovea_region[-1]-center)/5), great_fovea_region[-1]+1))
    elif great_fovea_region[-1] == center:
        buffer_frame = list(range(
            great_fovea_region[0], great_fovea_region[0]+int((center-great_fovea_region[0])/5)+1))
    # print("buffer frames are", buffer_frame)
    return buffer_frame


def find_distribution(img_obj, ILM, RPE):
    between = np.zeros(img_obj.shape)
    rpe_offset = 0.4
    for f in range(ILM.shape[1]):
        for x in range(ILM.shape[0]):
            between[int(ILM[x, f]):int(RPE[x, f]-(RPE[x, f]-ILM[x, f])*rpe_offset),
                    x,
                    f] = img_obj[int(ILM[x, f]):int(RPE[x, f]-(RPE[x, f]-ILM[x, f])*rpe_offset),
                                 x, f]

    this_mean = np.mean(between[between != 0])
    this_median = np.median(between[between != 0])
    this_std = np.std(between[between != 0])

    stats_z, pval = skewtest(between[between != 0])
    if stats_z > 4 and pval <= 0.05:
        return "right", this_median, this_std
    elif stats_z < -4 and pval <= 0.05:
        return "left", this_median, this_std
    else:
        return "normal", this_mean, this_std


def find_distribution_frame(f, stru3d, ILM, RPE):
    between = np.zeros(stru3d[:, :, f].shape)
    rpe_offset = 0.4
    for x in range(ILM.shape[0]):
        between[int(ILM[x, f]):int(RPE[x, f]-(RPE[x, f]-ILM[x, f])*rpe_offset),
                x] = stru3d[int(ILM[x, f]):int(RPE[x, f]-(RPE[x, f]-ILM[x, f])*rpe_offset),
                            x, f]

    this_mean = np.mean(between[between != 0])
    this_median = np.median(between[between != 0])
    this_std = np.std(between[between != 0])

    stats_z, pval = skewtest(between[between != 0])
    if stats_z > 4 and pval <= 0.05:
        return "right", this_median, this_std
    elif stats_z < -4 and pval <= 0.05:
        return "left", this_median, this_std
    else:
        return "normal", this_mean, this_std


def short_dis_threshold_frame(frame, ILM, RPE):
    between_dis = RPE[:, frame] - ILM[:, frame]
    dis_mean = np.mean(between_dis)
    dis_std = np.std(between_dis)
    return dis_mean - dis_std, dis_mean + dis_std


def get_int_threshold(distribution, med_or_mean, this_std):
    if distribution == "normal":
        return med_or_mean - 0.5*this_std
    else:
        return med_or_mean - 0.5*this_std


def find_buffer_x(fovea_index, frame, ILM, RPE):
    dis = RPE - ILM
    mean_dis = np.mean(dis[int(fovea_index[0]):int(fovea_index[-1]+1), frame])
    threshold = mean_dis

    ans = []
    just_buffer = []
    # search left
    for x in range(fovea_index[0]-1, 0, -1):
        if dis[x, frame] > threshold:
            break
        ans.append(x)
        just_buffer.append(x)

    ans.reverse()
    just_buffer.reverse()
    ans += list(range(fovea_index[0], fovea_index[-1]+1))

    # search right
    for x in range(fovea_index[-1], RPE.shape[0]):
        if dis[x, frame] > threshold:
            break
        ans.append(x)
        just_buffer.append(x)

    # if no extension - add 5 pixels
    if ans[0] == fovea_index[0]:
        ans = list(range(fovea_index[0]-5, fovea_index[0])) + ans
        just_buffer += list(range(fovea_index[0]-5, fovea_index[0]))
    if ans[-1] == fovea_index[0]:
        ans += list(range(fovea_index[-1]+1, fovea_index[-1]+6))
        just_buffer += list(range(fovea_index[-1]+1, fovea_index[-1]+6))

    return just_buffer, ans


def refine_fovea_frame(fovea_frame, center, stru3d):
    fovea_frame_new = list(fovea_frame).copy()

    # start at left-hand side
    while not compare_image_diff(fovea_frame_new[0], center, stru3d) or (compare_image_diff(fovea_frame_new[0], center, stru3d) and not compare_image_diff(fovea_frame_new[0]+1, center, stru3d) and fovea_frame_new[0] != center):
        fovea_frame_new = fovea_frame_new[1:]
    if fovea_frame_new[0] == fovea_frame[0]:
        while compare_image_diff(fovea_frame_new[0]-1, center, stru3d):
            fovea_frame_new = [fovea_frame_new[0]-1] + fovea_frame_new

    # process the right-hand side
    while not compare_image_diff(fovea_frame_new[-1], center, stru3d) or (compare_image_diff(fovea_frame_new[-1], center, stru3d) and not compare_image_diff(fovea_frame_new[-1]-1, center, stru3d) and fovea_frame_new[-1] != center):
        fovea_frame_new = fovea_frame_new[:-1]
    if fovea_frame_new[-1] == fovea_frame[-1]:
        while compare_image_diff(fovea_frame_new[-1]+1, center, stru3d):
            fovea_frame_new.append(fovea_frame_new[-1]+1)

    # print(f"The great buffer frame for this volumn is {fovea_frame_new}")
    return fovea_frame_new


def find_fovea_center(arr, frame, ILM, RPE):
    dis = ILM[arr[0]:arr[-1]+1, frame]
    max_list = np.argwhere(dis == np.amax(dis)).flatten().tolist()
    return max_list[len(max_list)//2] + arr[0]


def get_moving_slice(frame, frames_num, stru3d):
    if (frame > 2) and (frame < frames_num-3):
        img_slice = np.mean(stru3d[:, :, frame-2:frame+3], axis=2)
    elif frame >= 0 and frame <= 2:
        img_slice = np.mean(stru3d[:, :, 0:5], axis=2)
    else:
        img_slice = np.mean(stru3d[:, :, frames_num-5:], axis=2)
    return img_slice


def compare_image_diff(frame1, frame2, stru3d):
    frame_size = stru3d.shape[1]

    img = cv2.cvtColor(
        stru3d[:, int(frame_size/3):int(frame_size*2/3)+1, frame1], cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    image1_hash = imagehash.dhash(im_pil)

    img = cv2.cvtColor(
        stru3d[:, int(frame_size/3):int(frame_size*2/3)+1, frame2], cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    image2_hash = imagehash.dhash(im_pil)

    cutoff = 20  # maximum bits that could be different between the hashes.

    if image1_hash - image2_hash < cutoff:
        return True
    else:
        return False


def clean_local_outliers(y_):
    y = y_.copy()
    partition = 50
    for count in range(y.shape[0]//partition):
        if count == 0 or count == y.shape[0]//partition - 1:
            low_q = 15
            high_q = 85
        else:
            low_q = 5
            high_q = 95

        if (count+1) * partition < y.shape[0]:
            this_slice = y[count*partition:(count+1)*partition+1]
            y[count*partition:(count+1)*partition+1] = np.where((this_slice < np.nanpercentile(this_slice, low_q)) | (this_slice > np.nanpercentile(this_slice, high_q)),
                                                                np.nan, this_slice)
        else:
            this_slice = y[count*partition:]
            y[count*partition:] = np.where((this_slice < np.nanpercentile(this_slice, low_q)) | (this_slice > np.nanpercentile(this_slice, high_q)),
                                           np.nan, this_slice)
    return y


def find_fill_nan(arr, frame, upper, lower, ILM, max_window=50, count=0, edge=False):
    for x in range(ILM.shape[0]):
        if (count >= max_window and ~np.isnan(arr[x])) or (count >= max_window and x == arr.shape[0]-1):
            if ~np.isnan(offset):
                arr[start_index:x] = ILM[start_index:x, frame] + offset
            else:
                offset = arr[x] - ILM[x, frame]
                arr[start_index:x] = ILM[start_index:x, frame] + offset

            for c, val in enumerate(arr[start_index:x]):
                if val < upper[start_index+c]:
                    arr[start_index+c] = upper[start_index+c]+5
                elif val > lower[start_index+c]:
                    arr[start_index+c] = lower[start_index+c]-5
                if np.isnan(upper[start_index+c]):
                    arr[start_index+c] = np.nan
            max_window = 50

        if ~np.isnan(arr[x]) or x == 0:
            offset, count, start_index = 0, 0, 0
            continue

        if offset == 0:
            start_index = x
            if start_index != 0:
                offset = arr[x-1] - ILM[x-1, frame]
            else:
                offset = np.nan
            if start_index in range(50) or start_index in range(arr.shape[0]-50, arr.shape[0]):
                max_window = 0
        count += 1
    return arr


def interp_fill(arr, ILM, RPE):
    arr_interp = arr.copy()
    for i in range(0, arr.shape[1]):
        try:
            arr_interp[:, i] = np.interp(np.arange(0, arr.shape[0]),
                                         np.argwhere(
                ~np.isnan(arr[:, i])).flatten(),
                arr[~np.isnan(arr[:, i]), i])
        except ValueError:
            # print("Interpolation warning: empty array in frame", i)
            try:
                arr_interp[:, i] = (ILM[:, i] + RPE[:, i])/2
            except Exception:
                arr_interp[:, i] = np.zeros(ILM[:, i].shape[0])
    return arr_interp


def rearange_fovea_x(fovea_index, fovea_center):
    if fovea_index[-1] - fovea_center > fovea_center - fovea_index[0]:
        fovea_index = [2*fovea_center - fovea_index[-1],
                       fovea_index[-1]+1]
    elif fovea_index[-1] - fovea_center < fovea_center - fovea_index[0]:
        fovea_index = [fovea_index[0], 2*fovea_center - fovea_index[0]]
    return list(range(fovea_index[0], fovea_index[-1]+1))
