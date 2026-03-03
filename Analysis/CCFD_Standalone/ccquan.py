from scipy.ndimage import gaussian_filter, median_filter
import cv2
import time
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from skimage.morphology import remove_small_objects
import skfuzzy as fuzz
from skimage.measure import label, regionprops
import math
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import imageio
from Analysis.CCFD_Standalone.frangiFilter2D import *
from skimage.morphology import *
import sys
import os
from skimage import io
from skimage.filters import frangi
from skimage.morphology import disk, binary_closing, remove_small_objects


from scipy.ndimage.morphology import *

def PAremoval(reference_layer, target_layer, ratio):
    """
    Projection artificts removal, weighted substraction
    :param reference_layer:
    :param target_layer:
    :param ratio:
    :return:
    """
    removed_layer = target_layer - reference_layer * ratio
    return removed_layer


def scaleMaxMin(imgin, int_low, int_high):
    img_limit = 1
    img_cst = (imgin - int_low) * img_limit / (int_high - int_low)  # [0, 1]
    img_cst[img_cst < 0] = 0
    img_cst[img_cst > img_limit] = img_limit
    imgin[imgin < int_low] = int_low
    imgin[imgin > int_high] = int_high
    img_cut = (imgin - int_low) / (int_high - int_low)

    return img_cut

def bestK(img, pr):
    """
    Find the best number of cluster of K mean algorithm
    :param img:
    :param pr:
    :param CC_f:
    :return:
    """
    X = img.flatten()
    # random sample the image to reduce the
    X = np.random.choice(X, 50000)
    X = X.reshape(-1, 1)
    distortions=[]
    for k_temp in range(4, 11):
        kmeanModel = KMeans(n_clusters=k_temp).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)))
    # CC_f.flatten()
    jk1 = np.asarray(distortions[0:-1])
    jk2 = np.asarray(distortions[1::])
    variance = jk1 - jk2
    distortion_percent = np.cumsum(variance) / (distortions[0] - distortions[-1])
    res = list(map( lambda i: i > pr, distortion_percent)).index (True)
    K = res + 1
    print('The best K for this case is:', K)
    return K


def cc_comp(imgOCTACC, imgOCTCC, imgRetina, para1=1.5, para2=0.8, para3=0.001,
            exclusion_mask=None, hyperTD_mask=None):
    """
    make the compensation
    """
    img_ccocta = np.copy(imgOCTACC)
    img_ccoct = np.copy(imgOCTCC)
    img_retina = np.copy(imgRetina)

    if imgRetina.ndim == 3:
        imgRetina = imgRetina[:, :, 0]

    # --- Helper to get percentiles based on mask ---
    def get_robust_percentiles(image_data, mask, low_p=0.1, high_p=99.9):
        """Calculates percentiles using only VALID pixels."""
        if mask is not None:
            # Resize mask if needed
            if mask.shape != image_data.shape:
                from skimage.transform import resize
                # resizing boolean mask
                curr_mask = resize(mask, image_data.shape, anti_aliasing=False) > 0
            else:
                curr_mask = mask

            # Select valid pixels (where mask is False)
            valid_pixels = image_data[curr_mask == False]

            # Fallback if mask covers everything
            if len(valid_pixels) == 0:
                return np.percentile(image_data, low_p), np.percentile(image_data, high_p)

            return np.percentile(valid_pixels, low_p), np.percentile(valid_pixels, high_p)
        else:
            return np.percentile(image_data, low_p), np.percentile(image_data, high_p)

    # -----------------------------------------------

    kernel_sharpening = np.array([[-0.0024, - 0.0106, - 0.0176, - 0.0106, - 0.0024],
                                  [-0.0106, - 0.0477, - 0.0787, - 0.0477, - 0.0106],
                                  [-0.0176, - 0.0787, 1.6703, - 0.0787, - 0.0176],
                                  [-0.0106, - 0.0477, - 0.0787, - 0.0477, - 0.0106],
                                  [-0.0024, - 0.0106, - 0.0176, - 0.0106, - 0.0024]])

    sharpened = cv2.filter2D(img_ccocta, -1, kernel_sharpening)
    CC_Sum = cv2.GaussianBlur(sharpened, ksize=(3, 3), sigmaX=0.7, sigmaY=0.7, borderType=cv2.BORDER_DEFAULT)

    # --- Step 1: Normalize OCTA ---
    # Use exclusion_mask so dark vessels don't skew the range
    int_low0, int_high0 = get_robust_percentiles(CC_Sum, exclusion_mask, 0.1, 99.9)
    CC_Sum0 = scaleMaxMin(CC_Sum, int_low0, int_high0)

    # --- Step 2: Normalize Structural OCT ---
    int_low, int_high = get_robust_percentiles(img_ccoct, exclusion_mask, 1, 100)
    int_high = int_high * para1  # apply parameter 1

    CC_Str_Sum_cut = scaleMaxMin(img_ccoct, int_low, int_high)
    Fliter_MaskCC = 1 - median_filter(CC_Str_Sum_cut, size=5)
    Fliter_MaskCC = Fliter_MaskCC / (max(Fliter_MaskCC.flatten()))

    # Apply Parameter 2 (Gamma)
    Fliter_MaskCC = Fliter_MaskCC ** para2

    # --- Step 3: Apply HyperTD Logic ---
    # Force compensation to 1.0 (No compensation) in HyperTD regions
    if hyperTD_mask is not None:
        if hyperTD_mask.shape != Fliter_MaskCC.shape:
            from skimage.transform import resize
            hyperTD_mask = resize(hyperTD_mask, Fliter_MaskCC.shape, anti_aliasing=False) > 0
        Fliter_MaskCC[hyperTD_mask == True] = 0.8

    MaxCC_Correct_CCSum = CC_Sum * (Fliter_MaskCC)

    # --- Step 4: Final Normalization ---
    # CRITICAL: Use exclusion_mask here.
    # This prevents the black GA regions from pulling 'int_low' down.
    # HEADROOM_FACTOR maintains contrast against bright HyperTDs.
    HEADROOM_FACTOR = 1.35

    int_low, int_high_raw = get_robust_percentiles(MaxCC_Correct_CCSum, exclusion_mask, 1.0, 99.0)
    int_high = int_high_raw * HEADROOM_FACTOR

    MaxCC_Correct_CCSum = scaleMaxMin(MaxCC_Correct_CCSum, int_low, int_high)

    CC_f = PAremoval(img_retina, MaxCC_Correct_CCSum, para3)

    # --- Step 5: Final Output Normalization ---
    int_low, int_high_raw = get_robust_percentiles(CC_f, exclusion_mask, 1.0, 99.0)
    int_high = int_high_raw * HEADROOM_FACTOR

    CC_f = scaleMaxMin(CC_f, int_low, int_high)

    if CC_f.ndim == 3:
        CC_f = CC_f[:, :, 0]

    return CC_f, CC_Sum0


def fuzz_CC_thresholding(img_retina_3, CC_f_3, threshold_largevessel, scansize=6, k_val=None, CCthresAdj=2.0,
                         img_mask=None, scaleX=1.0, scaleY=1.0, save_path=None, def_fovea_center=None):
    """
    Refined Thresholding function that ensures clean binary map return.
    """
    # 1. Standardize Inputs to 2D
    if img_retina_3.ndim == 3:
        img_retina = img_retina_3[:, :, 0]
    else:
        img_retina = img_retina_3

    if CC_f_3.ndim == 3:
        CC_f = CC_f_3[:, :, 0]
    else:
        CC_f = CC_f_3

    # 2. Resize Retina to match CC_f if needed
    target_h, target_w = CC_f.shape
    if img_retina.shape != CC_f.shape:
        img_retina = cv2.resize(img_retina, (target_w, target_h))

    CC_f_3 = np.copy(CC_f_3)
    pixelsize = scansize * 1000 / target_h
    pixelsizeX = pixelsize * scaleX
    pixelsizeY = pixelsize * scaleY

    CC_f_res = cv2.resize(CC_f_3, (target_w, target_h))
    CC_f = np.copy(CC_f_res)

    # 3. Generate Vessel Mask
    if abs(threshold_largevessel - 0.6) < 0.01:
        print(f"Using Standard Thresholding (Old Method, Thresh={threshold_largevessel})...")
        J = cv2.GaussianBlur(img_retina, (3, 3), cv2.BORDER_DEFAULT)
        J = scaleMaxMin(J, J.min(), J.max())
        binary = J > threshold_largevessel
        retina_mask = remove_small_objects(binary, 200)
    else:
        print("Using Frangi Vesselness (New Method)...")
        img_norm = img_retina.astype(float)
        if img_norm.max() > img_norm.min():
            img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())

        sigmas = np.linspace(3.5, 10.0, 4)
        vesselness = frangi(img_norm, sigmas=sigmas, black_ridges=False)
        thresh_val = 0.12 * vesselness.max()
        binary = vesselness > thresh_val
        footprint = disk(2)
        bridged_mask = binary_closing(binary, footprint)
        retina_mask = remove_small_objects(bridged_mask, min_size=500)

    # 4. Handle User Masks
    if img_mask is None:
        img_mask = np.zeros((target_h, target_w), dtype=bool)
    else:
        # Resize mask if dimensions don't match
        if img_mask.shape[:2] != (target_h, target_w):
            img_mask = cv2.resize(img_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        img_mask = img_mask.astype(bool)
        if img_mask.ndim == 3: img_mask = img_mask[:, :, 0]

    # Combine Masks
    total_mask = np.logical_or(retina_mask, img_mask)

    # 5. FCM Thresholding Logic
    CC_f_ori = np.copy(CC_f)
    CC_f[total_mask == True] = 0  # Zero out mask for clustering

    if k_val is None:
        K = bestK(CC_f, 0.96)
    else:
        K = k_val

    X = CC_f.flatten()
    X = X[X != 0]  # remove 0 (masked areas)
    X = X.reshape(-1, 1)

    # Fuzzy C-Means
    try:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, K, 2, error=0.005, maxiter=1000, seed=int(42))
        FCM_labels = np.argmax(u, axis=0)
        min_ind = np.argmin(cntr)
        CC_group = X[FCM_labels == min_ind]
        CC_thres = CC_group.max() * CCthresAdj
    except:
        # Fallback if FCM fails (e.g. empty image)
        CC_thres = 0.1
        K = 0

    # 6. Apply Threshold
    # Apply to flow map (result inverted: 1=deficit, 0=flow)
    CC_m = np.copy(CC_f)
    CC_m[CC_m <= CC_thres] = 0
    CC_m[CC_m > CC_thres] = 1
    CC_m = 1 - CC_m

    # Filter small objects
    thres_size = round(((24 / 2) ** 2 * math.pi) / pixelsizeX / pixelsizeY)

    # Clean Mask
    CC_m[total_mask == True] = 0
    CCMask1 = CC_m > 0
    CCMask1 = remove_small_objects(CCMask1, thres_size)

    # 7. Calculate Metrics
    label_image = label(CCMask1)
    CCMask = CCMask1 * 1
    CCFDN = label_image.max()

    valid_area = CCMask.shape[0] * CCMask.shape[1] - total_mask.sum()
    CCFDD = (CCMask.sum()) / valid_area if valid_area > 0 else 0
    CCFDSize = (CCMask.sum() * pixelsizeX * pixelsizeY) / CCFDN if CCFDN > 0 else 0

    # 8. Visualization (Separate function)
    vols, means = visualize_ccfdd(CCMask1, CCFDN, total_mask, pixelsizeX, pixelsizeY, scansize, save_path,
                                  def_fovea_center)

    # 9. Create Color Overlay Image
    if CC_f.ndim == 3:
        CC_f_stack = CC_f[:, :, 0]
    else:
        CC_f_stack = CC_f

    img_color = np.dstack((CC_f_stack, CC_f_stack, CC_f_stack))

    # We return the RAW binary mask (CCMask1) so the main code can use it for Row 4/5
    return CCMask1, img_color, CCFDSize, CCFDD, K, CC_f_ori, vols, means, total_mask


import numpy as np
from scipy.ndimage import generic_filter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def visualize_ccfdd(CCMask1, CCFDN, total_mask, pixelsizeX, pixelsizeY, scansize, save_path=None,
                    def_fovea_center=None):
    """
    Calculates 1mm, 3mm, 5mm stats and saves circular overlays if save_path is provided.
    Does NOT affect the return of the main binary mask.
    """
    if CCMask1.ndim == 3: CCMask1 = CCMask1[:, :, 0]
    if total_mask.ndim == 3: total_mask = total_mask[:, :, 0]

    img_shape = CCMask1.shape
    center = (img_shape[0] // 2, img_shape[1] // 2)
    if def_fovea_center is not None: center = def_fovea_center

    circ_save_ccfdd = np.zeros(3)
    circ_save_size = np.zeros(3)
    radii_mm = [0.5, 1.5, 2.5]
    radii_px = [int(r * 1000 / pixelsizeX) for r in radii_mm]

    # Calculate stats for each circle
    cnt = 0
    for radius_px in radii_px:
        # Create mask for this circle
        Y, X = np.ogrid[:img_shape[0], :img_shape[1]]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        circle_mask = dist_from_center <= radius_px

        circular_CCMask = CCMask1 * circle_mask
        circular_total_mask = total_mask > 0
        circular_total_mask = circular_total_mask * circle_mask

        valid_circle_area = circle_mask.sum() - circular_total_mask.sum()

        if valid_circle_area > 0:
            circular_CCFDD = circular_CCMask.sum() / valid_circle_area
        else:
            circular_CCFDD = 0

        circular_CCFDN = CCFDN * (circle_mask.sum() / CCMask1.size)  # Approx
        circular_CCFDSize = (
                                        circular_CCMask.sum() * pixelsizeX * pixelsizeY) / circular_CCFDN if circular_CCFDN > 0 else 0

        circ_save_ccfdd[cnt] = circular_CCFDD
        circ_save_size[cnt] = circular_CCFDSize
        cnt += 1

    # Save the figures
    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ccfdd_path = os.path.join(save_dir, "ccfdd_visualization.png")
        size_path = os.path.join(save_dir, "ccfdsize_visualization.png")
        #circ_save_ccfdd.savefig(ccfdd_path, dpi=300, bbox_inches='tight')
        #circ_save_size.savefig(size_path, dpi=300, bbox_inches='tight')
        print(f"CCFDD visualization saved to {ccfdd_path}")

    # Return
    return circ_save_ccfdd, circ_save_size
def phansalkar_thresholding(img_retina, CC_f, scansize=6, img_mask=None, scaleX=1.0, scaleY=1.0):
    """
    Thresholding the image using phansalkar method
    :param img_retina:
    :param CC_f: compensated flow
    :param scansize:
    :return:
    """
    CC_f = np.copy(CC_f)
    # img_retina = imageio.imread('F:/OCTpy/Test/demo/ZSubZeissFlowFilter/projs_flow_0.png')
    # CC_f = imageio.imread('F:/OCTpy/Test/demo/comp_img.png')
    # scansize = 6

    pixelsize = scansize * 1000 / img_retina.shape[0]

    pixelsizeX = pixelsize*scaleX
    pixelsizeY = pixelsize*scaleY

    # get the retina vessel mask first
    J = cv2.GaussianBlur(img_retina, (3, 3), cv2.BORDER_DEFAULT )
    J = scaleMaxMin(J, J.min(), J.max())
    binary = J > 0.6
    retina_mask = remove_small_objects(binary, 200)

    #total mask
    total_mask = np.logical_or(retina_mask, img_mask)


    # use the average value to fill the mask region
    CC_f_ori = np.copy(CC_f)
    ave = np.mean(CC_f[total_mask==True])
    ave = 0 # fill with zero
    CC_f[total_mask==True] = ave

    CC_m = 1 - phansalkar(CC_f, window=(2, 2))

    thres = round(((24/2)**2*math.pi)/pixelsizeX/pixelsizeY)

    # remove the masked region
    cc_original = np.copy(CC_m)     # apply to origianl CC_F, this is the CCFD without applying the excluded mask
    CC_m[total_mask == True] = 0
    CCMask1 = CC_m > 0  # convert to Binary

    CCMask1 = remove_small_objects(CCMask1, thres)
    label_image = label(CCMask1)
    CCMask = CCMask1*1
    CCFDN = label_image.max()
    CCFDSize = (CCMask.sum() * pixelsizeX * pixelsizeY) /CCFDN    # mean FD size
    CCFDD = (CCMask.sum()) / (CCMask.shape[0]*CCMask.shape[1] - total_mask.sum())  # FD density removing the complete mask from the denominator

    img_color = np.dstack((CC_f, CC_f, CC_f))

    out = img_color.copy()
    img_layer = img_color.copy()
    img_layer[CCMask1] = [0.8, 0, 0]
    img_layer[total_mask] = [1, 1, 0]
    out = cv2.addWeighted(img_layer, 0.9, out, 0.1, 0, out)
    cc_original = np.uint8(cc_original)*255
    K = -1
    return CCMask, out, CCFDSize, CCFDD, K, cc_original





def phansalkar(image, window=(15, 15), k=0.25, padding='reflect'):
    # Convert to float
    image = image.astype(float)

    # Mean value
    mean = generic_filter(image, np.mean, size=window, mode=padding)

    # Standard deviation
    mean_square = generic_filter(image ** 2, np.mean, size=window, mode=padding)
    deviation = np.sqrt(mean_square - mean ** 2)

    # Phansalkar
    R = deviation.max()
    p = 2
    q = 10
    threshold = mean * (1 + p * np.exp(-q * mean) + k * ((deviation / R) - 1))

    output = (image > threshold).astype(np.uint8)

    return output


# Example usage:
# Load an image (replace 'eight.tif' with your image file path)
# from skimage import io
# image = io.imread('eight.tif', as_gray=True)

# Call the function
# result = phansalkar(image, window=(150, 150))

# Show the result using matplotlib
# import matplotlib.pyplot as plt
# plt.imshow(result, cmap='gray')
# plt.show()


def get_bw_image(img, thresLV):
    """
    get larger vessel
    :param img:
    :param thresLV: Input threshold for large vessels, from 0.5 to 3.0 (usually 1.1 for 6x6mm)
    :return:
    """

    # magic parameter
    XWidth = img.shape[1]
    YHeight = img.shape[0]
    if XWidth == 840:
        addOn = 40
    else:
        addOn = 25


    Totalength = XWidth + addOn * 2

    largeImg = np.ones((Totalength, Totalength))
    largeImg[0: addOn, addOn : addOn + XWidth] = img[XWidth - addOn: XWidth, 0: XWidth]
    largeImg[addOn + XWidth: XWidth + 2 * addOn, addOn: addOn + XWidth] = img[0: addOn, 0: XWidth]
    largeImg[addOn: addOn + XWidth, addOn: addOn + XWidth] = img    # center img itself

    largeImg[addOn: addOn + XWidth, 0: addOn] = img[0: XWidth, XWidth - addOn: XWidth]
    largeImg[addOn: addOn + XWidth, addOn + XWidth: XWidth + addOn * 2] = img[0: XWidth, 0: addOn]

    imgLV = largeImg

    imgLV = gaussian_filter(imgLV, thresLV)


    if imgLV.ndim>2:
        imgLV2 = np.double(imgLV[:, :, 0])
    else:
        imgLV2 = np.double(imgLV)

    [outIm, whatScale, direction] = FrangiFilter2D(imgLV2, FrangiScaleRange=np.array([2, 6]),
                                                   FrangiScaleRatio=1,
                                                   FrangiBetaOne=1, FrangiBetaTwo=13, verbose=True,
                                                   BlackWhite=False)
    # outIm = frangi(imgLV2, scale_range=(2, 6), scale_step=0.1, beta1=1, beta2=13, black_ridges=False)

    img_bw = outIm > 0.85 # binarize and clean

    img_bw = remove_small_objects(img_bw, 15, 8)  # remove small objects

    # did not find
    # img_bw = bwmorph(img_bw, 'bridge') # connect broken parts

    se = generate_binary_structure(2, 1)    # remove small objects
    img_bw = binary_dilation(img_bw, se)    #dilate image
    img_bw_clean = remove_small_objects(img_bw, 50, 8)
    img_close = thin(img_bw_clean, 1)

    # plt.imshow(img_bw_clean)
    # plt.show()
    #
    # plt.imshow(img_close)
    # plt.show()

    img_bw_clean = remove_small_objects(img_close, 100, 8)

    # img_bw_clean = bwmorph(img_bw_clean, 'bridge');

    # largeImg = np.ones((XWidth, YHeight))
    largeImg = img_bw_clean[addOn:addOn + XWidth, addOn: addOn + XWidth]
    # img_bw_clean = np.double(largeImg)
    # img_bw_cleanN = img_bw_clean
    # img_bw_cleanN[img_bw_clean == 0] = np.NaN

    return largeImg

def get_region_score(cc_img, mask, GA_mask, mask_ind, pixelsize_X, pixelsize_Y):
    """
    Args:
        cc_img:
        mask:
        mask_ind:

    Returns:

    """
    # pixelsize = 12
    new_mask = np.zeros_like(cc_img)
    print("Shape of new_mask:", new_mask.shape)
    print("Shape of mask:", mask.shape)
    mask = resize(mask, (500, 500), anti_aliasing=True)
    print("Shape of GA mask after rez:", GA_mask.shape)
    new_mask[mask==mask_ind] = 1

    CCMask = cc_img*new_mask

    label_image = label(CCMask)

    CCFDN = label_image.max()
    CCFDSize = (CCMask.sum() * pixelsize_X * pixelsize_Y*1000*1000) /CCFDN    # mean FD size
    CCFDD = (CCMask.sum()) / (new_mask.sum())  # FD density
    new_mask = new_mask[:,:, 0]
    GA_mask_circle = GA_mask*new_mask
    MaskPrec = (GA_mask_circle.sum()/ new_mask.sum())
    print('Saved')

    return CCFDD, CCFDSize, CCFDN, MaskPrec


