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
from Analysis.CCFD import frangiFilter2D
import skimage.morphology
import sys
import os
from skimage import io


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


def cc_comp(imgOCTACC, imgOCTCC, imgRetina, para1=1.5, para2=0.8, para3=0.001):
    """
    make the compasitation
    :param imgOCTACC:
    :param imgOCTCC:
    :param imgRetina:
    :return:
    """
    print('Para 1 is', para1, ' and Para 2 is', para2)
    img_ccocta = np.copy(imgOCTACC)
    img_ccoct = np.copy(imgOCTCC)
    img_retina = np.copy(imgRetina)

    # Create our shapening kernel, it must equal to one eventually
    # kernel_sharpening = np.array([[-1, -1, -1],
    #                               [-1, 9, -1],
    #                               [-1, -1, -1]])

    kernel_sharpening = np.array([[-0.0024, - 0.0106, - 0.0176, - 0.0106, - 0.0024],
        [-0.0106, - 0.0477, - 0.0787, - 0.0477, - 0.0106],
        [-0.0176, - 0.0787,  1.6703, - 0.0787, - 0.0176],
        [-0.0106, - 0.0477, - 0.0787, - 0.0477, - 0.0106],
        [-0.0024, - 0.0106, - 0.0176, - 0.0106, - 0.0024]])

    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(img_ccocta, -1, kernel_sharpening)
    CC_Sum = cv2.GaussianBlur(sharpened, ksize=(3, 3), sigmaX=0.7, sigmaY=0.7, borderType=cv2.BORDER_DEFAULT)

    # plt.imshow(CC_Sum, cmap='gray'), plt.title('Averaging')
    # plt.show()
    # OCTA normalized without the 1.5 parameter
    int_low0 = np.percentile(CC_Sum, 0.1)
    int_high0 = np.percentile(CC_Sum, 99.9)
    CC_Sum0 = scaleMaxMin(CC_Sum, int_low0, int_high0)

    # OCT normalized with the 1.5 parameter
    int_low = np.percentile(img_ccoct, 1)
    int_high = np.percentile(img_ccoct, 100) * para1

    CC_Str_Sum_cut = scaleMaxMin(img_ccoct, int_low, int_high)
    Fliter_MaskCC = 1 - median_filter(CC_Str_Sum_cut, size=5)
    Fliter_MaskCC = Fliter_MaskCC / (max(Fliter_MaskCC.flatten()))
    Fliter_MaskCC = Fliter_MaskCC ** para2
    MaxCC_Correct_CCSum = CC_Sum * (Fliter_MaskCC)
    int_low = np.percentile(MaxCC_Correct_CCSum, 0.1)
    int_high = np.percentile(MaxCC_Correct_CCSum, 99.9)
    MaxCC_Correct_CCSum = scaleMaxMin(MaxCC_Correct_CCSum, int_low, int_high)

    CC_f = PAremoval(img_retina, MaxCC_Correct_CCSum, para3)    # tailing artificts
    int_low = np.percentile(CC_f, 0.1)
    int_high = np.percentile(CC_f, 99.9)
    CC_f = scaleMaxMin(CC_f, int_low, int_high)
    return CC_f, CC_Sum0

def cc_comp_nan(imgOCTACC, imgOCTCC, imgRetina, para1=1.5, para2=0.8, para3=0.001):
    """
    make the compasitation
    :param imgOCTACC:
    :param imgOCTCC:
    :param imgRetina:
    :return:
    """
    img_ccocta = np.copy(imgOCTACC)
    img_ccoct = np.copy(imgOCTCC)
    img_retina = np.copy(imgRetina)

    # Create our shapening kernel, it must equal to one eventually
    # kernel_sharpening = np.array([[-1, -1, -1],
    #                               [-1, 9, -1],
    #                               [-1, -1, -1]])

    kernel_sharpening = np.array([[-0.0024, - 0.0106, - 0.0176, - 0.0106, - 0.0024],
        [-0.0106, - 0.0477, - 0.0787, - 0.0477, - 0.0106],
        [-0.0176, - 0.0787,  1.6703, - 0.0787, - 0.0176],
        [-0.0106, - 0.0477, - 0.0787, - 0.0477, - 0.0106],
        [-0.0024, - 0.0106, - 0.0176, - 0.0106, - 0.0024]])

    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(img_ccocta, -1, kernel_sharpening)
    CC_Sum = cv2.GaussianBlur(sharpened, ksize=(3, 3), sigmaX=0.7, sigmaY=0.7, borderType=cv2.BORDER_DEFAULT)

    # OCTA normalized without the 1.5 parameter
    int_low0 = np.percentile(CC_Sum, 0.1)
    int_high0 = np.percentile(CC_Sum, 99.9)
    CC_Sum0 = scaleMaxMin(CC_Sum, int_low0, int_high0)

    # OCT normalized with the 1.5 parameter
    int_low = np.percentile(img_ccoct, 1)
    int_high = np.percentile(img_ccoct, 100) * para1

    CC_Str_Sum_cut = scaleMaxMin(img_ccoct, int_low, int_high)
    Fliter_MaskCC = 1 - median_filter(CC_Str_Sum_cut, size=5)
    Fliter_MaskCC = Fliter_MaskCC / (max(Fliter_MaskCC.flatten()))
    Fliter_MaskCC = Fliter_MaskCC ** para2
    MaxCC_Correct_CCSum = CC_Sum * (Fliter_MaskCC)
    int_low = np.percentile(MaxCC_Correct_CCSum, 0.1)
    int_high = np.percentile(MaxCC_Correct_CCSum, 99.9)
    MaxCC_Correct_CCSum = scaleMaxMin(MaxCC_Correct_CCSum, int_low, int_high)

    CC_f = PAremoval(img_retina, MaxCC_Correct_CCSum, para3)    # tailing artificts
    int_low = np.percentile(CC_f, 0.1)
    int_high = np.percentile(CC_f, 99.9)

    CC_f = scaleMaxMin(CC_f, int_low, int_high)

    return CC_f, CC_Sum0

def fuzz_CC_thresholding(img_retina_3, CC_f_3, threshold_largevessel, scansize=6, k_val=None, CCthresAdj=2.0, img_mask=None, scaleX=1.0, scaleY=1.0, save_path = None, def_fovea_center = None):
    """
    Thresholding the image
    :param img_retina:
    :param CC_f:
    :param scansize:
    :return:
    """
    CC_f_3 = np.copy(CC_f_3)
    pixelsize = scansize * 1000 / img_retina_3.shape[0]

    pixelsizeX = pixelsize*scaleX
    pixelsizeY = pixelsize*scaleY
    img_retina_3 = cv2.resize(img_retina_3, (500,500))
    img_retina = img_retina_3
    #check here
    CC_f_res = cv2.resize(CC_f_3, (500, 500))
    #CC_f = CC_f_res[:,:,0]
    CC_f = np.copy(CC_f_res)


    J = cv2.GaussianBlur ( img_retina , (3 , 3) , cv2.BORDER_DEFAULT )
    J = scaleMaxMin ( J , J.min() , J.max() )
    #change the binary value
    binary = J > threshold_largevessel
    print(threshold_largevessel)
    retina_mask = remove_small_objects (binary, 200)


    #dfolder = "D:\\test\\check"
    #result_path = os.path.join(dfolder, f'processed_img_i.png')
    #io.imsave(result_path, np.uint8(retina_mask)*255)
    #was 0.6


    #img_mask = reddims(img_mask)
    #binary_mask = (img_mask > 0).astype(np.uint8)
    #mod_img_mask = np.stack([binary_mask] * 3, axis=-1) * 255
    if img_mask is None:
        img_mask = np.zeros((500, 500), dtype=bool)
    else:
        img_mask = img_mask.astype(bool)
    #img_mask = img_mask.astype(bool)
    if img_mask.shape == (500,500,3):
        mod_img_mask = img_mask[:,:,0]
    else:
        mod_img_mask = img_mask
    total_mask = np.logical_or(retina_mask, mod_img_mask)

    # use the average value to fill the mask region
    CC_f_ori = np.copy(CC_f)
    #ave = np.mean(CC_f[total_mask==True])
    ave = 0 # fill with zero
    CC_f[total_mask==True] = ave


    # plt.imshow ( retina_mask , cmap='gray'  )
    # plt.show ()

    # calculate the best K if does not have input
    if k_val is None:
        K = bestK(CC_f, 0.96)
    else:
        K = k_val

    X = CC_f.flatten()

    X = X[X != 0]   # remove 0
    X = X.reshape(-1, 1)

    # get the threshold from fuzz C mean algorithm
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T, K, 2, error=0.005, maxiter=1000, seed=int(42))

    FCM_labels = np.argmax(u, axis=0)
    FCM_labels = FCM_labels.reshape(-1, 1)
    min_ind = np.argmin(cntr)
    CC_group = X[FCM_labels == min_ind]     # get the membership for CC group
    CC_thres = CC_group.max() * CCthresAdj

    print('Threshold: ', CC_thres)   #print the final threshold

    CC_m = np.copy(CC_f)
    CC_m[CC_m <= CC_thres] = 0
    CC_m[CC_m > CC_thres] = 1
    CC_m = 1 - CC_m

    thres = round(((24/2)**2*math.pi)/pixelsizeX/pixelsizeY)
    print('Threshold: ', thres)
    # remove the masked region
    cc_original = np.copy(CC_m)
    CC_m[total_mask == True] = 0
    CCMask1 = CC_m > 0  # convert to Binary

    CCMask1 = remove_small_objects(CCMask1, thres)


    # apply to origianl CC_F, this is the CCFD without applying the excluded mask
    CC_f_ori[CC_f_ori<=CC_thres] = 0
    CC_f_ori[CC_f_ori > CC_thres] = 1
    CC_f_ori = 1 - CC_f_ori

    cc_original = CC_f_ori > 0
    cc_original = remove_small_objects(cc_original, thres)

    label_image = label(CCMask1)
    CCMask = CCMask1*1
    CCFDN = label_image.max()

    scansize = 6  # mm
    img_shape = (500, 500)  # Resized shape
    pixelsize = scansize * 1000 / img_shape[0]
    pixelsizeX = pixelsize  # Assuming scaleX = 1.0
    pixelsizeY = pixelsize  # Assuming scaleY = 1.0

    #total_mask = np.zeros(img_shape, dtype=bool)
    vols, means = visualize_ccfdd(CCMask1, CCFDN, total_mask, pixelsizeX, pixelsizeY, scansize, save_path, def_fovea_center)


    CCFDSize = (CCMask.sum() * pixelsizeX * pixelsizeY) /CCFDN    # mean FD size
    CCFDD = (CCMask.sum()) / (CCMask.shape[0]*CCMask.shape[1] - total_mask.sum())  # FD density

    #img_color = np.dstack((CC_f, CC_f, CC_f))
    img_color = np.dstack((CC_f, CC_f, CC_f))
    total_mask = total_mask.astype(bool)

    out = img_color.copy()
    img_layer = img_color.copy()
    img_layer[CCMask1] = [0.8, 0, 0]
    img_layer[total_mask] = [1, 1, 0]
    out = cv2.addWeighted(img_layer, 0.9, out, 0.1, 0, out)
    CCMask = np.uint8(CCMask) * 255
    cc_original = np.uint8(cc_original)*255
    return CCMask, out, CCFDSize, CCFDD, K, cc_original, vols, means


import numpy as np
from scipy.ndimage import generic_filter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask

def visualize_ccfdd(CCMask1, CCFDN, total_mask, pixelsizeX, pixelsizeY, scansize, save_path=None, def_fovea_center = None):
    img_shape = CCMask1.shape
    center = (img_shape[0] // 2, img_shape[1] // 2)
    if def_fovea_center is not None:
        center = def_fovea_center
    circ_save_ccfdd = np.zeros(3)
    circ_save_size = np.zeros(3)

    # Calculate overall CCFDD and CCFDSize
    overall_CCFDD = CCMask1.sum() / (
                CCMask1.shape[0] * CCMask1.shape[1] - total_mask.sum())
    overall_CCFDSize = (
                                   CCMask1.sum() * pixelsizeX * pixelsizeY) / CCFDN if CCFDN > 0 else 0

    # Calculate radii in pixels
    radii_mm = [0.5, 1.5, 2.5]  # Radii for 1mm, 3mm, 5mm diameter circles
    radii_px = [int(r * 1000 / pixelsizeX) for r in
                radii_mm]  # Convert mm to μm, then to pixels

    # Create figures for visualization
    fig_ccfdd = Figure(figsize=(10, 10))
    canvas_ccfdd = FigureCanvasAgg(fig_ccfdd)
    ax_ccfdd = fig_ccfdd.add_subplot(111)

    fig_size = Figure(figsize=(10, 10))
    canvas_size = FigureCanvasAgg(fig_size)
    ax_size = fig_size.add_subplot(111)

    # Plot CCMask1 on both figures
    ax_ccfdd.imshow(CCMask1, cmap='gray')
    ax_size.imshow(CCMask1, cmap='gray')

    cnt = 0
    colors = ['y', 'y', 'y']
    for radius_px, diameter_mm, color in zip(radii_px, [1, 3, 5], colors):
        # Create circular mask
        circle_mask = create_circular_mask(img_shape[0], img_shape[1], center,
                                           radius_px)

        # Then modify the calculation to use the 2D version:
        circular_CCMask = CCMask1 * circle_mask

        circular_total_mask = total_mask * circle_mask
        circular_CCFDD = circular_CCMask.sum() / (
                    circle_mask.sum() - circular_total_mask.sum())

        # Calculate circular CCFDN (proportional to the mask area)
        circular_CCFDN = CCFDN * (circle_mask.sum() / CCMask1.size)
        circular_CCFDSize = (
                                        circular_CCMask.sum() * pixelsizeX * pixelsizeY) / circular_CCFDN if circular_CCFDN > 0 else 0

        # Plot circles and add text for CCFDD
        circle_patch_ccfdd = plt.Circle(center, radius_px, fill=False,
                                        edgecolor=color, linewidth=1)
        ax_ccfdd.add_patch(circle_patch_ccfdd)

        # Plot circles and add text for CCFDSize
        circle_patch_size = plt.Circle(center, radius_px, fill=False,
                                       edgecolor=color, linewidth=1)
        ax_size.add_patch(circle_patch_size)
        print(
            f"{diameter_mm}mm diameter circle CCFDD: {circular_CCFDD:.4f}, Size: {circular_CCFDSize:.2f} μm²")
        circ_save_ccfdd[cnt] = circular_CCFDD
        circ_save_size[cnt] = circular_CCFDSize
        cnt = cnt + 1

        # Export circle masked image
        if save_path:
            print(save_path)
            finsave = (save_path)
            masked_image = CCMask1 * circle_mask
            masked_image_path = os.path.join(finsave,
                                             f"circle_masked_{diameter_mm}mm.png")
            plt.imsave(masked_image_path, masked_image, cmap='gray')
            print(
                f"Circle masked image ({diameter_mm}mm) saved to {masked_image_path}")

    ax_ccfdd.set_title(
        f'CCMask1 with Circular Regions\nOverall CCFDD: {overall_CCFDD:.4f}',
        fontweight='bold', fontsize=14)
    ax_size.set_title(
        f'CCMask1 with Circular Regions\nOverall CCFDSize: {overall_CCFDSize:.2f} μm²',
        fontweight='bold', fontsize=14)

    # Remove axis ticks
    ax_ccfdd.set_xticks([])
    ax_ccfdd.set_yticks([])
    ax_size.set_xticks([])
    ax_size.set_yticks([])

    # Tight layout
    fig_ccfdd.tight_layout()
    fig_size.tight_layout()

    # Save the figures if a save path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ccfdd_path = os.path.join(save_dir, "ccfdd_visualization.png")
        size_path = os.path.join(save_dir, "ccfdsize_visualization.png")
        fig_ccfdd.savefig(ccfdd_path, dpi=300, bbox_inches='tight')
        fig_size.savefig(size_path, dpi=300, bbox_inches='tight')
        print(f"CCFDD visualization saved to {ccfdd_path}")
        print(f"CCFDSize visualization saved to {size_path}")

    # Display the plots
    # plt.show()

    print(f"\nImage dimensions: {img_shape[0]}x{img_shape[1]} pixels")
    print(f"Pixel size X: {pixelsizeX:.2f} μm, Y: {pixelsizeY:.2f} μm")
    print(f"Scan size: {scansize}x{scansize} mm")
    print(f"Overall CCFDD: {overall_CCFDD:.4f}")
    print(f"Overall CCFDSize: {overall_CCFDSize:.2f} μm²")
    print(f"Exporting values CCFDD: {circ_save_ccfdd}")
    print(f"Exporting values CCFDSize: {circ_save_size}")

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
    CCFDD = (CCMask.sum()) / (CCMask.shape[0]*CCMask.shape[1] - total_mask.sum())  # FD density

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


