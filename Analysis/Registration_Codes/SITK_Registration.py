import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.pyplot as plt

def display_image(image):
    if isinstance(image, sitk.Image):
        # Convert SimpleITK image to numpy array
        array = sitk.GetArrayFromImage(image)
    plt.imshow(array, cmap='gray')  # use cmap='gray' for grayscale images
    plt.axis('off')
    plt.show()


def load_image(file_path, is_color=False):
    return sitk.ReadImage(file_path)


def register_images(fixed_image, moving_image):
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    regi_sitk = sitk.ImageRegistrationMethod()
    # For the following registration method, the following metrics need
    # to be set: Similarity, Transformation, Optimizer, Interpolator


    # Similarity metric
    regi_sitk.SetMetricAsCorrelation()

    # Interpolator
    regi_sitk.SetInterpolator(sitk.sitkLinear)

    # Optimizer
    regi_sitk.SetOptimizerAsLBFGS2(numberOfIterations=100)
    regi_sitk.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework
    regi_sitk.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    regi_sitk.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    regi_sitk.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would like to see how the transform improves
    regi_sitk.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = regi_sitk.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32))

    print(
        'Final metric value: {0}'.format(regi_sitk.GetMetricValue()))
    return final_transform

def apply_transform(image, reference_image, transform):
    return sitk.Resample(image, reference_image, transform, sitk.sitkLinear, 0.0, image.GetPixelID())


def apply_transform_multichannel(image, reference_image, transform):
    # Convert SimpleITK image to numpy array
    image_array = sitk.GetArrayFromImage(image)

    # Get image dimensions
    shape = image_array.shape
    channels = 0
    if len(shape) == 2:
        depth, height = shape
        channels = 0
    else:
        depth, height, channels = shape
    # Create an empty array for the transformed image
    transformed_array = np.zeros_like(image_array)

    # Process each channel separately
    if channels == 0:
        channel_image = sitk.GetImageFromArray(image_array[:, :])
        channel_image.CopyInformation(image)
        transformed_channel = apply_transform(channel_image, reference_image,
                                              transform)

        # Store the transformed channel
        transformed_array[:, :] = sitk.GetArrayFromImage(
            transformed_channel)
    else:
        for c in range(channels):
            # Create a SimpleITK image for this channel
            channel_image = sitk.GetImageFromArray(image_array[:, :, c])
            channel_image.CopyInformation(image)

            # Apply transform to this channel
            transformed_channel = apply_transform(channel_image,
                                                  reference_image,
                                                  transform)

            # Store the transformed channel
            transformed_array[:, :, c] = sitk.GetArrayFromImage(
                transformed_channel)



    # Convert the transformed array back to a SimpleITK image
    transformed_image = sitk.GetImageFromArray(transformed_array, isVector=True)
    transformed_image.CopyInformation(reference_image)

    return transformed_image
def create_enhanced_overlay(reference, registered):
    # Compute absolute difference
    diff = np.abs(reference - registered)

    # Enhance contrast of the difference
    diff_enhanced = exposure.equalize_hist(diff)

    # Create RGB overlay
    overlay = np.zeros((*reference.shape, 3))
    overlay[:, :, 1] = reference  # Green channel for reference image
    overlay[:, :, 0] = registered  # Red channel for registered image

    # Enhance red channel based on differences
    overlay[:, :, 0] += diff_enhanced

    # Normalize the overlay image
    overlay = np.clip(overlay, 0, 1)

    return overlay

def SITK_Registration_Test():
    # Load images
    fixed_image = load_image('D:\\Thanu - ORL Regi\\New_Calibration\\base_flow.png')
    moving_image = load_image('D:\\Thanu - ORL Regi\\New_Calibration\\ref_flow.png')

    # Load thickness maps
    fixed_thickness = load_image('D:\\Thanu - ORL Regi\\New_Calibration\\base_flow_thic.tiff')
    moving_thickness = load_image('D:\\Thanu - ORL Regi\\New_Calibration\\ref_flow_thic.tif')

    # Perform registration on original images
    final_transform = register_images(fixed_image, moving_image)

    # Apply transform
    transformed_image = sitk.Resample(moving_image, fixed_image,
                                      final_transform, sitk.sitkLinear, 0.0,
                                      moving_image.GetPixelID())

    # Apply same transform to moving thickness map
    transformed_thickness = apply_transform(moving_thickness, fixed_thickness,
                                            final_transform)



    # Display results
    #display_images(fixed_image, moving_image, transformed_image)
    # Create enhanced RGB overlay
    npfixed = sitk.GetArrayFromImage(fixed_image)
    npmov = sitk.GetArrayFromImage(moving_image)
    nptrans = sitk.GetArrayFromImage(transformed_image)
    npthick = sitk.GetArrayFromImage(transformed_thickness)

    overlay = create_enhanced_overlay(npfixed, nptrans)
    # Display and save the results
    fig = plt.figure(figsize=(20, 15))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(np.uint8(npfixed), cmap='gray')
    ax1.set_title('Reference Image')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(npmov, cmap='gray')
    ax2.set_title('Target Image')
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(npthick, cmap='gray')
    ax3.set_title('Registered Image')
    ax3.axis('off')

    ax5 = fig.add_subplot(2, 2, 4)
    difference = np.abs(npfixed - 0.5*nptrans)
    ax5.imshow(difference, cmap='plasma')
    ax5.set_title('Difference Map')
    ax5.axis('off')
    plt.tight_layout()
    plt.savefig('D:\\Thanu - ORL Regi\\New_Calibration\\registration_result1.png', dpi=300, bbox_inches='tight')

    plt.close()
    plt.imshow(overlay)

    import tifffile
    tiffloc1 = ('D:\\Thanu - ORL Regi\\New_Calibration\\registration_result1Tiff_Regi_withmask.tif')
    tifffile.imwrite(tiffloc1, npthick.astype(np.float32))

def sitk_to_numpy(image, is_color=False):
    array = sitk.GetArrayFromImage(image)
    return array.squeeze()  # Remove single-dimensional entries

def check_equal_shape(map):
    if map.shape[0] != map.shape[1]:
        max_dim = max(map.shape[0], map.shape[1])
        if map.shape[0] < max_dim:
            # Add row(s) by duplicating the last row
            rows_to_add = max_dim - map.shape[0]
            map = np.vstack(
                [map, np.tile(map[-1:], (rows_to_add, 1))])
        else:
            # Add column(s) by duplicating the last column
            cols_to_add = max_dim - map.shape[1]
            map = np.hstack(
                [map, np.tile(map[:, -1:], (1, cols_to_add))])
    return map

def diffbetween(npthick, fix_thick):
    fixed_thick = np.array(Image.open(fix_thick), dtype=np.float32)
    npthick = np.array(npthick, dtype=np.float32)
    map1 = np.nan_to_num(npthick, nan=np.min(npthick))
    map2 = np.nan_to_num(fixed_thick, nan=np.min(fixed_thick))
    map1 = check_equal_shape(map1)
    map2 = check_equal_shape(map2)


    # Calculate the difference
    difference = map1 - map2
    actual_diff = difference.copy()

    # Calculate maximum absolute difference, ensuring it's not zero
    max_abs_diff = np.max(np.abs(difference))

    if max_abs_diff < 1e-10:
        normalized_diff = np.zeros_like(difference)
    else:
        normalized_diff = difference / max_abs_diff

        # Calculate threshold value
        threshold = 5 / 100.0

        # Create masks for different regions
        small_diff_mask = np.abs(normalized_diff) < threshold
        positive_mask = normalized_diff >= threshold
        negative_mask = normalized_diff <= -threshold

        # Set all small differences to exactly threshold value (with correct sign)
        normalized_diff[small_diff_mask & (normalized_diff >= 0)] = threshold
        normalized_diff[small_diff_mask & (normalized_diff < 0)] = -threshold

    # Clip the normalized values to [-1, 1] range
    normalized_diff = np.clip(normalized_diff, -1, 1)

    # Get the colormap
    try:
        colormap = plt.get_cmap('RdBu')
    except:
        # Fallback to a simple custom colormap
        colors = ['blue', 'green', 'red']
        nodes = [0.0, 0.5, 1.0]
        colormap = mcolors.LinearSegmentedColormap.from_list("custom", list(
            zip(nodes, colors)))

    # Create colored image using the colormap
    colored_diff = colormap(0.5 * (normalized_diff + 1.0))

    # Convert to 8-bit RGB
    colored_diff = (colored_diff[:, :, :3] * 255).astype(np.uint8)
    return colored_diff, actual_diff

def SITK_Registration(fixed_imagel, moving_imagel, fixed_thicknessl, moving_thicknessl, al1, al2):
    # Load images
    #fixed_image = load_image(fixed_imagel)
    #moving_image = load_image(moving_imagel)
    fixed_pil = Image.open(fixed_imagel).convert('L')
    fixed_array = np.array(fixed_pil)
    moving_pil = Image.open(moving_imagel).convert('L')
    moving_array = np.array(moving_pil)

    print(al1)
    print(al2)

    if al1 == 1:
        fixed_rot = np.rot90(fixed_array, k=1)
        fixed_final = np.flipud(fixed_rot)
        fixed_image = sitk.GetImageFromArray(fixed_final)
        print("hui")
        #fixed_image = sitk.Flip(fixed_image,[False, True])
    else:
        fixed_image = sitk.GetImageFromArray(fixed_array)

    if al2 == 1:
        mov_rot = np.rot90(moving_array, k=1)
        mov_final = np.flipud(mov_rot)
        moving_image = sitk.GetImageFromArray(mov_final)
        print("hui")
        #fixed_image = sitk.Flip(fixed_image,[False, True])
    else:
        moving_image = sitk.GetImageFromArray(moving_array)
        # 90 degree rotation
        #moving_image = sitk.Flip(moving_image,[False, True])

    #display_image(fixed_image)
    #display_image(moving_image)

    # Load thickness maps
    fixed_thickness = load_image(fixed_thicknessl)
    moving_thickness = load_image(moving_thicknessl)
    #display_image(fixed_thickness)
    #display_image(moving_thickness)

    # Perform registration on original images
    final_transform = register_images(fixed_image, moving_image)

    # Apply transform
    transformed_image = apply_transform(moving_image, fixed_image, final_transform)

    # Apply same transform to moving thickness map
    transformed_thickness = apply_transform(moving_thickness, fixed_thickness, final_transform)
    #display_image(transformed_image)
    #display_image(transformed_thickness)


    npfixed = sitk_to_numpy(fixed_image)
    npmov = sitk_to_numpy(moving_image)
    nptrans = sitk_to_numpy(transformed_image)
    npthick = sitk_to_numpy(transformed_thickness)
    print(f"Data type of color thickness map: {npthick.dtype}")
    return nptrans, npthick






if __name__ == "__main__":
    SITK_Registration_Test()