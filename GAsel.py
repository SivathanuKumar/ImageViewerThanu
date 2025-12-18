import cv2
import numpy as np
import os


def remove_small_objects(binary_img, min_size):
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8)

    # Create a new image to store the result
    result = np.zeros_like(binary_img)

    # Loop through all components (excluding the background which is label 0)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            result[labels == label] = 255

    return result
def detect_ga_area(image_path, otsu_bias=0, min_object_size=100):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply Otsu's thresholding
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply bias to Otsu's threshold
    final_thresh = otsu_thresh + otsu_bias

    # Apply the biased threshold
    _, thresh = cv2.threshold(blurred, final_thresh, 255, cv2.THRESH_BINARY)

    thresh_cleaned = remove_small_objects(thresh, min_object_size)

    # Find contours
    contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Calculate total area of all contours in pixels
    total_area_pixels = sum(cv2.contourArea(contour) for contour in contours)

    # Calculate the conversion factor
    # Image size: 500x500 pixels
    # Scanning area: 6mm x 6mm
    pixels_per_mm = 500 / 6  # pixels per mm
    mm2_per_pixel = (1 / pixels_per_mm) ** 2  # mm² per pixel

    # Convert pixel area to mm²
    total_area_mm2 = total_area_pixels * mm2_per_pixel

    # Create a binary mask
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (255),
                     -1)  # Fill all contours with white

    # Draw all contours on the original image
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)

    # Get the directory and filename of the source image
    dir_path = os.path.dirname(image_path)
    file_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save the result image
    result_path = os.path.join(dir_path, f"{file_name}_result.jpg")
    cv2.imwrite(result_path, img_color)

    # Save the binary mask
    mask_path = os.path.join(dir_path, f"{file_name}_ga_mask.png")
    cv2.imwrite(mask_path, mask)

    return total_area_pixels, total_area_mm2, mask, result_path, mask_path


# Usage
#image_path = 'C:\\Users\\BAILGPU1\\Desktop\\results\\Patient2399 032718\\noc1.png'
#area_pixels, area_mm2, binary_mask, result_file, mask_file = detect_ga_area(image_path, 10, 100)
#print(
#    f"The total area of Geographic Atrophy is {area_pixels:.2f} pixels or {area_mm2:.2f} mm².")
#print(f"The result image has been saved as '{result_file}'.")
#print(f"The binary mask has been saved as '{mask_file}'.")