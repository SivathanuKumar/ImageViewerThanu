import numpy as np
import cv2
from scipy.signal import medfilt
import os
from pathlib import Path
from tqdm import tqdm
from OCTpy import Oct3Dimage
from scipy import integrate
import tkinter as tk
from tkinter import filedialog, ttk
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from scipy.interpolate import make_interp_spline


add = os.getcwd()
img_obj = Oct3Dimage()

def get_parameter_sets():
    """
    Returns dictionary of all parameter sets
    Window Size: first layer segmentation median filter adjustment
    Contrast Height: Region where the intensity graph is checked. Dont try to include the choroid into it
    fromtop: Distance from the first line to the RPE, to disregard the retinal vessels
    searchupperline:
    searchvalueupperline:
    medfiltadjupp/low/top: Median filter for the other two functions
    GaussGauss: Gausian filter for the first function
    Gausstop/low: Gausian filter for the other two functions
    n_edge_ot: edge correction for the bottom two lines
    """
    return {
        "Zeiss 6x6 SS - OCT": {
            'window_size': 11,
            'contrast_height': 180,
            'fromtop': 60,
            'searchupperline': 40,
            'searchvalueupperline': 6,
            'medfiltadjupp': 17,
            'medfiltadjlow': 17,
            'medfiltadjtop': 97,
            'n_edge_ot': 40,
            'gaussgauss': 0.0,
            'gaussupp': 0.0,
            'gausslow': 0.0,
            'gausstop': 0.0,
            'call': 1
        },
        "Zeiss 6x6 SD - OCT": {
            'window_size': 17,
            'contrast_height': 180,
            'fromtop': 60,
            'searchupperline': 40,
            'searchvalueupperline': 6,
            'medfiltadjupp': 37,
            'medfiltadjlow': 37,
            'medfiltadjtop': 97,
            'n_edge_ot': 40,
            'gaussgauss': 0.0,
            'gaussupp': 0.0,
            'gausslow': 0.0,
            'gausstop': 0.0,
            'call': 1
        },
        "Intalight 6x6 3mm depth": {
            'window_size': 21,
            'contrast_height': 115,
            'fromtop': 40,
            'searchupperline': 40,
            'searchvalueupperline': 6,
            'medfiltadjupp': 47,
            'medfiltadjlow': 47,
            'medfiltadjtop': 97,
            'n_edge_ot': 8,
            'gaussgauss': 0.0,
            'gaussupp': 0.0,
            'gausslow': 0.0,
            'gausstop': 0.0,
            'call': 2
        },
        "Heidelburg (Alpha)": {
            'window_size': 15,
            'contrast_height': 100,
            'fromtop': 60,
            'searchupperline': 40,
            'searchvalueupperline': 6,
            'medfiltadjupp': 55,
            'medfiltadjlow': 85,
            'medfiltadjtop': 55,
            'n_edge_ot': 8,
            'gaussgauss': 1.0,
            'gaussupp': 5.5,
            'gausslow': 5.5,
            'gausstop': 5.0
        },
        "Topcon (Alpha)": {
            'window_size': 15,
            'contrast_height': 100,
            'fromtop': 60,
            'searchupperline': 40,
            'searchvalueupperline': 6,
            'medfiltadjupp': 55,
            'medfiltadjlow': 85,
            'medfiltadjtop': 55,
            'n_edge_ot': 8,
            'gaussgauss': 1.0,
            'gaussupp': 5.5,
            'gausslow': 5.5,
            'gausstop': 5.0
        }
    }

def select_parameter_set():
    """
    Opens a dialog with dropdown for parameter set selection
    Returns the selected parameter set
    """
    dialog = tk.Tk()  # Make this the main window instead of a Toplevel
    dialog.title("Select Processing Parameters")
    dialog.geometry("300x150")

    parameter_sets = get_parameter_sets()

    # Center the dialog on screen
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f'{width}x{height}+{x}+{y}')

    tk.Label(dialog, text="Choose processing parameters:").pack(pady=10)
    combo = ttk.Combobox(dialog, values=list(parameter_sets.keys()),
                         state="readonly")
    combo.set(list(parameter_sets.keys())[0])
    combo.pack(pady=10)

    result = {"selection": None}

    def on_select():
        result["selection"] = parameter_sets[combo.get()]
        dialog.quit()

    def on_cancel():
        dialog.quit()

    button_frame = tk.Frame(dialog)
    button_frame.pack(pady=10)
    tk.Button(button_frame, text="Select", command=on_select).pack(side=tk.LEFT,
                                                                   padx=5)
    tk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT,
                                                                   padx=5)

    dialog.protocol("WM_DELETE_WINDOW", on_cancel)
    dialog.mainloop()

    dialog.destroy()
    return result["selection"]

def quick_process_video(img_obj):
    processing_params = select_parameter_set()
    if processing_params is None:
        print("No parameter set selected")
        exit()
    """
    Process video and save combined layers and visualizations
    Args:
        video_path: Path to input .avi video
        output_path: Path to save output files
    """
    dicom_data = img_obj.stru3d
    print("running dataset")
    OAC_conv = OAC_calculation(dicom_data)
    max_val = np.nanpercentile(OAC_conv, 99)
    OAC_conv[OAC_conv > max_val] = max_val
    OAC_conv = OAC_conv / max_val * 255
    img = OAC_conv
    height, width, num_frames = img.shape

    layers_fast = np.zeros((width, num_frames, 4))

    # Process each frame
    print(f"Processing {num_frames} frames...")

    for frame_idx in tqdm(range(num_frames)):
        current_frame = img[:, :, frame_idx].astype(np.uint8)
        processed_img, initial_line, upper_line, lower_line, top_line = process_single_image_better(
            current_frame, processing_params['window_size'],
            processing_params['contrast_height'],
            processing_params['fromtop'],
            processing_params['searchupperline'],
            processing_params['searchvalueupperline'],
            processing_params['medfiltadjupp'],
            processing_params['medfiltadjlow'],
            processing_params['medfiltadjtop'],
            processing_params['call'],
            processing_params['n_edge_ot'])

        layers_fast[:, frame_idx, 0] = initial_line
        layers_fast[:, frame_idx, 2] = upper_line
        layers_fast[:, frame_idx, 3] = lower_line
        layers_fast[:, frame_idx, 1] = top_line
        layers_fast[:, :, 0] = gaussian_filter(layers_fast[:, :, 0], sigma=processing_params['gaussgauss'])
        layers_fast[:, :, 2] = gaussian_filter(layers_fast[:, :, 2], sigma=processing_params['gaussupp'])
        layers_fast[:, :, 3] = gaussian_filter(layers_fast[:, :, 3], sigma=processing_params['gausslow'])
        layers_fast[:, :, 1] = gaussian_filter(layers_fast[:, :, 1], sigma=processing_params['gausstop'])

    print("Processing complete!")
    layers_fast[:, num_frames - 1, :] = layers_fast[:, num_frames - 2, :]
    layers_fast[:, 0, :] = layers_fast[:, 1, :]
    layers = edge_correct(layers_fast, processing_params['n_edge_ot'])
    return layers

def process_single_image_better(img, window_size=11, contrast_height=180, fromtop = 50, searchupperline = 40, searchvalueupperline = 6, medfiltadjupp = 17, medfiltadjlow = 17, medfiltadjtop = 57, call = 1, n_edge_ot = 5):
    """
    Process a single grayscale image with contrast enhancement and line detection

    Args:
        img (numpy.ndarray): Input grayscale image
        window_size (int): Size of the median filter window for initial filtering
        contrast_height (int): Number of pixels below the line for contrast enhancement
        line_filter_size (int): Size of the median filter for detected lines
    """
    height, width = img.shape
    line_points = []
    for x in range(width):
        column = img[:, x]
        threshold_points = np.where(column > 4)[0]
        if len(threshold_points) > 0:
            line_points.append((x, threshold_points[0]))
        else:
            if line_points:
                line_points.append((x, line_points[-1][1]))
            else:
                line_points.append((x, 0))
    line_points = np.array(line_points)
    filtered_y = medfilt(line_points[:, 1], window_size)
    line_points[:, 1] = filtered_y

    # Edge correction for initial line
    n_edge = 5
    valid_start = np.median(line_points[n_edge:2 * n_edge, 1])
    line_points[:n_edge, 1] = valid_start
    valid_end = np.median(line_points[-2 * n_edge:-n_edge, 1])
    line_points[-n_edge:, 1] = valid_end
    processed_img = img.copy()

    # Arrays to store the detected lines
    upper_line = np.zeros(width)
    lower_line = np.zeros(width)
    top_line = np.zeros(width)

    for x in range(width):
        y_line = int(line_points[x, 1])
        roi_start = y_line + fromtop
        roi_end = min(y_line + contrast_height, height)

        if roi_start < roi_end:
            roi = processed_img[roi_start:roi_end, x].copy()
            if len(roi) > 0:
                # Apply linear contrast enhancement
                alpha = 1.5
                beta = 0
                roi = roi.astype(np.float32)
                roi_enhanced = np.clip(alpha * roi + beta, 0, 255).astype(
                    np.uint8)

                # Find the highest intensity point
                max_intensity_idx = np.argmax(roi_enhanced)

                # Set upper line 4 pixels above the highest intensity point
                upper_idx = max_intensity_idx - 4
                upper_idx = max(0,
                                upper_idx)  # Ensure we don't go out of bounds

                # Find the next valley after the peak
                valley_idx = max_intensity_idx
                for i in range(max_intensity_idx + 1, len(roi_enhanced) - 1):
                    # Check if current point is a local minimum
                    if (roi_enhanced[i - 1] > roi_enhanced[i] and
                            roi_enhanced[i] <= roi_enhanced[i + 1]):
                        valley_idx = i
                        break

                # Adjust indices to image coordinates
                upper_line[x] = roi_start + upper_idx
                lower_line[x] = roi_start + valley_idx + 4

                processed_img[roi_start:roi_end, x] = roi_enhanced

        if roi_end < height:
            processed_img[roi_end:, x] = 0

    # After determining upper_line, search upward for the top line
    for x in range(width):
        # Get the upper line position for this column
        upper_y = int(upper_line[x]) - searchupperline

        # Search upward from upper_line position
        for y in range(upper_y - 1, -1,
                       -1):  # Start from one pixel above upper_line
            if img[y, x] > searchvalueupperline:
                top_line[x] = y
                break
            if y == 0:  # If we reach the top without finding a point
                if x > 0:  # Use previous column's value if available
                    top_line[x] = top_line[x - 1]
                else:
                    top_line[x] = 0

    # Apply median filtering to the detected lines
    upper_line = medfilt(upper_line, medfiltadjupp)
    lower_line = medfilt(lower_line, medfiltadjlow)
    top_line = medfilt(top_line, medfiltadjtop)  # Apply filtering to top line

    # Edge correction for detected lines
    #def correct_edge_effects(line, n_edge=5):
    #    valid_start_value = np.median(line[n_edge:2 * n_edge])
    #    line[:n_edge] = valid_start_value
    #    valid_end_value = np.median(line[-2 * n_edge:-n_edge])
    #    line[-n_edge:] = valid_end_value
    #    return line

    #upper_line = edge_correct(upper_line, n_edge_ot)
    #lower_line = edge_correct(lower_line, n_edge_ot)
    #top_line = edge_correct(top_line, n_edge_ot)

    return processed_img, line_points[:, 1].astype(int), upper_line.astype(
        int), lower_line.astype(int), top_line.astype(int)

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

def save_visualization(original_img, processed_img, initial_line, upper_line,
                       lower_line, top_line, output_path, filename):
    output_path = Path(output_path)
    processed_with_lines = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    width = original_img.shape[1]
    for x in range(width - 1):
        # Initial line (green)
        cv2.line(processed_with_lines,
                 (x, int(initial_line[x])),
                 (x + 1, int(initial_line[x + 1])),
                 (0, 255, 0), 1)
        # Upper line (red)
        cv2.line(processed_with_lines,
                 (x, upper_line[x]),
                 (x + 1, upper_line[x + 1]),
                 (0, 0, 255), 1)
        # Lower line (blue)
        cv2.line(processed_with_lines,
                 (x, lower_line[x]),
                 (x + 1, lower_line[x + 1]),
                 (255, 0, 0), 1)
        cv2.line(processed_with_lines,
                 (x, top_line[x]),
                 (x + 1, top_line[x + 1]),
                 (255, 255, 0), 1)
    cv2.imwrite(str(output_path / f'segmented_{filename}'),
                processed_with_lines)


def edge_correct(layers, n_edge):

    corrected = layers.copy()
    for layer_idx in range(layers.shape[2]):
        for col in range(layers.shape[1]):
            column_values = layers[:, col, layer_idx]
            # Get values from 40-80 and calculate the trend
            normal_section = column_values[n_edge:2 * n_edge]
            x_normal = np.arange(len(normal_section))
            # Create a smooth spline fit of the normal section
            spl = make_interp_spline(x_normal, normal_section,
                                     k=2)  # k=3 for cubic spline

            # Generate points for extrapolation (going backwards)
            x_extrap = np.arange(n_edge,
                                 0)  # negative indices for extrapolation

            # Get derivatives at the start of normal section
            derivatives = spl.derivative()

            # Use the derivatives to extrapolate backwards
            predicted_trend = np.zeros(n_edge)
            current_value = normal_section[0]

            for i in range(n_edge - 1, -1, -1):
                # Get the derivative at this point
                dx = derivatives(0)  # derivative at the start
                # Update the value using the derivative
                predicted_trend[i] = current_value
                current_value = current_value - dx  # step backwards using derivative

            try:
                # Sanity check bounds
                max_normal = np.max(normal_section)
                min_normal = np.min(normal_section)
                margin = (max_normal - min_normal) * 0.3
                predicted_trend = np.clip(predicted_trend, min_normal - margin,
                                          max_normal + margin)

                # Calculate adaptive threshold based on normal section
                threshold = np.std(normal_section) * 3

                # Smooth transition for corrections
                for i in range(40):
                    difference = abs(column_values[i] - predicted_trend[i])
                    weight = np.clip(difference / threshold, 0, 1)
                    corrected[i, col, layer_idx] = (1 - weight) * column_values[
                        i] + weight * predicted_trend[i]

            except (ValueError, np.linalg.LinAlgError) as e:
                print(
                    f"Warning: Failed for layer {layer_idx}, column {col}. Error: {e}")
                continue
    return corrected

if __name__ == "__main__":

    '''
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Video to Segment",
        filetypes=[("Image files", "*.avi *.dcm *.tif *.tiff *.img")]
    )

    if not video_path:
        print("No file selected")
    else:
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(os.path.dirname(video_path), base_filename)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    try:
        layers = quick_process_video_test(video_path, output_path, img_obj)

    except Exception as e:
        print(f"Error processing folder: {e}")
    '''


