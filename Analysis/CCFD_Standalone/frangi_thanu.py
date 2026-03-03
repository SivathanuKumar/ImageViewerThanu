import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology
import os


def process_octa_vessels(image_path, show_plot=True):
    """
    Reads an OCTA image, applies Frangi vesselness to isolate large vessels,
    thresholds the result, and cleans up the binary mask.
    """
    # 1. Load the Image
    print(f"Loading image: {image_path}")
    image = io.imread(image_path, as_gray=True)

    # ---------------------------------------------------------
    # STEP 1: Frangi Vesselness Filter
    # ---------------------------------------------------------
    # Parameters matched to your Fiji workflow:
    # Scales=4, Min=3.0, Max=8.0
    sigmas = np.linspace(3.5, 10.0, 4)

    print(f"Running Frangi filter with sigmas: {sigmas}...")
    vesselness_map = filters.frangi(image, sigmas=sigmas, black_ridges=False)

    # ---------------------------------------------------------
    # STEP 2: Thresholding
    # ---------------------------------------------------------
    # Threshold at 9% of the maximum intensity found in the map
    threshold_value = 0.12 * vesselness_map.max()
    binary_mask = vesselness_map > threshold_value
    print(f"Threshold applied (Value: {threshold_value:.4f})")

    # ---------------------------------------------------------
    # STEP 3: Post-Processing
    # ---------------------------------------------------------
    # A. Join disjointed regions (Closing)
    # Radius 2 bridges small gaps (~4 pixels wide)
    closing_radius = 2
    footprint = morphology.disk(closing_radius)
    bridged_mask = morphology.binary_closing(binary_mask, footprint)

    # B. Remove standalone noise (Area Opening)
    # Removes objects smaller than 500 pixels
    min_size_pixels = 500
    clean_mask = morphology.remove_small_objects(bridged_mask, min_size=min_size_pixels)
    print("Morphological cleaning complete.")

    # ---------------------------------------------------------
    # Visualization (Optional)
    # ---------------------------------------------------------
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original En Face OCTA')

        axes[1].imshow(vesselness_map, cmap='magma')
        axes[1].set_title('Frangi Vesselness Map')

        axes[2].imshow(clean_mask, cmap='gray')
        axes[2].set_title('Final Large Vessel Mask')

        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.show()

    return clean_mask


# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Change this path to point to your actual image file
    INPUT_FILE = "/home/skthanu/Documents/Dancing_Lion/Thanu_SSvsSD/SS/Thanu_Zeiss_Plex/PUW123 20251223/PUW123_Angio (6mmx6mm)_12-23-2025_16-30-40_OD_sn17004_FlowCube_z/projs_flow_1.png"

    # Check if file exists before running
    if not os.path.exists(INPUT_FILE):
        print(f"Error: The file '{INPUT_FILE}' was not found.")
        print("Please update the 'INPUT_FILE' variable in the script.")
    else:
        # Run processing
        final_mask = process_octa_vessels(INPUT_FILE, show_plot=True)

        # Save the result
        output_filename = "large_vessels_mask.png"

        # Convert boolean mask (True/False) to uint8 (0/255) for saving
        io.imsave(output_filename, (final_mask * 255).astype(np.uint8))
        print(f"\nSuccess! Processed mask saved to: {output_filename}")