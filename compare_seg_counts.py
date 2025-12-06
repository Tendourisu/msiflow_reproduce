import tifffile
import numpy as np
from skimage.measure import label

def count_objects(mask_path, chan=1):
    img = tifffile.imread(mask_path)
    # Handle multi-channel: take the segmented channel (index = chan - 1)
    if img.ndim == 3:
        img = img[chan-1]
    
    # Binary mask
    binary = img > 0
    
    # Label connected components
    labeled_img, num_features = label(binary, return_num=True)
    return num_features

def main():
    orig_path = "demo/data/if_segmentation/segmented/UPEC_12.tif"
    watershed_path = "demo/data/if_segmentation/segmented/UPEC_12_watershed_v2.tif"
    
    print("Counting objects (cells)...")
    
    # Check if original exists (it might not if we didn't run the original seg flow manually yet)
    # Let's assume user wants me to run it if missing, but let's just check first.
    try:
        n_orig = count_objects(orig_path, chan=2)
        print(f"Original Otsu: {n_orig} objects")
    except FileNotFoundError:
        print(f"Original Otsu file not found at {orig_path}")

    try:
        n_water = count_objects(watershed_path, chan=2)
        print(f"Watershed:     {n_water} objects")
    except FileNotFoundError:
        print("Watershed file not found!")

if __name__ == "__main__":
    main()

