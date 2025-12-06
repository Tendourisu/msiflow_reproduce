import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import exposure, transform
from skimage.util import img_as_float

def normalize(img):
    """Normalize image to 0-1 range for visualization."""
    img = img_as_float(img)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img

def create_checkerboard(im1, im2, n_squares=8):
    """Create a checkerboard image from two images."""
    assert im1.shape == im2.shape, f"Images must have same shape, got {im1.shape} and {im2.shape}"
    
    rows, cols = im1.shape
    row_step = rows // n_squares
    col_step = cols // n_squares
    
    out = np.zeros_like(im1)
    
    for r in range(n_squares):
        for c in range(n_squares):
            r_start = r * row_step
            r_end = (r + 1) * row_step if r < n_squares - 1 else rows
            c_start = c * col_step
            c_end = (c + 1) * col_step if c < n_squares - 1 else cols
            
            if (r + c) % 2 == 0:
                out[r_start:r_end, c_start:c_end] = im1[r_start:r_end, c_start:c_end]
            else:
                out[r_start:r_end, c_start:c_end] = im2[r_start:r_end, c_start:c_end]
    return out

def create_overlay(im1, im2):
    """Create a red-green overlay image."""
    # im1 (Fixed) -> Red
    # im2 (Moving/Reg) -> Green
    
    # Handle shapes if they are 3D (multi-channel)
    if im1.ndim == 3: im1 = np.mean(im1, axis=0)
    if im2.ndim == 3: im2 = np.mean(im2, axis=0)

    # Normalize
    im1_n = normalize(im1)
    im2_n = normalize(im2)
    
    # Create RGB
    rows, cols = im1_n.shape
    out = np.zeros((rows, cols, 3), dtype=np.float32)
    
    out[..., 0] = im1_n # Red
    out[..., 1] = im2_n # Green
    out[..., 2] = 0     # Blue
    
    return out

import sys

def main():
    # Usage: python check_registration_quality.py [reg_file_path] [label]
    
    # Paths (Hardcoded for the demo case)
    base_dir = "demo/data/msi_if_registration"
    fixed_path = os.path.join(base_dir, "fixed/umap/umap_grayscale_UPEC_12.tif")
    
    # Default to original
    reg_path = os.path.join(base_dir, "registered/UPEC_12.tif")
    label = "baseline"

    if len(sys.argv) > 1:
        reg_path = sys.argv[1]
    if len(sys.argv) > 2:
        label = sys.argv[2]
    
    out_dir = "visualization_results"
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Reading Fixed: {fixed_path}")
    fixed_img = tifffile.imread(fixed_path)
    
    print(f"Reading Registered ({label}): {reg_path}")
    reg_img = tifffile.imread(reg_path)
    
    # ... rest of the code ...


    # If Reg is multi-channel (C, H, W) or (H, W, C), we need to handle it.
    # Tifffile usually reads as (C, H, W) for ImageJ stacks.
    reg_channel = reg_img
    if reg_img.ndim == 3:
        # Assuming channel 0 is relevant for structure, or try to merge.
        # Let's try to find which channel matches best or just use channel 0 (usually DAPI or AF).
        # In config.yaml, 'af_chan' is specified. Let's assume channel 0 for now.
        print("Registered image is multi-channel. Using Channel 0 for visualization.")
        reg_channel = reg_img[0, :, :]
        
    # Ensure shapes match exactly
    if fixed_img.shape != reg_channel.shape:
        print("Warning: Shapes do not match exactly. Resizing Fixed to match Registered.")
        fixed_img = transform.resize(fixed_img, reg_channel.shape, preserve_range=True)

    # 1. Checkerboard
    print("Generating Checkerboard...")
    checker = create_checkerboard(fixed_img, reg_channel, n_squares=10)
    plt.imsave(os.path.join(out_dir, f"comparison_checkerboard_{label}.png"), checker, cmap='gray')
    
    # 2. Overlay
    print("Generating RGB Overlay...")
    overlay = create_overlay(fixed_img, reg_channel)
    plt.imsave(os.path.join(out_dir, f"comparison_overlay_{label}.png"), overlay)
    
    print(f"Done! Check {out_dir} for results.")

if __name__ == "__main__":
    main()

