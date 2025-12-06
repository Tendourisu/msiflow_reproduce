import argparse
import tifffile
import os
import sys
import numpy as np
from skimage.morphology import binary_closing, remove_small_objects
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from pkg import utils

def segment_watershed(img, sigma, min_size, bin_closing_size, min_distance=10):
    # 1. Normalize
    img = utils.NormalizeData(img)

    # 2. Gaussian smoothing
    if sigma > 0:
        img_smooth = gaussian(img, sigma=sigma)
    else:
        img_smooth = img

    # 3. Thresholding (Generate Binary Mask)
    thr = threshold_otsu(img_smooth)
    binary_mask = img_smooth > thr

    # 4. Cleanup (Remove small objects & Closing)
    binary_mask = remove_small_objects(binary_mask, min_size)
    if bin_closing_size > 0:
        binary_mask = binary_closing(binary_mask, footprint=np.ones((bin_closing_size, bin_closing_size)))

    # 5. Watershed Logic
    # Calculate distance transform (distance from background)
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # Find peaks (seeds for watershed)
    # min_distance controls how close two cells can be.
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    
    # Run Watershed
    # -distance because watershed looks for basins (minima), but we have peaks (maxima)
    labels = watershed(-distance, markers, mask=binary_mask)

    # 6. Convert labels back to binary mask (or keep as labels if we want instance seg)
    # For compatibility with current pipeline (which expects binary mask), we convert back to 0/255.
    # Note: This loses the instance information (id 1 vs id 2), but separates the blobs spatially if there is a gap.
    # Watershed typically leaves a 1-pixel line between touched objects.
    
    # Convert to uint8 binary mask (0 or 255)
    result_mask = (labels > 0).astype('uint8') * 255
    
    return result_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create binary image using Watershed')
    parser.add_argument('input', type=str, help='input tif image (stack)')
    parser.add_argument('-output', type=str, default='', help='output file to save segmented image')
    parser.add_argument('-sigma', type=int, default=1, help='sigma for Gaussian smoothing')
    parser.add_argument('-min_size', type=int, default=10, help='all objects below this size will be removed')
    parser.add_argument('-bin_closing_size', type=int, default=0, help='structuring element size')
    parser.add_argument('-min_distance', type=int, default=5, help='min distance between peaks for watershed')
    parser.add_argument('-chan_to_seg_list', type=lambda s: [int(item)-1 for item in s.split(',')], default=[],
                        help='pass delimited list of image channels to segment')
    args = parser.parse_args()

    if args.output == '':
        out_dir = os.path.abspath(os.path.join(os.path.dirname(args.input), 'segmented_watershed'))
    else:
        out_dir = os.path.dirname(args.output)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # read data
    img = tifffile.imread(args.input)

    # segment image stack
    if img.squeeze().ndim > 2:
        output_stack = np.zeros(img.shape, dtype='uint8')
        for i in range(img.shape[0]):
            if i in args.chan_to_seg_list:
                print(f"Segmenting channel {i+1} with Watershed...")
                output_stack[i] = segment_watershed(img[i], args.sigma, args.min_size, args.bin_closing_size, args.min_distance)
            else:
                # Keep original channel but cast to uint8 if needed (usually raw is uint16/float)
                # To be safe for mask overlay, we might want to zero it out or normalize.
                # Following original script behavior: copy original.
                norm_orig = utils.NormalizeData(img[i])
                output_stack[i] = (norm_orig * 255).astype('uint8')
    else:
        output_stack = segment_watershed(img, args.sigma, args.min_size, args.bin_closing_size, args.min_distance)

    # write segmented output
    tifffile.imwrite(args.output, output_stack, photometric='minisblack')








