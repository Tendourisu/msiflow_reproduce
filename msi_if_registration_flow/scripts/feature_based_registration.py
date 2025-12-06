import argparse
import os
import tifffile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure, transform

def normalize_to_uint8(img):
    """Normalize image to 0-255 uint8 range for OpenCV."""
    if img.dtype == np.uint8:
        return img
    img_float = img.astype(float)
    # Robust normalization using percentiles
    p2, p98 = np.percentile(img_float, (2, 98))
    img_rescaled = exposure.rescale_intensity(img_float, in_range=(p2, p98))
    # Scale to 0-255
    img_uint8 = (img_rescaled * 255).astype(np.uint8)
    return img_uint8

def feature_registration(fixed_img_path, moving_img_path, af_chan, out_file_path, method='ORB', plot=False):
    # 1. Read Images
    print(f"Reading Fixed: {fixed_img_path}")
    fixed_img = tifffile.imread(fixed_img_path)
    
    print(f"Reading Moving: {moving_img_path}")
    moving_img_stack = tifffile.imread(moving_img_path)
    # Adjust 0-based index
    af_chan_idx = af_chan - 1
    moving_img = moving_img_stack[af_chan_idx]

    # 2. Preprocess for Feature Detection (Normalize to uint8)
    fixed_uint8 = normalize_to_uint8(fixed_img)
    moving_uint8 = normalize_to_uint8(moving_img)

    # 3. Feature Detection
    if method == 'ORB':
        # ORB is fast and rotation invariant
        detector = cv2.ORB_create(nfeatures=2000)
    elif method == 'SIFT':
        # SIFT is more robust but slower (patented in some cv2 versions, but usually ok now)
        detector = cv2.SIFT_create()
    elif method == 'AKAZE':
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Detecting features using {method}...")
    kp1, des1 = detector.detectAndCompute(fixed_uint8, None)
    kp2, des2 = detector.detectAndCompute(moving_uint8, None)

    if des1 is None or des2 is None:
        raise RuntimeError("No features detected in one of the images!")
    
    print(f"Found {len(kp1)} features in Fixed, {len(kp2)} features in Moving.")

    # 4. Feature Matching
    if method == 'ORB':
        # Hamming distance for binary descriptors (ORB, AKAZE)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
    else:
        # L2 norm for SIFT/SURF
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Keep top matches
    # good_matches = matches[:int(len(matches) * 0.5)]
    good_matches = matches # RANSAC handles outliers well, pass all reasonably good ones?
    print(f"Using {len(good_matches)} matches for estimation.")

    if len(good_matches) < 4:
        raise RuntimeError("Not enough matches found to compute homography.")

    # 5. Compute Homography / Affine Transform
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) # Moving
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) # Fixed

    # Use RANSAC to filter outliers
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is None:
        raise RuntimeError("Homography estimation failed.")
        
    print("Transformation Matrix computed.")

    # 6. Apply Transformation to ALL channels of Moving Image
    # Output shape should match Fixed Image
    h, w = fixed_img.shape
    
    registered_stack = np.zeros((moving_img_stack.shape[0], h, w), dtype=moving_img_stack.dtype)
    
    print("Warping image stack...")
    for i in range(moving_img_stack.shape[0]):
        # Warp using the calculated matrix M
        # Note: cv2.warpPerspective expects (width, height) for dsize
        warped = cv2.warpPerspective(moving_img_stack[i], M, (w, h))
        registered_stack[i] = warped

    # 7. Save Output
    out_dir = os.path.dirname(out_file_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    tifffile.imwrite(out_file_path, registered_stack, photometric='minisblack')
    print(f"Saved registered stack to {out_file_path}")
    
    # 8. Plot matches for debugging
    if plot:
        img_matches = cv2.drawMatches(fixed_uint8, kp1, moving_uint8, kp2, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Overlay Result
        warped_moving_disp = normalize_to_uint8(registered_stack[af_chan_idx])
        overlay = np.dstack((fixed_uint8, warped_moving_disp, np.zeros_like(fixed_uint8)))
        
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.title(f"Matches ({method})")
        plt.imshow(img_matches)
        plt.subplot(1, 2, 2)
        plt.title("Result Overlay (R=Fixed, G=Moving)")
        plt.imshow(overlay)
        plt.savefig(os.path.join(out_dir, f"feature_matching_debug_{method}.png"))
        # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs feature-based registration')
    parser.add_argument('fixed_img', type=str, help='path to fixed UMAP image')
    parser.add_argument('moving_img', type=str, help='path to moving image tif stack')
    parser.add_argument('-af_chan', type=int, default=1, help='autofluorescence image channel (1-based)')
    parser.add_argument('-out_file', type=str, required=True, help='registered output file path')
    parser.add_argument('-method', type=str, default='ORB', help='Feature detector: ORB, SIFT, AKAZE')
    parser.add_argument('-plot', action='store_true', help='Generate debug plots')
    
    args = parser.parse_args()

    feature_registration(args.fixed_img, args.moving_img, args.af_chan, args.out_file, args.method, args.plot)








