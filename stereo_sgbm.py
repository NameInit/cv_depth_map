import numpy as np
import cv2

def calc_depth_map(image_left, image_right, baseline_x_cm=8.0, baseline_y_cm=0.0, focal_length_px=441.25, max_distance = 12.0):
    baseline_cm = np.sqrt(baseline_x_cm**2 + baseline_y_cm**2)
    
    if len(image_left.shape) == 3:
        left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = image_left.copy()
        right_gray = image_right.copy()
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    left_gray = clahe.apply(left_gray)
    right_gray = clahe.apply(right_gray)
    
    window_size = 7
    min_disp = 0
    num_disp = 160
    
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 1 * window_size**2, 
        P2=32 * 1 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,       
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    try:
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(8000.0)
        wls_filter.setSigmaColor(1.5)
        
        disp_left = left_matcher.compute(left_gray, right_gray)
        disp_right = right_matcher.compute(right_gray, left_gray)
        
        filtered_disp = wls_filter.filter(np.int16(disp_left), left_gray, None, np.int16(disp_right))
        disparity = filtered_disp.astype(np.float32) / 16.0
    except AttributeError:
        disparity = left_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disparity = cv2.medianBlur(disparity, 5)

    disparity[disparity <= 0.5] = 0.5

    baseline_m = baseline_cm / 100.0
    distance_m = (focal_length_px * baseline_m) / disparity

    distance_m = np.clip(distance_m, 0, max_distance)

    norm = 255 - np.uint8((distance_m / max_distance) * 255)

    result_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return result_color, distance_m, norm

def distance_from_norm_pixel(pixel_value, max_distance=12.0):
    distance = ((255 - pixel_value) / 255.0) * max_distance
    return distance

def exact_distance_at_coords(x, y, distance_matrix):
    h, w = distance_matrix.shape

    if 0 <= y < h and 0 <= x < w:
        return float(distance_matrix[y, x])
    return 0.0