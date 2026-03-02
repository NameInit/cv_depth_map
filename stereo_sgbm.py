import cv2
import numpy as np
from config import Config

class StereoSGBM:
    def __init__(self, scale=1.0, use_wls=True):
        self.scale = scale
        self.use_wls = use_wls
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        raw_num_disp = int(Config.SGBM_NUM_DISP * scale)
        self.num_disp = raw_num_disp - (raw_num_disp % 16)
        if self.num_disp < 16: self.num_disp = 16
        
        window_size = Config.SGBM_WINDOW_SIZE
        
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disp,
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
        
        self.has_ximgproc = hasattr(cv2, 'ximgproc')
        if self.use_wls and self.has_ximgproc:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
            self.wls_filter.setLambda(8000.0)
            self.wls_filter.setSigmaColor(1.5)
        else:
            self.use_wls = False

    def compute(self, left_img, right_img, focal_length=None, baseline_cm=None):
        if focal_length is None: focal_length = Config.FOCAL_LENGTH
        if baseline_cm is None: baseline_cm = Config.BASELINE_CM

        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
            
        if self.scale != 1.0:
            h, w = left_gray.shape[:2]
            new_size = (int(w * self.scale), int(h * self.scale))
            left_small = cv2.resize(left_gray, new_size, interpolation=cv2.INTER_LINEAR)
            right_small = cv2.resize(right_gray, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            left_small = left_gray
            right_small = right_gray

        left_eq = self.clahe.apply(left_small)
        right_eq = self.clahe.apply(right_small)

        if self.use_wls:
            disp_left = self.left_matcher.compute(left_eq, right_eq)
            disp_right = self.right_matcher.compute(right_eq, left_eq)
            filtered_disp = self.wls_filter.filter(disp_left, left_eq, None, disp_right)
            disparity = filtered_disp.astype(np.float32) / 16.0
        else:
            disp_left = self.left_matcher.compute(left_eq, right_eq)
            disparity = disp_left.astype(np.float32) / 16.0
            disparity = cv2.medianBlur(disparity, 5)

        if self.scale != 1.0:
            disparity = cv2.resize(disparity, (left_img.shape[1], left_img.shape[0]), interpolation=cv2.INTER_LINEAR)
            disparity = disparity * (1.0 / self.scale)

        disparity[disparity <= 0.5] = 0.5

        baseline_m = baseline_cm / 100.0
        distance_m = (focal_length * baseline_m) / disparity

        distance_m = np.clip(distance_m, 0, Config.MAX_DIST_M)
        
        norm = 255 - np.uint8((distance_m / Config.MAX_DIST_M) * 255)
        color_map = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        
        return color_map, distance_m, norm