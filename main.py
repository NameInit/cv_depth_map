import cv2
from camera_manager import CameraManager
from stereo_sgbm import StereoSGBM
from detector import ObjectDetector
from fps_counter import FPSCounter
import utils
import numpy as np

def main():
    cam = CameraManager().start()
    real_focal, real_baseline = cam.get_calibration()

    depth_engine = StereoSGBM(scale=0.6, use_wls=True)
    
    detector = ObjectDetector()
    
    fps = FPSCounter()
    
    width_depth = 640
    height_depth = 400
    cx = width_depth / 2
    cy = height_depth / 2

    try:
        while True:
            left, right, rgb = cam.get_frames()

            if left is None: continue

            depth_vis, distance_map, _ = depth_engine.compute(left, right, focal_length=real_focal, baseline_cm=real_baseline*1000)
            
            detections = detector.detect(rgb)
            
            utils.draw_results(rgb, detections, distance_map)
            
            fps.update()
            cv2.putText(rgb, fps.get_fps_text(), (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            
            cv2.imshow("Main", cv2.resize(rgb, (640, 360)))
            cv2.imshow("Depth", cv2.resize(depth_vis, (640, 360)))
            
            if cv2.waitKey(1) == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()