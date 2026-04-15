import cv2
from src.camera_manager import CameraManager
from src.stereo_sgbm import StereoSGBM
from src.detector import ObjectDetector
from src.fps_counter import FPSCounter
from src.config import Config
from src import utils

def main():
    cam = CameraManager().start()

    stereo_sgbm = StereoSGBM(Config.SGDM_SCALE, Config.SGDM_WLS)

    detector = ObjectDetector()

    fps = FPSCounter()

    try:
        while True:
            left, right, rgb = cam.get_frames()

            depth_vis, distance_map, _ = stereo_sgbm.compute(left, right)

            detections = detector.detect(rgb, Config.MODEL_CONF)
            
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