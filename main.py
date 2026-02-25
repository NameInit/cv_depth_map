import depthai as dai
import cv2
import numpy as np
from stereo_sgbm import calc_depth_map
from fps_counter import FPSCounter
from ultralytics import YOLO

model_detect_name = "yolov10n.pt"
model_detect_path = ".models/model_pt/"

def run_pipeline():
    
    model_detect = YOLO(model_detect_path + model_detect_name)
    pipeline = dai.Pipeline()
    
    left_cam = pipeline.create(dai.node.MonoCamera)
    right_cam = pipeline.create(dai.node.MonoCamera)
    middle_cam = pipeline.create(dai.node.ColorCamera)
    
    left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    middle_cam.setBoardSocket(dai.CameraBoardSocket.CENTER)
    
    left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    middle_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    middle_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.setLeftRightCheck(True)
    
    left_cam.out.link(stereo.left)
    right_cam.out.link(stereo.right)
    
    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_left.setStreamName("left")
    xout_right = pipeline.create(dai.node.XLinkOut)
    xout_right.setStreamName("right")
    xout_middle = pipeline.create(dai.node.XLinkOut)
    xout_middle.setStreamName("middle")
    
    stereo.rectifiedLeft.link(xout_left.input)
    stereo.rectifiedRight.link(xout_right.input)
    
    middle_cam.video.link(xout_middle.input)
    
    with dai.Device(pipeline) as device:
        device.setIrLaserDotProjectorBrightness(800)
        
        left_queue = device.getOutputQueue("left", 1, False)
        right_queue = device.getOutputQueue("right", 1, False)
        middle_queue = device.getOutputQueue("middle", 1, False)

        fps = FPSCounter()

        while True:
            left_frame = left_queue.get().getCvFrame()
            right_frame = right_queue.get().getCvFrame()
            middle_frame = middle_queue.get().getCvFrame()
            
            custom_disp, distance_m, _ = calc_depth_map(left_frame, right_frame)
            
            results = model_detect(middle_frame, verbose=False)
            
            h_rgb, w_rgb = middle_frame.shape[:2]  # (1080, 1920)
            h_depth, w_depth = distance_m.shape[:2]  # (400, 640)
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model_detect.names[cls_id]

                    cx_rgb = (x1 + x2) // 2
                    cy_rgb = (y1 + y2) // 2

                    cx_depth = int(cx_rgb * (w_depth / w_rgb))
                    cy_depth = int(cy_rgb * (h_depth / h_rgb))

                    dist_m = 0.0
                    
                    if 0 <= cx_depth < w_depth and 0 <= cy_depth < h_depth:
                        x_min = max(0, cx_depth - 5)
                        x_max = min(w_depth, cx_depth + 5)
                        y_min = max(0, cy_depth - 5)
                        y_max = min(h_depth, cy_depth + 5)
                        
                        roi = distance_m[y_min:y_max, x_min:x_max]
                        
                        valid_distances = roi[(roi > 0.5) & (roi < 11.9)]
                        
                        if len(valid_distances) > 0:
                            dist_m = np.median(valid_distances)

                    color = (0, 0, 255)
                    cv2.rectangle(middle_frame, (x1, y1), (x2, y2), color, 4)
                    
                    if dist_m > 0:
                        label = f"{cls_name} {dist_m:.2f}m"
                    else:
                        label = f"{cls_name} N/A"
                        
                    cv2.putText(middle_frame, label, (x1, max(y1 - 10, 30)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

            left_small = cv2.resize(left_frame, (320, 200))
            right_small = cv2.resize(right_frame, (320, 200))
            middle_small = cv2.resize(middle_frame, (320, 200))
            custom_disp_small = cv2.resize(custom_disp, (320, 200))
                
            fps.update()
            
            cv2.putText(middle_small, fps.get_fps_text(), (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("OpenCV SGDB", custom_disp_small)
            cv2.imshow("Left Rectified", left_small)
            cv2.imshow("Right Rectified", right_small)
            cv2.imshow("RGB", middle_small)
            
            if cv2.waitKey(1) == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()