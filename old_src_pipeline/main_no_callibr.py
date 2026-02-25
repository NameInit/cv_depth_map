import depthai as dai
import cv2
import numpy as np
import stereo_sgbm
from ultralytics import YOLO

detect_path = "model_pt/"
detect_name = "yolov10n.pt"

def run_pipeline():
    model_detect = YOLO(detect_path + detect_name)
    pipeline = dai.Pipeline()
    
    left_cam = pipeline.create(dai.node.MonoCamera)
    right_cam = pipeline.create(dai.node.MonoCamera)
    middle_cam = pipeline.create(dai.node.ColorCamera)
    
    left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    middle_cam.setBoardSocket(dai.CameraBoardSocket.CENTER)
    
    left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    right_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    middle_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    
    middle_cam.setInterleaved(False)
    middle_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    
    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_left.setStreamName("left")
    xout_right = pipeline.create(dai.node.XLinkOut)
    xout_right.setStreamName("right")
    xout_middle = pipeline.create(dai.node.XLinkOut)
    xout_middle.setStreamName("middle")
    
    left_cam.out.link(xout_left.input)
    right_cam.out.link(xout_right.input)
    middle_cam.video.link(xout_middle.input)
    
    with dai.Device(pipeline) as device:
        device.setIrLaserDotProjectorBrightness(800)

        right_queue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
        left_queue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        middle_queue = device.getOutputQueue(name="middle", maxSize=4, blocking=False)

        while True:
            left_frame_data = left_queue.get()
            right_frame_data = right_queue.get()
            middle_frame_data = middle_queue.get()
            
            left_frame = left_frame_data.getCvFrame()
            right_frame = right_frame_data.getCvFrame()
            middle_frame = middle_frame_data.getCvFrame()
            
            custom_disp, distance_m, _ = stereo_sgbm.calc_depth_map(left_frame, right_frame)

            results = model_detect(middle_frame, verbose=False, conf=0.7)
            h_rgb, w_rgb = middle_frame.shape[:2]
            h_depth, w_depth = distance_m.shape[:2]

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model_detect.names[cls_id]
                    
                    cx_rgb = (x1 + x2) // 2
                    cy_rgb = (y1 + y2) // 2
                    
                    cx_depth = int(cx_rgb * (w_depth / w_rgb))
                    cy_depth = int(cy_rgb * (h_depth / h_rgb))
                    
                    obj_dist_m = 0.0
                    
                    if 0 <= cx_depth < w_depth and 0 <= cy_depth < h_depth:
                        roi_size = 10
                        x_min = max(0, cx_depth - roi_size)
                        x_max = min(w_depth, cx_depth + roi_size)
                        y_min = max(0, cy_depth - roi_size)
                        y_max = min(h_depth, cy_depth + roi_size)
                        
                        roi = distance_m[y_min:y_max, x_min:x_max]
                        
                        valid_pixels = roi[(roi > 0.2) & (roi < 12.0)]
                        
                        if len(valid_pixels) > 0:
                            obj_dist_m = np.median(valid_pixels)
                    
                    color = (0, 0, 255)
                    cv2.rectangle(middle_frame, (x1, y1), (x2, y2), color, 2)
                    
                    if obj_dist_m > 0:
                        label = f"{class_name}: {obj_dist_m:.2f}m"
                    else:
                        label = f"{class_name}: N/A"
                        
                    (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(middle_frame, (x1, y1 - 25), (x1 + w_text, y1), color, -1)
                    cv2.putText(middle_frame, label, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            left_frame_resized = cv2.resize(left_frame, (640, 360))
            right_frame_resized = cv2.resize(right_frame, (640, 360))
            middle_frame_resized = cv2.resize(middle_frame, (640, 360))
            disp_resized = cv2.resize(custom_disp,  (640, 360))
            
            cv2.imshow("OAK-D Camera LEFT", left_frame_resized)
            cv2.imshow("OAK-D Camera RIGHT", right_frame_resized)
            cv2.imshow("OAK-D Camera MIDDLE", middle_frame_resized)
            cv2.imshow("SGBM", disp_resized)
            
            if cv2.waitKey(1) == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()