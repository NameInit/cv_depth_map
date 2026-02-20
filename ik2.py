import depthai as dai
from ultralytics import YOLO
import cv2
import numpy as np

def run_pipeline():
    pipeline = dai.Pipeline()

    model = YOLO("./yolov10n.pt")
    
    rgb_center_cam = pipeline.create(dai.node.ColorCamera)
    mono_left_cam = pipeline.create(dai.node.MonoCamera)
    mono_right_cam = pipeline.create(dai.node.MonoCamera)
    stereo_depth = pipeline.create(dai.node.StereoDepth)

    stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo_depth.setDepthAlign(dai.CameraBoardSocket.CENTER)

    mono_left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    rgb_center_cam.setBoardSocket(dai.CameraBoardSocket.CENTER)

    mono_left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    rgb_center_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    rgb_center_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
 
    mono_left_cam.out.link(stereo_depth.left)
    mono_right_cam.out.link(stereo_depth.right)

    stereo_depth_out = pipeline.create(dai.node.XLinkOut)
    stereo_depth_out.setStreamName("depth")
    stereo_depth.depth.link(stereo_depth_out.input)

    rgb_out = pipeline.create(dai.node.XLinkOut)
    rgb_out.setStreamName("rgb")
    rgb_center_cam.video.link(rgb_out.input)

    with dai.Device(pipeline) as device:
        cur_brightness = 800
        depth_queue: dai.DataOutputQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        rgb_queue: dai.DataOutputQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        device.setIrLaserDotProjectorBrightness(cur_brightness)
        print(f"Яркость: {cur_brightness}")

        while True:
            depth_data = depth_queue.get()
            depth_frame = depth_data.getCvFrame()
             
            rgb_data = rgb_queue.get()
            rgb_frame = rgb_data.getCvFrame()

            results = model(rgb_frame, verbose=False)
            
            rgb_frame_with_boxes = rgb_frame.copy()
            
            if len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        if conf > 0.7:
                            x1, y1, x2, y2 = box.astype(int)
                            
                            class_name = result.names[class_id] if hasattr(result, 'names') else f"Class_{class_id}"
                            
                            color = (0,0,255)
                            
                            cv2.rectangle(rgb_frame_with_boxes, (x1, y1), (x2, y2), color, 4)
                            
                            label = f"{class_name}: {conf:.2f}"
                            
                            (label_width, label_height), baseline = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                            )
                            
                            cv2.rectangle(
                                rgb_frame_with_boxes,
                                (x1, y1 - label_height - baseline - 5),
                                (x1 + label_width, y1),
                                color,
                                -1
                            )
                            
                            cv2.putText(
                                rgb_frame_with_boxes,
                                label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                2
                            )
                            
                            if depth_frame is not None:
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                
                                if (0 <= center_x < depth_frame.shape[1] and 
                                    0 <= center_y < depth_frame.shape[0]):
                                    depth_value = depth_frame[center_y, center_x]
                                    
                                    if depth_value > 0:
                                        cv2.putText(
                                            rgb_frame_with_boxes,
                                            f"Depth: {depth_value}mm",
                                            (x1, y2 + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            color,
                                            2
                                        )

            depth_frame_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            depth_frame_vis = cv2.applyColorMap(depth_frame_vis, cv2.COLORMAP_HOT)
            
            depth_frame_vis = cv2.resize(depth_frame_vis, (800, 400))
            rgb_frame_vis = cv2.resize(rgb_frame_with_boxes, (800, 400))

            cv2.putText(
                rgb_frame_vis,
                f"Objects detected: {len(boxes) if 'boxes' in locals() else 0}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Depth", depth_frame_vis)
            cv2.imshow("RGB with YOLO detections", rgb_frame_vis)

            cv2.moveWindow("Depth", 100, 50)
            cv2.moveWindow("RGB with YOLO detections", 100, 570)

            key = cv2.waitKey(1)
            if key == ord('w'):
                cur_brightness -= 200
                cur_brightness = 0 if cur_brightness < 0 else cur_brightness
                device.setIrLaserDotProjectorBrightness(cur_brightness)
                print(f"Яркость: {cur_brightness}")
            elif key == ord('e'):
                cur_brightness += 200
                cur_brightness = 1200 if cur_brightness > 1200 else cur_brightness
                device.setIrLaserDotProjectorBrightness(cur_brightness)
                print(f"Яркость: {cur_brightness}")
            elif key == ord('q'):
                break

if __name__ == "__main__":
    run_pipeline()