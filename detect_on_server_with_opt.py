import depthai as dai
import cv2
import numpy as np
from pathlib import Path
import torch
import time
from threading import Thread
from collections import deque
import queue

mxid_oak: str = "184430105105351300"

YOLO_MODELS = {
    "yolov8n": {"name": "YOLOv8 Nano", "size": "6.2 MB", "fps": "высокий"},
    "yolov8s": {"name": "YOLOv8 Small", "size": "22.5 MB", "fps": "средний"},
    "yolov10n": {"name": "YOLOv10 Nano", "size": "5.7 MB", "fps": "очень высокий"},
    "yolo11n": {"name": "YOLO11 Nano", "size": "5.4 MB", "fps": "очень высокий"},
}

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class OptimizedYOLODetector:
    def __init__(self, model_name="yolov10n", conf_threshold=0.25, iou_threshold=0.45):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = self._get_device()
        self.classes = COCO_CLASSES
        
        self.frame_size = (640, 640)
        self.half_precision = self.device == "cuda"
        self.warmup_done = False
        
    def _get_device(self):
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ Используется CUDA GPU ({gpu_mem:.1f} GB)")
            return "cuda"
        else:
            print("✓ Используется CPU")
            return "cpu"
    
    def load_model(self):
        try:
            from ultralytics import YOLO
            
            print(f"Загрузка модели {YOLO_MODELS[self.model_name]['name']}...")
            
            self.model = YOLO(self.model_name)
            
            if self.device != "cpu":
                self.model.to(self.device)
                
                print("Прогрев модели...")
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                for _ in range(3):
                    _ = self.model(dummy, verbose=False)
                
                if self.half_precision:
                    self.model.model.half()
                    print("  ✓ FP16 оптимизация включена")
            
            print(f"✓ Модель {YOLO_MODELS[self.model_name]['name']} загружена")
            return True
            
        except Exception as e:
            print(f"✗ Ошибка загрузки модели: {e}")
            return False
    
    @torch.no_grad()
    def detect(self, frame):
        """Оптимизированная детекция"""
        if self.model is None:
            return []
        
        try:
            if frame.shape[:2] != self.frame_size:
                frame_resized = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
            else:
                frame_resized = frame
            
            results = self.model(
                frame_resized, 
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                half=self.half_precision,
                augment=False,
                agnostic_nms=False
            )
            
            scale_x = frame.shape[1] / self.frame_size[0]
            scale_y = frame.shape[0] / self.frame_size[1]
            
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    detections.append({
                        'bbox': [
                            int(x1 * scale_x),
                            int(y1 * scale_y),
                            int(x2 * scale_x),
                            int(y2 * scale_y)
                        ],
                        'confidence': float(confidences[i]),
                        'class_id': int(class_ids[i]),
                        'class_name': self.classes[int(class_ids[i])] if int(class_ids[i]) < len(self.classes) else f"class_{int(class_ids[i])}"
                    })
            
            return detections
            
        except Exception as e:
            print(f"Ошибка детекции: {e}")
            return []

class PipelineOptimizer:
    """Класс для оптимизации пайплайна OAK"""
    
    @staticmethod
    def create_optimized_pipeline():
        pipeline = dai.Pipeline()
        
        middle_cam = pipeline.create(dai.node.ColorCamera)
        middle_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        middle_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        middle_cam.setInterleaved(False)
        middle_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        
        middle_cam.setFps(60)
        
        middle_cam.setPreviewSize(640, 360)
        middle_cam.setPreviewKeepAspectRatio(False)
        
        left_cam = pipeline.create(dai.node.MonoCamera)
        right_cam = pipeline.create(dai.node.MonoCamera)
        left_cam.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        right_cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        
        left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        right_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        left_cam.setFps(20)
        right_cam.setFps(20)
        
        xout_left = pipeline.create(dai.node.XLinkOut)
        xout_left.setStreamName("left")
        xout_right = pipeline.create(dai.node.XLinkOut)
        xout_right.setStreamName("right")
        xout_middle = pipeline.create(dai.node.XLinkOut)
        xout_middle.setStreamName("middle")
        
        left_cam.out.link(xout_left.input)
        right_cam.out.link(xout_right.input)
        middle_cam.preview.link(xout_middle.input)
        
        return pipeline

def run_optimized_pipeline():
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("No OAK devices found.")
        return
    
    print("\nДоступные модели (рекомендации по FPS):")
    models_list = list(YOLO_MODELS.items())
    for i, (model_key, model_info) in enumerate(models_list, 1):
        print(f"{i}. {model_info['name']} ({model_info['size']}, {model_info['fps']} FPS)")
    
    choice = input("\nВыберите модель (Enter для YOLOv10n): ").strip()
    if not choice:
        model_key = "yolov10n"
    else:
        try:
            model_key = models_list[int(choice)-1][0]
        except:
            model_key = "yolov10n"
    
    detector = OptimizedYOLODetector(
        model_name=model_key,
        conf_threshold=0.7
    )
    
    if not detector.load_model():
        return
    
    pipeline = PipelineOptimizer.create_optimized_pipeline()
    
    with dai.Device(pipeline) as device:
        device.startPipeline()
        
        queues = {
            'left': device.getOutputQueue(name="left", maxSize=1, blocking=False),
            'right': device.getOutputQueue(name="right", maxSize=1, blocking=False),
            'middle': device.getOutputQueue(name="middle", maxSize=1, blocking=False)
        }
        
        print("Стабилизация камеры...")
        for _ in range(10):
            for q in queues.values():
                q.tryGet()
            time.sleep(0.01)
        
        fps_history = deque(maxlen=30)
        detection_times = deque(maxlen=30)
        last_time = time.time()
        frame_count = 0
        
        print("\n" + "="*50)
        print("ЗАПУЩЕНО (оптимизированный режим)")
        print("Управление: q - выход, +/- - порог уверенности")
        print("="*50)
        
        while True:
            left_frame = queues['left'].tryGet()
            right_frame = queues['right'].tryGet()
            middle_frame = queues['middle'].tryGet()
            
            if middle_frame is None:
                time.sleep(0.001)
                continue
            
            middle_frame = middle_frame.getCvFrame()
            
            detect_start = time.time()
            detections = detector.detect(middle_frame)
            detect_time = time.time() - detect_start
            detection_times.append(detect_time)
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                class_name = det['class_name']
                
                cv2.rectangle(middle_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if conf > 0.7:
                    cv2.putText(middle_frame, f"{class_name}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= 0.5:
                fps = frame_count / (current_time - last_time)
                fps_history.append(fps)
                avg_fps = sum(fps_history) / len(fps_history)
                avg_detect = sum(detection_times) / len(detection_times) * 1000
                
                print(f"\rFPS: {avg_fps:.1f} | Детекция: {avg_detect:.1f}ms | Объектов: {len(detections)}", end="")
                
                frame_count = 0
                last_time = current_time
            
            cv2.putText(middle_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(middle_frame, f"Model: {YOLO_MODELS[model_key]['name']}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("OAK-D Detection (Optimized)", middle_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                detector.conf_threshold = min(0.95, detector.conf_threshold + 0.05)
                print(f"\nConfidence: {detector.conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                detector.conf_threshold = max(0.05, detector.conf_threshold - 0.05)
                print(f"\nConfidence: {detector.conf_threshold:.2f}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_optimized_pipeline()