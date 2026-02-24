import depthai as dai
import cv2
import numpy as np
from pathlib import Path
import urllib.request
import os
import torch
import time

'''
Для OAK-D PRO:
CAM_A - центральная цветная камера (RGB)
CAM_B - правая монохромная (правая)
CAM_C - левая монохромная (левая)
'''

mxid_oak: str = "184430105105351300"

YOLO_MODELS = {
    "yolov8n": {"name": "YOLOv8 Nano", "size": "6.2 MB", "url": "ultralytics/yolov8n"},
    "yolov8s": {"name": "YOLOv8 Small", "size": "22.5 MB", "url": "ultralytics/yolov8s"},
    "yolov8m": {"name": "YOLOv8 Medium", "size": "52.1 MB", "url": "ultralytics/yolov8m"},
    "yolov9t": {"name": "YOLOv9 Tiny", "size": "20.5 MB", "url": "WongKinYiu/yolov9t"},
    "yolov10n": {"name": "YOLOv10 Nano", "size": "5.7 MB", "url": "jameslahm/yolov10n"},
    "yolov10s": {"name": "YOLOv10 Small", "size": "16.5 MB", "url": "jameslahm/yolov10s"},
    "yolo11n": {"name": "YOLO11 Nano", "size": "5.4 MB", "url": "ultralytics/yolo11n"},
    "yolo11s": {"name": "YOLO11 Small", "size": "18.4 MB", "url": "ultralytics/yolo11s"},
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

class YOLODetector:
    def __init__(self, model_name="yolov8n", conf_threshold=0.25, iou_threshold=0.45):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = self._get_device()
        self.classes = COCO_CLASSES
        
    def _get_device(self):
        """Определение доступного устройства (CUDA / MPS / CPU)"""
        if torch.cuda.is_available():
            print("✓ Используется CUDA")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("✓ Используется MPS (Apple Silicon)")
            return "mps"
        else:
            print("✓ Используется CPU")
            return "cpu"
    
    def load_model(self):
        """Загрузка модели YOLO"""
        try:
            from ultralytics import YOLO
            
            print(f"Загрузка модели {YOLO_MODELS[self.model_name]['name']}...")
            
            self.model = YOLO(self.model_name)
            
            if self.device != "cpu":
                self.model.to(self.device)
            
            print(f"✓ Модель {YOLO_MODELS[self.model_name]['name']} загружена")
            return True
            
        except ImportError:
            print("✗ Установите ultralytics: pip install ultralytics")
            return False
        except Exception as e:
            print(f"✗ Ошибка загрузки модели: {e}")
            return False
    
    def detect(self, frame):
        """Детекция объектов на кадре"""
        if self.model is None:
            return []
        
        try:
            results = self.model(
                frame, 
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            detections = []
            if len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i in range(len(boxes)):
                        detections.append({
                            'bbox': boxes[i].astype(int),
                            'confidence': float(confidences[i]),
                            'class_id': int(class_ids[i]),
                            'class_name': self.classes[int(class_ids[i])] if int(class_ids[i]) < len(self.classes) else f"class_{int(class_ids[i])}"
                        })
            
            return detections
            
        except Exception as e:
            print(f"Ошибка детекции: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """Отрисовка результатов детекции"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            color = self._get_color(det['class_id'])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} {conf:.2f}"
            
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def _get_color(self, class_id):
        np.random.seed(class_id)
        return tuple(int(x) for x in np.random.randint(0, 255, 3))

def select_model():
    print("\nДоступные модели YOLO:")
    models_list = list(YOLO_MODELS.items())
    
    for i, (model_key, model_info) in enumerate(models_list, 1):
        print(f"{i}. {model_info['name']} ({model_info['size']})")
    
    while True:
        try:
            choice = input(f"\nВыберите модель (1-{len(models_list)}), Enter для YOLOv8n: ").strip()
            if choice == "":
                return "yolov8n"
            
            choice = int(choice)
            if 1 <= choice <= len(models_list):
                return models_list[choice-1][0]
            else:
                print(f"Введите число от 1 до {len(models_list)}")
        except ValueError:
            print("Введите корректное число")

def run_pipeline():
    devices = dai.Device.getAllAvailableDevices()
    if not len(devices):
        print("No OAK devices found.")
        return
    
    device_found = False
    for dev in devices:
        if dev.mxid == mxid_oak:
            device_found = True
            break
    
    if not device_found:
        print(f"Device {mxid_oak} not found. Available devices:")
        for dev in devices:
            print(f"  - {dev.mxid}")
        return
    
    # Выбор модели
    model_key = select_model()
    
    # Инициализация детектора
    detector = YOLODetector(
        model_name=model_key,
        conf_threshold=0.7,
        iou_threshold=0.45
    )
    
    # Загрузка модели
    if not detector.load_model():
        print("Не удалось загрузить модель. Запуск без детекции.")
    
    device_info = dai.DeviceInfo(mxid_oak)
    pipeline = dai.Pipeline()
    
    # Создание узлов камер
    left_cam = pipeline.create(dai.node.MonoCamera)
    right_cam = pipeline.create(dai.node.MonoCamera)
    middle_cam = pipeline.create(dai.node.ColorCamera)
    
    # Настройка камер
    left_cam.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right_cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    middle_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    
    left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    right_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    middle_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    
    middle_cam.setInterleaved(False)
    middle_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    
    # Выходные узлы
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
        device.startPipeline()
        
        right_queue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
        left_queue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        middle_queue = device.getOutputQueue(name="middle", maxSize=4, blocking=False)

        fps_counter = 0
        fps_start = time.time()
        fps = 0
        
        detection_times = []
        
        while True:
            left_frame_data = left_queue.get()
            right_frame_data = right_queue.get()
            middle_frame_data = middle_queue.get()
            
            left_frame = left_frame_data.getCvFrame()
            right_frame = right_frame_data.getCvFrame()
            middle_frame = middle_frame_data.getCvFrame()
            
            if detector.model is not None:
                detection_start = time.time()
                detections = detector.detect(middle_frame)
                detection_time = time.time() - detection_start
                detection_times.append(detection_time)
                
                middle_frame = detector.draw_detections(middle_frame, detections)
                
                if len(detections) > 0:
                    print(f"\rОбъектов: {len(detections)} | Время детекции: {detection_time*1000:.1f}ms", end="")
            
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start = time.time()
                
                if detection_times:
                    avg_time = np.mean(detection_times) * 1000
                    print(f"\nFPS: {fps} | Среднее время детекции: {avg_time:.1f}ms")
                    detection_times = []
            
            left_frame_resized = cv2.resize(left_frame, (640, 360))
            right_frame_resized = cv2.resize(right_frame, (640, 360))
            middle_frame_resized = cv2.resize(middle_frame, (640, 360))
            
            cv2.putText(middle_frame_resized, 
                       f"{YOLO_MODELS[model_key]['name']} | FPS: {fps}", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(middle_frame_resized, 
                       f"Device: {detector.device}", 
                       (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("OAK-D Camera LEFT", left_frame_resized)
            cv2.imshow("OAK-D Camera RIGHT", right_frame_resized)
            cv2.imshow("OAK-D Camera MIDDLE", middle_frame_resized)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                detector.conf_threshold = min(1.0, detector.conf_threshold + 0.05)
                print(f"\nConfidence threshold: {detector.conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                detector.conf_threshold = max(0.05, detector.conf_threshold - 0.05)
                print(f"\nConfidence threshold: {detector.conf_threshold:.2f}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()