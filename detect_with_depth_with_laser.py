import depthai as dai
import cv2
import numpy as np
from pathlib import Path
import torch
import time
from collections import deque

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
        self.half_precision = False
        
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
                print("  ✓ CUDA оптимизация включена")
            
            # Прогрев модели
            print("Прогрев модели...")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(2):
                _ = self.model(dummy, verbose=False)
            
            print(f"✓ Модель загружена")
            return True
        except Exception as e:
            print(f"✗ Ошибка загрузки модели: {e}")
            return False
    
    @torch.no_grad()
    def detect(self, frame):
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
                half=False,
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
                        'center': [
                            int((x1 + x2) / 2 * scale_x),
                            int((y1 + y2) / 2 * scale_y)
                        ],
                        'confidence': float(confidences[i]),
                        'class_id': int(class_ids[i]),
                        'class_name': self.classes[int(class_ids[i])] if int(class_ids[i]) < len(self.classes) else f"class_{int(class_ids[i])}"
                    })
            
            return detections
        except Exception as e:
            print(f"Ошибка детекции: {e}")
            return []

def create_depth_pipeline_with_laser(max_distance_meters=30.0):
    """Создание пайплайна с картой глубины"""
    pipeline = dai.Pipeline()
    
    # Цветная камера для детекции
    color_cam = pipeline.create(dai.node.ColorCamera)
    color_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color_cam.setInterleaved(False)
    color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    color_cam.setPreviewSize(640, 360)
    color_cam.setFps(30)
    
    # Моно камеры для стерео
    left_cam = pipeline.create(dai.node.MonoCamera)
    right_cam = pipeline.create(dai.node.MonoCamera)
    left_cam.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right_cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left_cam.setFps(30)
    right_cam.setFps(30)
    
    # Стерео обработка для глубины
    stereo = pipeline.create(dai.node.StereoDepth)
    
    # === НАСТРОЙКИ ДЛЯ ДАЛЬНОСТИ 30 МЕТРОВ ===
    
    # 1. Extended disparity обязателен
    stereo.setExtendedDisparity(True)
    
    # 2. Включаем все алгоритмы улучшения
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(640, 360)
    
    # 3. Confidence threshold (меньше = больше дальность)
    stereo.initialConfig.setConfidenceThreshold(80)
    
    # 4. Настройка left-right check threshold
    stereo.initialConfig.setLeftRightCheckThreshold(2)
    
    # 5. Медианный фильтр
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    
    # 6. Дополнительные настройки для дальних объектов
    # Увеличиваем максимальную диспаратность
    stereo.setExtendedDisparity(True)
    
    print(f"✓ Настройки стерео применены:")
    print(f"  - Confidence threshold: 80")
    print(f"  - Left-right check: 2")
    print(f"  - Extended disparity: включен")
    print(f"  - Max distance: до {max_distance_meters}м")
    
    left_cam.out.link(stereo.left)
    right_cam.out.link(stereo.right)
    
    # Выходы
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    color_cam.preview.link(xout_rgb.input)
    
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)
    
    xout_disparity = pipeline.create(dai.node.XLinkOut)
    xout_disparity.setStreamName("disparity")
    stereo.disparity.link(xout_disparity.input)
    
    return pipeline

def calculate_distance(depth_frame, x, y, window_size=15, max_distance_m=30.0):
    """Расчет расстояния до точки"""
    h, w = depth_frame.shape[:2]
    
    # Проверка границ
    x = max(window_size, min(x, w - window_size - 1))
    y = max(window_size, min(y, h - window_size - 1))
    
    # Окно для усреднения
    roi = depth_frame[y-window_size:y+window_size, x-window_size:x+window_size]
    
    # Фильтрация нулевых значений
    valid_pixels = roi[roi > 0]
    
    if len(valid_pixels) > 5:
        # Используем медиану для устойчивости к шуму
        distance_mm = np.median(valid_pixels)
        distance_m = distance_mm / 1000.0
        
        # Проверка на максимальную дальность
        if distance_m <= max_distance_m:
            return distance_m
    
    return 0.0

def create_colored_depth_map(disparity_frame, max_distance_m=30.0):
    """Создание цветной карты глубины"""
    
    # Параметры камеры OAK-D PRO
    baseline_mm = 75  # расстояние между камерами в мм
    focal_length_px = 880  # фокусное расстояние в пикселях
    
    # Защита от деления на ноль
    disparity_frame_safe = np.maximum(disparity_frame, 1)
    
    # Расчет глубины в метрах: distance = (focal_length * baseline) / disparity
    depth_m = (baseline_mm * focal_length_px) / (disparity_frame_safe * 1000)
    
    # Обрезаем по максимальной дальности
    depth_m = np.clip(depth_m, 0, max_distance_m)
    
    # Нормализация для визуализации
    depth_normalized = (depth_m / max_distance_m * 255).astype(np.uint8)
    
    # Инвертируем для цветовой карты (близко - красный, далеко - синий)
    depth_normalized = 255 - depth_normalized
    
    # Применение цветовой карты
    colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    return colored_depth

def run_pipeline_with_laser():
    # Проверка устройства
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("No OAK devices found.")
        return
    
    # Выбор модели
    print("\nДоступные модели:")
    models_list = list(YOLO_MODELS.items())
    for i, (model_key, model_info) in enumerate(models_list, 1):
        print(f"{i}. {model_info['name']} ({model_info['fps']} FPS)")
    
    try:
        choice = input("\nВыберите модель (Enter для YOLOv10n): ").strip()
        if not choice:
            model_key = "yolov10n"
        else:
            model_key = models_list[int(choice)-1][0]
    except:
        model_key = "yolov10n"
        print(f"Используется {model_key}")
    
    # Инициализация детектора
    detector = OptimizedYOLODetector(
        model_name=model_key,
        conf_threshold=0.5
    )
    
    if not detector.load_model():
        return
    
    # Максимальная дальность
    MAX_DISTANCE = 30.0
    
    # Создание пайплайна
    print("\n" + "="*60)
    print("СОЗДАНИЕ ПАЙПЛАЙНА OAK-D PRO")
    print("="*60)
    pipeline = create_depth_pipeline_with_laser(max_distance_meters=MAX_DISTANCE)
    
    print("\n" + "="*60)
    print("ЗАПУСК УСТРОЙСТВА")
    print("="*60)
    
    laser_brightness = 800  # Начальная яркость
    laser_enabled = True
    
    with dai.Device(pipeline) as device:
        # Включение лазера
        if laser_enabled:
            try:
                device.setIrLaserDotProjectorBrightness(laser_brightness)
                print(f"✓ Лазер активирован (яркость: {laser_brightness})")
            except Exception as e:
                print(f"✗ Не удалось включить лазер: {e}")
                laser_enabled = False
        
        # Очереди
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        depth_queue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        disparity_queue = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
        
        # Стабилизация
        print("Стабилизация камеры...")
        for i in range(20):
            rgb_queue.tryGet()
            depth_queue.tryGet()
            disparity_queue.tryGet()
            if i % 5 == 0:
                print(f"  {i}/20", end="\r")
            time.sleep(0.05)
        print("  Готово!          ")
        
        # Статистика
        fps_history = deque(maxlen=30)
        last_time = time.time()
        frame_count = 0
        current_fps = 0
        
        print("\n" + "="*70)
        print(f"ЗАПУЩЕНО: Детекция + Карта глубины (макс: {MAX_DISTANCE}м)")
        print("Управление: q - выход, +/- - порог уверенности, s - сохранить")
        print("           l - вкл/выкл лазер, [ ] - яркость лазера -/+")
        print("="*70)
        
        while True:
            try:
                # Получение кадров
                rgb_data = rgb_queue.get()
                depth_data = depth_queue.get()
                disparity_data = disparity_queue.get()
                
                rgb_frame = rgb_data.getCvFrame()
                depth_frame = depth_data.getFrame()
                disparity_frame = disparity_data.getFrame()
                
                # Детекция объектов
                detect_start = time.time()
                detections = detector.detect(rgb_frame)
                detect_time = time.time() - detect_start
                
                # Создание цветной карты глубины
                depth_colored = create_colored_depth_map(disparity_frame, MAX_DISTANCE)
                
                # Отрисовка детекций и расстояний
                objects_with_distance = []
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cx, cy = det['center']
                    conf = det['confidence']
                    class_name = det['class_name']
                    
                    # Получение расстояния
                    distance = calculate_distance(depth_frame, cx, cy, max_distance_m=MAX_DISTANCE)
                    
                    if distance > 0:
                        objects_with_distance.append((distance, class_name))
                    
                    # Цвет в зависимости от расстояния
                    if distance < 5.0:
                        color = (0, 0, 255)  # Красный
                    elif distance < 15.0:
                        color = (0, 255, 255)  # Желтый
                    elif distance < 25.0:
                        color = (0, 255, 0)  # Зеленый
                    else:
                        color = (255, 0, 0)  # Синий
                    
                    # Отрисовка на RGB
                    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Информация об объекте
                    cv2.putText(rgb_frame, class_name, (x1, y1-35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if distance > 0:
                        distance_text = f"{distance:.1f}m"
                        cv2.putText(rgb_frame, distance_text, (x1, y1-15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Отрисовка на карте глубины
                    if 0 <= cx < depth_colored.shape[1] and 0 <= cy < depth_colored.shape[0]:
                        cv2.circle(depth_colored, (cx, cy), 5, color, -1)
                        if distance > 0:
                            cv2.putText(depth_colored, f"{distance:.1f}m", (cx+10, cy), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Расчет FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    current_fps = frame_count
                    fps_history.append(current_fps)
                    avg_fps = sum(fps_history) / len(fps_history)
                    
                    laser_status = "ВКЛ" if laser_enabled else "ВЫКЛ"
                    dist_info = f" | Объектов с расстоянием: {len(objects_with_distance)}"
                    if objects_with_distance:
                        min_dist = min(objects_with_distance)[0]
                        max_dist = max(objects_with_distance)[0]
                        dist_info += f" (мин: {min_dist:.1f}м, макс: {max_dist:.1f}м)"
                    
                    print(f"\rFPS: {avg_fps:.1f} | Объектов: {len(detections)}{dist_info} | "
                          f"Лазер: {laser_status} ({laser_brightness})", end="")
                    
                    frame_count = 0
                    last_time = current_time
                
                # Информация на кадрах
                cv2.putText(rgb_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                laser_status_text = f"LASER: {'ON' if laser_enabled else 'OFF'} ({laser_brightness})"
                cv2.putText(rgb_frame, laser_status_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.putText(depth_colored, f"Depth Map (0-{MAX_DISTANCE}m)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Отображение
                cv2.imshow("Object Detection", rgb_frame)
                # cv2.imshow("Depth Map", depth_colored)
                
                # Управление
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+') or key == ord('='):
                    detector.conf_threshold = min(0.95, detector.conf_threshold + 0.05)
                    print(f"\nПорог уверенности: {detector.conf_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    detector.conf_threshold = max(0.05, detector.conf_threshold - 0.05)
                    print(f"\nПорог уверенности: {detector.conf_threshold:.2f}")
                elif key == ord('l'):
                    laser_enabled = not laser_enabled
                    try:
                        if laser_enabled:
                            device.setIrLaserDotProjectorBrightness(laser_brightness)
                            print(f"\n✓ Лазер ВКЛ (яркость: {laser_brightness})")
                        else:
                            device.setIrLaserDotProjectorBrightness(0)
                            print(f"\n✗ Лазер ВЫКЛ")
                    except Exception as e:
                        print(f"\n✗ Ошибка управления лазером: {e}")
                elif key == ord(']'):
                    laser_brightness = min(1200, laser_brightness + 100)
                    if laser_enabled:
                        try:
                            device.setIrLaserDotProjectorBrightness(laser_brightness)
                        except:
                            pass
                    print(f"\nЯркость лазера: {laser_brightness}")
                elif key == ord('['):
                    laser_brightness = max(0, laser_brightness - 100)
                    if laser_enabled:
                        try:
                            device.setIrLaserDotProjectorBrightness(laser_brightness)
                        except:
                            pass
                    print(f"\nЯркость лазера: {laser_brightness}")
                elif key == ord('s'):
                    timestamp = int(time.time())
                    cv2.imwrite(f"detection_{timestamp}.jpg", rgb_frame)
                    cv2.imwrite(f"depth_{timestamp}.jpg", depth_colored)
                    print(f"\nСохранено detection_{timestamp}.jpg и depth_{timestamp}.jpg")
                    
            except Exception as e:
                print(f"Ошибка: {e}")
                continue
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline_with_laser()