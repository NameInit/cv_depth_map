#!/usr/bin/env python3
import depthai as dai
import cv2
import numpy as np
import time
from pathlib import Path

# Классы COCO для YOLOv8
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

class YoloV8DepthAI:
    def __init__(self, blob_path="yolov8n.blob", conf_threshold=0.5, iou_threshold=0.45):
        self.blob_path = blob_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = None
        self.rgb_queue = None
        self.nn_queue = None
        self.classes = COCO_CLASSES
        
        # Проверка существования blob файла
        if not Path(blob_path).exists():
            raise FileNotFoundError(f"Blob файл не найден: {blob_path}")
        
    def create_pipeline(self):
        """Создание пайплайна DepthAI"""
        pipeline = dai.Pipeline()
        
        # --- RGB камера ---
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)
        
        # Устанавливаем размер для нейросети (640x640)
        cam_rgb.setPreviewSize(640, 640)
        cam_rgb.setPreviewKeepAspectRatio(False)
        
        # --- Нейросеть ---
        detection_nn = pipeline.create(dai.node.NeuralNetwork)
        detection_nn.setBlobPath(self.blob_path)
        detection_nn.setNumPoolFrames(4)
        detection_nn.input.setBlocking(False)
        detection_nn.setNumInferenceThreads(2)
        
        # --- Выходные потоки ---
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)
        
        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("nn")
        detection_nn.out.link(xout_nn.input)
        
        # Линкуем preview камеры на вход нейросети
        cam_rgb.preview.link(detection_nn.input)
        
        return pipeline
    
    def decode_yolo_output(self, nn_data, frame_shape):
        """
        Декодирование выхода YOLOv8 из NNData для DepthAI 2.28.0
        """
        if nn_data is None:
            return []
        
        # Для DepthAI 2.28.0 используем getLayerFp16 или getLayerInt32
        try:
            # Пробуем разные методы получения данных
            if hasattr(nn_data, 'getFirstLayerFp16'):
                # Получаем первый слой как fp16
                output = np.array(nn_data.getFirstLayerFp16()).reshape(-1)
            elif hasattr(nn_data, 'getLayerFp16'):
                # Получаем конкретный слой
                layer_names = nn_data.getAllLayerNames()
                if layer_names:
                    output = np.array(nn_data.getLayerFp16(layer_names[0]))
                else:
                    return []
            else:
                # Альтернативный метод: конвертируем в numpy через буфер
                output = np.array(nn_data.getData()).view(np.float16)
        except:
            # Если ничего не работает, пробуем прямой доступ к данным
            try:
                output = np.array(nn_data.getData())
            except:
                print("Не удалось получить данные из NNData")
                return []
        
        # Определяем размерности
        if len(output.shape) == 1:
            # Пытаемся определить форму [1, 84, 8400]
            total_elements = output.shape[0]
            possible_shapes = [
                (1, 84, 8400),  # YOLOv8 standard
                (1, 84, 6300),  # YOLOv8 alternative
                (1, 56, 8400),  # YOLOv5
            ]
            
            for shape in possible_shapes:
                if total_elements == np.prod(shape):
                    output = output.reshape(shape)
                    break
            else:
                # Если не нашли подходящую форму, пробуем квадратную матрицу
                side = int(np.sqrt(total_elements / 84))
                if side * side * 84 == total_elements:
                    output = output.reshape(1, 84, side * side)
                else:
                    print(f"Неизвестная форма выходных данных: {total_elements} элементов")
                    return []
        
        # Транспонируем для удобства
        if len(output.shape) == 3:
            predictions = output[0].T  # [8400, 84]
        else:
            predictions = output
        
        detections = []
        frame_h, frame_w = frame_shape[:2]
        
        # Масштабирование координат
        scale_x = frame_w / 640
        scale_y = frame_h / 640
        
        for pred in predictions:
            if len(pred) < 5:
                continue
                
            # YOLOv8 формат: [x_center, y_center, width, height, class_scores...]
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]
            
            # Находим класс с максимальным score
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            # Фильтр по confidence
            if confidence < self.conf_threshold:
                continue
            
            # Конвертация в координаты [x1, y1, x2, y2]
            x1 = int((x_center - width/2) * scale_x)
            y1 = int((y_center - height/2) * scale_y)
            x2 = int((x_center + width/2) * scale_x)
            y2 = int((y_center + height/2) * scale_y)
            
            # Клиппинг координат
            x1 = max(0, min(x1, frame_w))
            y1 = max(0, min(y1, frame_h))
            x2 = max(0, min(x2, frame_w))
            y2 = max(0, min(y2, frame_h))
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'center': [int((x1 + x2)/2), int((y1 + y2)/2)],
                'confidence': float(confidence),
                'class_id': int(class_id),
                'class_name': self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
            })
        
        # Применяем NMS
        detections = self.nms(detections)
        
        return detections
    
    def nms(self, detections):
        """Non-Maximum Suppression"""
        if len(detections) == 0:
            return []
        
        # Сортируем по confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Удаляем пересекающиеся
            detections = [d for d in detections 
                         if self.iou(best['bbox'], d['bbox']) < self.iou_threshold]
        
        return keep
    
    def iou(self, box1, box2):
        """Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def draw_detections(self, frame, detections):
        """Отрисовка детекций на кадре"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Цвет в зависимости от класса
            color = (0, 255, 0)  # Зеленый по умолчанию
            
            # Разные цвета для разных категорий
            if class_name in ["person"]:
                color = (0, 255, 0)  # Зеленый
            elif class_name in ["car", "truck", "bus"]:
                color = (255, 0, 0)  # Синий
            elif class_name in ["dog", "cat", "bird"]:
                color = (0, 255, 255)  # Желтый
            
            # Рисуем bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Текст с классом и уверенностью
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Фон для текста
            cv2.rectangle(frame, 
                         (x1, y1 - label_h - 10), 
                         (x1 + label_w, y1), 
                         color, -1)
            
            # Текст
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def run(self):
        """Запуск пайплайна"""
        
        # Создание пайплайна
        pipeline = self.create_pipeline()
        
        # Подключение к устройству
        with dai.Device(pipeline) as device:
            print(f"✓ Устройство подключено: {device.getDeviceName()}")
            print(f"  Blob: {self.blob_path}")
            
            # Очереди
            self.rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.nn_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
            
            # Стабилизация
            print("Стабилизация камеры...")
            for _ in range(10):
                self.rgb_queue.tryGet()
                self.nn_queue.tryGet()
                time.sleep(0.1)
            
            print("✓ Запуск детекции")
            print("Управление: q - выход, +/- - порог уверенности")
            
            fps_counter = 0
            fps_last_time = time.time()
            current_fps = 0
            
            # Создание окон
            cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOv8 Detection", 960, 540)
            
            # Для отладки
            debug_output = True
            
            while True:
                # Получение кадров
                in_rgb = self.rgb_queue.tryGet()
                in_nn = self.nn_queue.tryGet()
                
                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()
                    
                    # Детекции
                    detections = []
                    if in_nn is not None:
                        # Декодируем выход нейросети
                        detections = self.decode_yolo_output(in_nn, frame.shape)
                        
                        # Отладка
                        if debug_output and len(detections) > 0:
                            print(f"Найдено объектов: {len(detections)}")
                            for i, det in enumerate(detections[:3]):  # Первые 3
                                print(f"  {i+1}: {det['class_name']} - {det['confidence']:.2f}")
                            debug_output = False
                    
                    # Отрисовка
                    frame = self.draw_detections(frame, detections)
                    
                    # FPS
                    fps_counter += 1
                    if time.time() - fps_last_time >= 1.0:
                        current_fps = fps_counter
                        fps_counter = 0
                        fps_last_time = time.time()
                    
                    # Информация на кадре
                    cv2.putText(frame, f"FPS: {current_fps}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Objects: {len(detections)}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Отображение
                    cv2.imshow("YOLOv8 Detection", frame)
                
                # Управление
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+') or key == ord('='):
                    self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                    print(f"Порог уверенности: {self.conf_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                    print(f"Порог уверенности: {self.conf_threshold:.2f}")
                elif key == ord('d'):
                    debug_output = not debug_output
                    print(f"Отладка: {'включена' if debug_output else 'выключена'}")
            
            cv2.destroyAllWindows()

def main():
    # Параметры
    blob_path = "yolov8n.blob"
    conf_threshold = 0.5
    
    # Проверка наличия blob
    if not Path(blob_path).exists():
        print(f"✗ Blob файл не найден: {blob_path}")
        return
    
    # Создание и запуск детектора
    detector = YoloV8DepthAI(
        blob_path=blob_path,
        conf_threshold=conf_threshold
    )
    
    try:
        detector.run()
    except KeyboardInterrupt:
        print("\n✓ Программа остановлена")
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()