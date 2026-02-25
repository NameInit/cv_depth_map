from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

class DetectionModel:
    def __init__(self, model_path: Path, model_name: str):
        self.model = YOLO(str(model_path / model_name))
        self.names = self.model.names
        
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[dict]:
        """Обнаружение объектов на изображении"""
        results = self.model(image, verbose=False)
        
        detections = []
        if len(results):
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    if conf > conf_threshold:
                        detections.append({
                            'box': box.astype(int),
                            'confidence': conf,
                            'class_id': class_id,
                            'class_name': result.names[class_id],
                            'center': self._get_center(box)
                        })
        return detections
    
    @staticmethod
    def _get_center(box):
        """Получение центра bounding box"""
        x1, y1, x2, y2 = box
        return (int((x1 + x2) // 2), int((y1 + y2) // 2))


class DepthModel:
    def __init__(self, encoder: str, features: int, out_channels: list, 
                 model_path: Path, model_name: str):
        self.device = self._get_device()
        
        self.model = DepthAnythingV2(
            encoder=encoder, 
            features=features, 
            out_channels=out_channels
        )
        self.model.load_state_dict(
            torch.load(model_path / model_name, map_location='cpu')
        )
        self.model = self.model.to(self.device).eval()
        
    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def infer(self, image: np.ndarray) -> np.ndarray:
        """Получение карты глубины"""
        depth = self.model.infer_image(image)
        return self._normalize_depth(depth)
    
    @staticmethod
    def _normalize_depth(depth: np.ndarray) -> np.ndarray:
        """Нормализация карты глубины"""
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min) * 255.0
        return depth.astype(np.uint8)