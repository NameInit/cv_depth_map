import cv2
import numpy as np
import matplotlib
from typing import List, Dict, Optional, Tuple

class FrameVisualizer:
    def __init__(self, config):
        self.config = config
        self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        self._setup_windows()
        
    def _setup_windows(self):
        """Настройка окон отображения"""
        for name in self.config.window_names.values():
            cv2.namedWindow(name)
        
        for window_key, position in self.config.window_positions.items():
            window_name = self.config.window_names[window_key]
            cv2.moveWindow(window_name, position[0], position[1])
    
    def visualize_depth_stereo(self, depth_frame: np.ndarray) -> np.ndarray:
        """Визуализация стерео-глубины"""
        depth_vis = cv2.normalize(depth_frame, None, 0, 255, 
                                  cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        return cv2.applyColorMap(depth_vis, cv2.COLORMAP_HOT)
    
    def visualize_depth_anything(self, depth_frame: np.ndarray) -> np.ndarray:
        """Визуализация DepthAnything"""
        colored = self.cmap(depth_frame)[:, :, :3] * 255
        return colored[:, :, ::-1].astype(np.uint8)
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                        depth_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Отрисовка обнаруженных объектов"""
        vis_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            color = (0, 0, 255) 
            
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            self._draw_label(vis_image, label, (int(x1), int(y1)), color)
            
            if depth_frame is not None:
                self._draw_depth(vis_image, det['center'], depth_frame, 
                                (int(x1), int(y2)), color)
        
        return vis_image
    
    def _draw_label(self, image: np.ndarray, label: str, position: Tuple, 
                   color: Tuple):
        """Отрисовка текстовой метки"""
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 2, 2
        )
        
        x, y = position
        cv2.rectangle(
            image,
            (x, y - label_height - baseline - 5),
            (x + label_width, y),
            color,
            -1
        )
        
        cv2.putText(
            image,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2
        )
    
    def _draw_depth(self, image: np.ndarray, center: Tuple, 
                   depth_frame: np.ndarray, position: Tuple, color: Tuple):
        """Отрисовка информации о глубине"""
        cx = int(center[0])
        cy = int(center[1])
        
        if (0 <= cx < depth_frame.shape[1] and 
            0 <= cy < depth_frame.shape[0]):
            depth_value = depth_frame[cy, cx]
            if depth_value > 0:
                self._draw_label(image,f"Depth: {depth_value}mm",position,color)
    
    def show_frames(self, frames: Dict):
        """Отображение всех кадров"""
        for window_key, frame in frames.items():
            window_name = self.config.window_names[window_key]
            cv2.imshow(window_name, frame)