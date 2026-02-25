from ultralytics import YOLO
from config import Config

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(Config.MODEL_PATH)
    
    def detect(self, frame, conf=0.5):
        results = self.model(frame, verbose=False, conf=conf)
        return results