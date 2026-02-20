from ultralytics import YOLO
import sys
import torch

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} [MODEL.pt]")
        sys.exit(1)
    
    try:
        model = YOLO(sys.argv[1])
        
        model.export(
            format="onnx",
            imgsz=640,
            opset=11,
            half=False, 
            simplify=True,
            dynamic=False
        )
        print(f"✓ Модель экспортирована: {sys.argv[1][:-3]}.onnx")
        
    except Exception as e:
        print(f"✗ Ошибка: {e}")