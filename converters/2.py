import blobconverter
import os
import subprocess

def onnx_to_blob(onnx_path='yolov8n.onnx'):
    """Конвертация ONNX в blob для OAK-D"""
    
    # 1. Создаем папки
    os.makedirs('./openvino_model', exist_ok=True)
    os.makedirs('./blobs', exist_ok=True)
    
    # 2. Конвертация в OpenVINO (исправленная команда)
    print("1. Конвертация в OpenVINO FP16...")
    subprocess.run([
        'mo',
        '--input_model', onnx_path,
        '--output_dir', './openvino_model',
        '--compress_to_fp16'  # Вместо --data_type FP16
    ], check=True)
    
    # 3. Конвертация в blob
    print("2. Конвертация в blob...")
    blob_path = blobconverter.from_openvino(
        xml='./openvino_model/yolov8n.xml',
        bin='./openvino_model/yolov8n.bin',
        data_type='fp16',
        shaves=6,
        version='2023.2',
        output_dir='./blobs'
    )
    
    print(f"✓ Готово! Blob: {blob_path}")
    return blob_path

if __name__ == "__main__":
    onnx_to_blob('yolov8n.onnx')