import blobconverter
import os

# Создаем папку для выходных файлов
os.makedirs("my_blobs", exist_ok=True)

# Компиляция с локальными инструментами
blob_path = blobconverter.from_onnx(
    model='yolov8n.onnx',
    data_type='fp16',
    shaves=6,
    version='2023.2',
    use_cache=False,
    output_dir='./my_blobs'
)

print(f'Blob сохранён: {blob_path}')