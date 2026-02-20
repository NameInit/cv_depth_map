import blobconverter
import os

# Конвертация OpenVINO модели в blob
blob_path = blobconverter.from_openvino(
    xml='./openvino_model/yolov8n.xml',      # Путь к .xml файлу
    bin='./openvino_model/yolov8n.bin',      # Путь к .bin файлу
    data_type='fp16',                         # Тип данных (fp16 или fp32)
    shaves=6,                                  # Количество SHAVE ядер (для OAK-D Pro)
    version='2023.2',                          # Версия MyriadX
    output_dir='./blobs'                       # Папка для сохранения
)

print(f'✓ Blob сохранён: {blob_path}')