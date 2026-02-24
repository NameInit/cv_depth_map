import blobconverter
import os

# Скачать готовый blob (если есть в кэше)
blob_path = blobconverter.from_zoo(
    name='yolov8n_coco_640x640',
    zoo_type='depthai',
    shaves=6,
    version='2023.2'
)

print(f'Blob из zoo: {blob_path}')