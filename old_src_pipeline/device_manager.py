import depthai as dai
from typing import Tuple
import cv2
import numpy as np

class OAKCameraManager:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.device = None
        self.queues = {}
        
    def create_pipeline(self) -> dai.Pipeline:
        """Создание пайплайна OAK камеры"""
        pipeline = dai.Pipeline()
        
        rgb_cam = pipeline.create(dai.node.ColorCamera)
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        
        stereo.setDefaultProfilePreset(
            getattr(dai.node.StereoDepth.PresetMode, self.config.depth_preset)
        )
        stereo.setDepthAlign(dai.CameraBoardSocket.CENTER)
        
        self._configure_cameras(rgb_cam, mono_left, mono_right)
        
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        self._create_output_streams(pipeline, stereo, rgb_cam)
        
        self.pipeline = pipeline
        return pipeline
    
    def _configure_cameras(self, rgb_cam, mono_left, mono_right):
        """Конфигурация отдельных камер"""
        rgb_cam.setBoardSocket(dai.CameraBoardSocket.CENTER)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        rgb_cam.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1080_P
        )
        mono_left.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P
        )
        mono_right.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P
        )
        
        rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    
    def _create_output_streams(self, pipeline, stereo, rgb_cam):
        """Создание выходных потоков"""
        depth_out = pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName("depth")
        stereo.depth.link(depth_out.input)
        
        rgb_out = pipeline.create(dai.node.XLinkOut)
        rgb_out.setStreamName("rgb")
        rgb_cam.video.link(rgb_out.input)
    
    def start(self):
        """Запуск устройства"""
        if not self.pipeline:
            self.create_pipeline()
        
        self.device = dai.Device(self.pipeline)
        self.queues = {
            'depth': self.device.getOutputQueue(
                name="depth", maxSize=4, blocking=False
            ),
            'rgb': self.device.getOutputQueue(
                name="rgb", maxSize=4, blocking=False
            )
        }
        
    def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Получение кадров"""
        depth_frame = self.queues['depth'].get().getCvFrame()
        rgb_frame = self.queues['rgb'].get().getCvFrame()
        return depth_frame, rgb_frame
    
    def set_ir_brightness(self, brightness: int):
        """Установка яркости ИК-лазера"""
        brightness = max(self.config.ir_brightness_min, 
                        min(brightness, self.config.ir_brightness_max))
        self.device.setIrLaserDotProjectorBrightness(brightness)
        return brightness
    
    def stop(self):
        """Остановка устройства"""
        if self.device:
            del self.device