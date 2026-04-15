import depthai as dai
import numpy as np
from .config import Config

class CameraManager:
    def __init__(self):
        self.pipeline = self._create_pipeline()
        self.device = None
        self.q_left = None
        self.q_right = None
        self.q_middle = None

        self.baseline_cm = 7.51
        self.focal_length = 397.6625061035156

    def _create_pipeline(self):
        pipeline = dai.Pipeline()

        left = pipeline.create(dai.node.MonoCamera)
        right = pipeline.create(dai.node.MonoCamera)
        color = pipeline.create(dai.node.ColorCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        color.setBoardSocket(dai.CameraBoardSocket.CENTER)
        
        left.setResolution(Config.RES_MONO)
        right.setResolution(Config.RES_MONO)
        
        color.setResolution(Config.RES_COLOR)
        color.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CENTER)

        left.out.link(stereo.left)
        right.out.link(stereo.right)

        xout_left = pipeline.create(dai.node.XLinkOut)
        xout_left.setStreamName("left")
        xout_right = pipeline.create(dai.node.XLinkOut)
        xout_right.setStreamName("right")
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")

        stereo.rectifiedLeft.link(xout_left.input)
        stereo.rectifiedRight.link(xout_right.input)
        color.video.link(xout_rgb.input)

        return pipeline

    def start(self):
        self.device = dai.Device(self.pipeline)
        self.device.setIrLaserDotProjectorBrightness(800)
        
        self._read_device_calibration()

        self.q_left = self.device.getOutputQueue("left", 1, False)
        self.q_right = self.device.getOutputQueue("right", 1, False)
        self.q_middle = self.device.getOutputQueue("rgb", 1, False)
        return self

    def _read_device_calibration(self):
        """
        Считывает точное фокусное расстояние и базу из EEPROM камеры.
        """
        try:
            calibData = self.device.readCalibration()
            
            width = 1280 if Config.RES_MONO == dai.MonoCameraProperties.SensorResolution.THE_800_P else 640
            height = 800 if width == 1280 else 400
            
            intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, width, height)
            self.focal_length = intrinsics[0][0]
            
            extrinsics = calibData.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT)

            self.baseline_cm = abs(extrinsics[1][0])
        except Exception as e:
            print(f"Error reading calibration: {e}. Using defaults.")

    def get_calibration(self):
        """Возвращает параметры для формулы глубины"""
        return self.focal_length, self.baseline_cm

    def get_frames(self):
        if not self.device: return None, None, None
        l = self.q_left.get().getCvFrame()
        r = self.q_right.get().getCvFrame()
        rgb = self.q_middle.get().getCvFrame()
        return l, r, rgb
    
    def close(self):
        if self.device: self.device.close()