import depthai as dai

class Config:
    MODEL_PATH = ".models/model_pt/yolov10n.pt"

    RES_MONO = dai.MonoCameraProperties.SensorResolution.THE_400_P
    RES_COLOR = dai.ColorCameraProperties.SensorResolution.THE_1080_P
    
    BASELINE_CM = 8.0
    FOCAL_LENGTH = 441.25
    MAX_DIST_M = 12.0
    
    SGBM_WINDOW_SIZE = 7
    SGBM_NUM_DISP = 160