import depthai as dai

class Config:
    MODEL_PATH = ".models/model_pt/yolov10n.pt"
    MODEL_CONF = 0.6

    RES_MONO = dai.MonoCameraProperties.SensorResolution.THE_400_P
    RES_COLOR = dai.ColorCameraProperties.SensorResolution.THE_1080_P
    
    BASELINE_CM = 7.51
    FOCAL_LENGTH = 397.6625 # for 640x400 if you need 1280x800: 795.3250 | OAK D PRO
    MAX_DIST_M = 20.0
    
    SGDM_SCALE = 0.7
    SGDM_WLS = True
    SGBM_WINDOW_SIZE = 7
    SGBM_NUM_DISP = 160