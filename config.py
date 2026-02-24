from dataclasses import dataclass
from pathlib import Path

@dataclass
class CameraConfig:
    rgb_resolution: tuple = (1920, 1080)
    mono_resolution: tuple = (640, 400)
    depth_preset: str = "HIGH_ACCURACY"
    ir_brightness_min: int = 0
    ir_brightness_max: int = 1200
    ir_brightness_default: int = 800

@dataclass
class ModelConfig:
    detect_path: Path = Path("model_pt/")
    detect_name: str = "yolov8n.pt"
    depth_path: Path = Path("model_depth_anything/")
    depth_name: str = "depth_anything_v2_vits.pth"
    depth_encoder: str = 'vits'
    depth_features: int = 64
    depth_out_channels: list = None

    def __post_init__(self):
        if self.depth_out_channels is None:
            self.depth_out_channels = [48, 96, 192, 384]

@dataclass
class DisplayConfig:
    window_names: dict = None
    window_positions: dict = None
    
    def __post_init__(self):
        if self.window_names is None:
            self.window_names = {
                'depth': 'Depth',
                'rgb': 'RGB',
                'depth_anything': 'DepthAnythingV2',
                'main': 'Main'
            }
        if self.window_positions is None:
            self.window_positions = {
                'depth': (100, 0),
                'rgb': (1050, 0),
                'depth_anything': (100, 600),
                'main': (1050, 600)
            }