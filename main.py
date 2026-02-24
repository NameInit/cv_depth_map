import cv2
from config import CameraConfig, ModelConfig, DisplayConfig
from device_manager import OAKCameraManager
from models import DetectionModel, DepthModel
from visualizer import FrameVisualizer
from fps_counter import FPSCounter

class VisionSystem:
    def __init__(self):
        self.camera_config = CameraConfig()
        self.model_config = ModelConfig()
        self.display_config = DisplayConfig()
        
        self.camera = None
        self.detector = None
        self.depth_model = None
        self.visualizer = None
        self.fps_counter = None
        
        self.ir_brightness = self.camera_config.ir_brightness_default
        self.running = False
        
    def initialize(self):
        """Инициализация всех компонентов"""
        self.camera = OAKCameraManager(self.camera_config)
        self.camera.create_pipeline()
        self.camera.start()
        self.camera.set_ir_brightness(self.ir_brightness)
        
        self.detector = DetectionModel(
            self.model_config.detect_path,
            self.model_config.detect_name
        )
        
        self.depth_model = DepthModel(
            self.model_config.depth_encoder,
            self.model_config.depth_features,
            self.model_config.depth_out_channels,
            self.model_config.depth_path,
            self.model_config.depth_name
        )
        
        self.visualizer = FrameVisualizer(self.display_config)
        self.fps_counter = FPSCounter()
        
        print(f"Начальная яркость: {self.ir_brightness}")
        
    def process_frame(self, rgb_frame: cv2.Mat, depth_frame: cv2.Mat):
        """Обработка одного кадра"""
        depth_anything = self.depth_model.infer(rgb_frame)
        
        detections = self.detector.detect(rgb_frame)
        
        rgb_vis = self.visualizer.draw_detections(
            rgb_frame, detections, depth_frame
        )
        
        depth_vis = self.visualizer.visualize_depth_stereo(depth_frame)
        depth_anything_vis = self.visualizer.visualize_depth_anything(
            depth_anything
        )
        
        target_size = (800, 400)
        frames = {
            'depth': cv2.resize(depth_vis, target_size),
            'rgb': cv2.resize(rgb_vis, target_size),
            'depth_anything': cv2.resize(depth_anything_vis, target_size),
            'main': cv2.resize(rgb_frame, target_size)
        }
        
        return frames
        
    def handle_input(self, key: int) -> bool:
        """Обработка пользовательского ввода"""
        if key == ord('q'):
            return False
        elif key == ord('w'):
            self.ir_brightness = self.camera.set_ir_brightness(
                self.ir_brightness - 200
            )
            print(f"Яркость: {self.ir_brightness}")
        elif key == ord('e'):
            self.ir_brightness = self.camera.set_ir_brightness(
                self.ir_brightness + 200
            )
            print(f"Яркость: {self.ir_brightness}")
        return True
    
    def add_fps_to_frame(self, frame: cv2.Mat):
        """Добавление FPS на кадр"""
        fps_text = self.fps_counter.get_fps_text()
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        return frame
    
    def run(self):
        """Основной цикл программы"""
        self.running = True
        
        try:
            while self.running:
                depth_frame, rgb_frame = self.camera.get_frames()
                
                frames = self.process_frame(rgb_frame, depth_frame)
                
                self.fps_counter.update()
                frames['main'] = self.add_fps_to_frame(frames['main'])
                
                self.visualizer.show_frames(frames)
                
                key = cv2.waitKey(1) & 0xFF
                self.running = self.handle_input(key)
                
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Очистка ресурсов"""
        print("Завершение работы...")
        cv2.destroyAllWindows()
        if self.camera:
            self.camera.stop()

def main():
    system = VisionSystem()
    system.initialize()
    system.run()

if __name__ == "__main__":
    main()