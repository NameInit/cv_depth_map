import time

class FPSCounter:
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.counter = 0
        self.fps = 0
        self.timer = time.time()
        
    def update(self) -> float:
        """Обновление счетчика и получение текущего FPS"""
        self.counter += 1
        current_time = time.time()
        time_diff = current_time - self.timer
        
        if time_diff >= self.update_interval:
            self.fps = self.counter / time_diff
            self.counter = 0
            self.timer = current_time
            
        return self.fps
    
    def get_fps_text(self) -> str:
        """Получение текста для отображения FPS"""
        return f"FPS: {self.fps:.1f}"