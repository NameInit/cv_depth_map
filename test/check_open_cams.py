import cv2
import os

def list_available_cameras(max_cameras=10):
    """Проверка доступных камер"""
    available_cameras = []
    
    print("Проверка доступных камер...")
    print("-" * 50)
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"✅ Камера {i} доступна")
                
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"   Разрешение: {width} x {height}")
            else:
                print(f"⚠️ Камера {i} открыта, но не дает кадры")
            cap.release()
        else:
            print(f"❌ Камера {i} недоступна")
    
    print("-" * 50)
    print(f"Доступные камеры: {available_cameras}")
    
    print("\nПроверка устройств в /dev/:")
    video_devices = [f for f in os.listdir('/dev/') if f.startswith('video')]
    print(f"Найдены устройства: {video_devices}")
    
    return available_cameras

if __name__ == "__main__":
    list_available_cameras()