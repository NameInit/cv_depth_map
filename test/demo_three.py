import depthai as dai
import cv2
import numpy as np

'''
Для OAK-D PRO:
CAM_A - центральная цветная камера (RGB)
CAM_B - правая монохромная (правая)
CAM_C - левая монохромная (левая)
'''

mxid_oak: str = "184430105105351300"

def run_pipeline():
    if not len(dai.Device.getAllAvailableDevices()) or \
        all(dev.mxid != mxid_oak for dev in dai.Device.getAllAvailableDevices()):
        print(f"Device {mxid_oak} not found.")
        return
    
    device_info = dai.DeviceInfo(mxid_oak)
    pipeline = dai.Pipeline()
    
    left_cam = pipeline.create(dai.node.MonoCamera)
    right_cam = pipeline.create(dai.node.MonoCamera)
    middle_cam = pipeline.create(dai.node.ColorCamera)
    
    left_cam.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right_cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    middle_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    
    left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    right_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    middle_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    
    middle_cam.setInterleaved(False)
    middle_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    
    xout_left = pipeline.create(dai.node.XLinkOut)
    xout_left.setStreamName("left")
    xout_right = pipeline.create(dai.node.XLinkOut)
    xout_right.setStreamName("right")
    xout_middle = pipeline.create(dai.node.XLinkOut)
    xout_middle.setStreamName("middle")
    
    left_cam.out.link(xout_left.input)
    right_cam.out.link(xout_right.input)
    middle_cam.video.link(xout_middle.input)
    
    with dai.Device(pipeline, device_info) as device:
        right_queue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
        left_queue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        middle_queue = device.getOutputQueue(name="middle", maxSize=4, blocking=False)

        while True:
            left_frame_data = left_queue.get()
            right_frame_data = right_queue.get()
            middle_frame_data = middle_queue.get()
            
            left_frame = left_frame_data.getCvFrame()
            right_frame = right_frame_data.getCvFrame()
            middle_frame = middle_frame_data.getCvFrame()
            
            left_frame_resized = cv2.resize(left_frame, (640, 360))
            right_frame_resized = cv2.resize(right_frame, (640, 360))
            middle_frame_resized = cv2.resize(middle_frame, (640, 360))
            
            cv2.imshow("OAK-D Camera LEFT", left_frame_resized)
            cv2.imshow("OAK-D Camera RIGHT", right_frame_resized)
            cv2.imshow("OAK-D Camera MIDDLE", middle_frame_resized)
            
            if cv2.waitKey(1) == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()