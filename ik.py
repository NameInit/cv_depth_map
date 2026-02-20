import depthai as dai
import cv2

def run_pipeline():
    pipeline = dai.Pipeline()

    rgb_center_cam = pipeline.create(dai.node.ColorCamera)
    mono_left_cam = pipeline.create(dai.node.MonoCamera)
    mono_right_cam = pipeline.create(dai.node.MonoCamera)
    stereo_depth = pipeline.create(dai.node.StereoDepth)

    stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo_depth.setDepthAlign(dai.CameraBoardSocket.CENTER)

    mono_left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    rgb_center_cam.setBoardSocket(dai.CameraBoardSocket.CENTER)

    mono_left_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    rgb_center_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    rgb_center_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
 
    mono_left_cam.out.link(stereo_depth.left)
    mono_right_cam.out.link(stereo_depth.right)

    stereo_depth_out = pipeline.create(dai.node.XLinkOut)
    stereo_depth_out.setStreamName("depth")
    stereo_depth.depth.link(stereo_depth_out.input)

    rgb_out = pipeline.create(dai.node.XLinkOut)
    rgb_out.setStreamName("rgb")
    rgb_center_cam.video.link(rgb_out.input)

    with dai.Device(pipeline) as device:
        cur_brightness=800
        depth_queue:dai.DataOutputQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        rgb_queue:dai.DataOutputQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        device.setIrLaserDotProjectorBrightness(cur_brightness)
        print(f"Яркость: {cur_brightness}")

        while True:
            depth_data = depth_queue.get()
            depth_frame = depth_data.getCvFrame()
             
            rgb_data = rgb_queue.get()
            rgb_frame = rgb_data.getCvFrame()


            depth_frame_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            depth_frame_vis = cv2.applyColorMap(depth_frame_vis, cv2.COLORMAP_HOT)
            
            depth_frame_vis = cv2.resize(depth_frame_vis, (800, 400))
            rgb_frame_vis = cv2.resize(rgb_frame, (800, 400))

            cv2.imshow("Depth", depth_frame_vis)
            cv2.imshow("RGB", rgb_frame_vis)

            cv2.moveWindow("Depth", 100, 50)
            cv2.moveWindow("RGB", 100, 570)

            key = cv2.waitKey(1)
            if key == ord('w'):
                cur_brightness-=200
                cur_brightness= 0 if cur_brightness<0 else cur_brightness
                device.setIrLaserDotProjectorBrightness(cur_brightness)
                print(f"Яркость: {cur_brightness}")
            elif key == ord('e'):
                cur_brightness+=200
                cur_brightness=1200 if cur_brightness>1200 else cur_brightness
                device.setIrLaserDotProjectorBrightness(cur_brightness)
                print(f"Яркость: {cur_brightness}")
            elif key == ord('q'):
                break

if __name__ == "__main__":
    run_pipeline()