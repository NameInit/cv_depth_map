import depthai as dai
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
import torch
import cv2
import numpy as np
import matplotlib
import time

'''
Для OAK-D PRO:
CAM_A - центральная цветная камера (RGB)
CAM_B - правая монохромная (правая)
CAM_C - левая монохромная (левая)
'''

'''
model_configs = {
	'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
	'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
	'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
	'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
'''

model_detect_path = "model_pt/"
model_detect_name = "yolov8n.pt"
model_depth_anything_path = "model_depth_anything/"
model_depth_anything_name = "depth_anything_v2_vits.pth"

def run_pipeline():
	DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

	pipeline = dai.Pipeline()

	model_detect = YOLO(model_detect_path+model_detect_name)
	
	model_depth_anything = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
	model_depth_anything.load_state_dict(torch.load(model_depth_anything_path+model_depth_anything_name, map_location='cpu'))
	model_depth_anything = model_depth_anything.to(DEVICE).eval()

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

	cmap = matplotlib.colormaps.get_cmap('Spectral_r')

	with dai.Device(pipeline) as device:
		cur_brightness=800
		depth_queue:dai.DataOutputQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
		rgb_queue:dai.DataOutputQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

		fps_counter = 0
		fps = 0
		fps_timer = time.time()
		fps_update_interval = 1.0

		device.setIrLaserDotProjectorBrightness(cur_brightness)
		print(f"Яркость: {cur_brightness}")

		cv2.namedWindow("Depth")
		cv2.namedWindow("RGB")
		cv2.namedWindow("DepthAnythingV2")
		cv2.namedWindow("Main")
		
		cv2.moveWindow("Depth", 100, 0)
		cv2.moveWindow("RGB", 1050, 0)
		cv2.moveWindow("DepthAnythingV2", 100, 600)
		cv2.moveWindow("Main", 1050, 600)

		while True:
			depth_data = depth_queue.get()
			depth_frame = depth_data.getCvFrame()
			 
			rgb_data = rgb_queue.get()
			rgb_frame = rgb_data.getCvFrame()

			depth_anything_data=model_depth_anything.infer_image(rgb_frame)
			depth_anything_data = (depth_anything_data - depth_anything_data.min()) / (depth_anything_data.max() - depth_anything_data.min()) * 255.0
			depth_anything_data = depth_anything_data.astype(np.uint8)
			depth_anything_data = (cmap(depth_anything_data)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

			rgb_frame_vis = rgb_frame.copy()

			res_model = model_detect(rgb_frame, verbose = False)
			if len(res_model):
				result = res_model[0]
				if result.boxes is not None:
					boxes = result.boxes.xyxy.cpu().numpy()
					confidences = result.boxes.conf.cpu().numpy()
					class_ids = result.boxes.cls.cpu().numpy().astype(int)

					for box, conf, class_id in zip(boxes, confidences, class_ids):
						if conf > 0.5:
							x1, y1, x2, y2 = box.astype(int)
							
							class_name = result.names[class_id] if hasattr(result, 'names') else f"Class_{class_id}"
							
							color = (0,0,255)
							
							cv2.rectangle(rgb_frame_vis, (x1, y1), (x2, y2), color, 4)
							
							label = f"{class_name}: {conf:.2f}"
							
							(label_width, label_height), baseline = cv2.getTextSize(
								label, cv2.FONT_HERSHEY_SIMPLEX, 2, 2
							)
							
							cv2.rectangle(
								rgb_frame_vis,
								(x1, y1 - label_height - baseline - 5),
								(x1 + label_width, y1),
								color,
								-1
							)
							
							cv2.putText(
								rgb_frame_vis,
								label,
								(x1, y1 - 5),
								cv2.FONT_HERSHEY_SIMPLEX,
								2,
								(255, 255, 255),
								2
							)
							
							if depth_frame is not None:
								center_x = (x1 + x2) // 2
								center_y = (y1 + y2) // 2
								
								if (0 <= center_x < depth_frame.shape[1] and 
									0 <= center_y < depth_frame.shape[0]):
									depth_value = depth_frame[center_y, center_x]
									
									if depth_value > 0:
										cv2.putText(
											rgb_frame_vis,
											f"Depth: {depth_value}mm",
											(x1, y2 + 20),
											cv2.FONT_HERSHEY_SIMPLEX,
											0.5,
											color,
											2
										)

			depth_frame_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
			depth_frame_vis = cv2.applyColorMap(depth_frame_vis, cv2.COLORMAP_HOT)
			
			depth_frame_vis = cv2.resize(depth_frame_vis, (800, 400))
			rgb_frame_vis = cv2.resize(rgb_frame_vis, (800, 400))
			depth_anything_data = cv2.resize(depth_anything_data, (800, 400))
			rgb_frame = cv2.resize(rgb_frame, (800, 400))

			################FPS################
			fps_counter += 1
			current_time = time.time()
			time_diff = current_time - fps_timer

			if time_diff >= fps_update_interval:
				fps = fps_counter / time_diff
				fps_counter = 0
				fps_timer = current_time

			fps_text = f"FPS: {fps:.1f}"
			cv2.putText(
				rgb_frame,
				fps_text,
				(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.0,
				(0, 255, 0),
				2
			)
			###################################

			cv2.imshow("Depth", depth_frame_vis)
			cv2.imshow("RGB", rgb_frame_vis)
			cv2.imshow("DepthAnythingV2", depth_anything_data)
			cv2.imshow("Main", rgb_frame)

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