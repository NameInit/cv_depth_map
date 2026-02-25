import cv2
import numpy as np

def get_distance_in_roi(distance_matrix, center_x, center_y, roi_size=5):
    """Считает медианную дистанцию в области вокруг центра"""
    h, w = distance_matrix.shape
    
    x_min = max(0, center_x - roi_size)
    x_max = min(w, center_x + roi_size)
    y_min = max(0, center_y - roi_size)
    y_max = min(h, center_y + roi_size)
    
    roi = distance_matrix[y_min:y_max, x_min:x_max]
    
    valid = roi[(roi > 0.2) & (roi < 12.0)]
    
    if len(valid) > 0:
        return np.median(valid)
    return 0.0

def draw_results(frame, results, distance_map):
    """Рисует боксы и дистанцию на RGB кадре"""
    h_rgb, w_rgb = frame.shape[:2]
    h_depth, w_depth = distance_map.shape[:2]
    
    scale_x = w_depth / w_rgb
    scale_y = h_depth / h_rgb
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label_name = r.names[cls_id]
            
            cx_rgb = (x1 + x2) // 2
            cy_rgb = (y1 + y2) // 2
            
            cx_depth = int(cx_rgb * scale_x)
            cy_depth = int(cy_rgb * scale_y)
            
            dist = get_distance_in_roi(distance_map, cx_depth, cy_depth)
            
            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            text = f"{label_name} {dist:.2f}m" if dist > 0 else f"{label_name}"
            
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + tw, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
    return frame