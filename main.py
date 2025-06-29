import cv2 as cv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import time

# Initializing our YOLO model
yolo = YOLO("model/best.pt")
tracker = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.4, nn_budget=100)

# Constraints for - minimum bounding box height and width , confidence level and IOU matching
MIN_W, MIN_H = 15, 35
CONF_THRESH = 0.2
IOU_MATCH_THRESHOLD = 0.3


label_positions = {}
id_map = {}  
track_boxes_prev = {}  
SMOOTHING_ALPHA = 0.6

# Function that is used to filter out referee from being detected as a player
def is_yellow_region(image, box):
    x1, y1, x2, y2 = box
    top = image[y1: y1 + (y2 - y1) // 3, x1:x2]
    if top.size == 0:
        return False
    hsv = cv.cvtColor(top, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (20, 100, 100), (40, 255, 255))
    ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return ratio > 0.5

# Taking the video as input
cap = cv.VideoCapture("15sec_input_720p.mp4")
if not cap.isOpened():
    raise RuntimeError("Unable to open input video")

# Creating a video writer object for the output video
W, H = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv.CAP_PROP_FPS) or 30.0
out = cv.VideoWriter("player_tracking.mp4", cv.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))



# Main loop for frame-by-frame detection
while True:
    ret, frame = cap.read()
    if not ret:
        break

   # Calling the results of YOLO predictions
    results = yolo.predict(frame, conf=CONF_THRESH, verbose=False)[0]
    yolo_boxes = []


    # Applying constraints
    for box in results.boxes:
        cls = int(box.cls[0])
        if yolo.names[cls] != "player":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        if w < MIN_W or h < MIN_H:
            continue
        if is_yellow_region(frame, (x1, y1, x2, y2)):
            continue
        yolo_boxes.append([x1, y1, x2, y2])

   # Appending detections to be sent to DeepSort
    detections = []
    for x1, y1, x2, y2 in yolo_boxes:
        detections.append([[x1, y1, x2 - x1, y2 - y1], 0.9, "player"])

    tracks = tracker.update_tracks(detections, frame=frame)

    drawn_boxes = []
    ids = []

    for tr in tracks:
        if not tr.is_confirmed():
            continue
        track_box = list(map(int, tr.to_ltrb()))
        original_tid = tr.track_id

        # Finding the best bounding box by IoU matching      
        best_iou, best_box = 0, None
        for xb in yolo_boxes:
            xa, ya, xb2, yb2 = xb
            inter_w = max(0, min(track_box[2], xb2) - max(track_box[0], xa))
            inter_h = max(0, min(track_box[3], yb2) - max(track_box[1], ya))
            area = inter_w * inter_h
            if area > 0:
                union = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1]) + \
                        (xb2 - xa) * (yb2 - ya) - area
                iou = area / union
                if iou > best_iou:
                    best_iou, best_box = iou, xb

        if best_iou < IOU_MATCH_THRESHOLD or best_box is None:
            continue

        x1, y1, x2, y2 = best_box

        # Conditional statement to avoid multiple ID assignments
        if original_tid in track_boxes_prev:
            prev_box = track_boxes_prev[original_tid]
            smoothed = [
                int(SMOOTHING_ALPHA * p + (1 - SMOOTHING_ALPHA) * c)
                for p, c in zip(prev_box, [x1, y1, x2, y2])
            ]
        else:
            smoothed = [x1, y1, x2, y2]
        track_boxes_prev[original_tid] = smoothed
        x1, y1, x2, y2 = smoothed

       
        if original_tid not in id_map:
            id_map[original_tid] = 11 + (hash(original_tid) % 89)
        mapped_id = id_map[original_tid]

       # Drawing the bounding boxes
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        drawn_boxes.append((x1, y1, x2, y2))
        ids.append(mapped_id)

    # Putting the text
    for box, tid in zip(drawn_boxes, ids):
        x1, y1, x2, y2 = box
        if tid not in label_positions:
            label_positions[tid] = 'above' if tid % 2 == 1 else 'below'

        label_pos = (x1, y1 - 10) if label_positions[tid] == 'above' else (x1, y2 + 20)
        cv.putText(frame, f"Player ID: {tid}", label_pos, cv.FONT_HERSHEY_SIMPLEX, 0.6, (255 , 255, 255), 2)

    
    out.write(frame)

# Releasing the created reader and writer objects

cap.release()
out.release()



