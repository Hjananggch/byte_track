import torch
from tracker.byte_tracker import BYTETracker
from dataclasses import dataclass
import cv2
import numpy as np
from collections import defaultdict

@dataclass
class BYTETrackerArgs:
    track_thresh: float = 0.6
    track_buffer: int = 60
    match_thresh: float = 0.9
    aspect_ratio_thresh: float = 3.5
    min_box_area: float = 1.0
    mot20: bool = False

byte_tracker = BYTETracker(BYTETrackerArgs())

model = torch.hub.load(r'E:\yolov5-master', 'custom'
                       ,path=r'E:\yolov5-master\spy_5.10.pt',source='local')
model.conf = 0.25
model.iou = 0.45
num = 0


def detect_box(results, frame):
    results_ = results.pandas().xyxy[0].to_numpy()
    color = (0, 0, 255)
    for box in results_:
        l, t, r, b = box[:4].astype('int')
        if box[6] == 'person' or box[6] == 'motorcycle' or box[6] == 'drip':
            cv2.rectangle(frame, (l, t), (r, b), color, 1)
    return frame


def main(path):
    stream_data = {
        'track_history': defaultdict(list)
    }
    cap = cv2.VideoCapture(path)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        # predict(frame,results,stream_data,num)
        frame = detect_box(results, frame)
        results = results.xyxy[0].cpu().numpy()

        filtered_detections = results[results[:, 5] == 0]
        default_color = (251, 238, 1)
        if filtered_detections.size > 0:
            det_info = filtered_detections[:, :4]
            confidences = filtered_detections[:, 4]

            byte_track_input = np.hstack((det_info, confidences[:, None]))
            tracks = byte_tracker.update(byte_track_input, frame.shape, frame.shape)

            for detection, track in zip(filtered_detections, tracks):
                track_id = track.track_id

                bbox = track.tlbr.astype(np.int32)
                x_center, y_center = bbox[0] + (bbox[2] - bbox[0]) // 2, bbox[1] + (bbox[3] - bbox[1]) // 2

                track = stream_data['track_history'][track_id]
                track.append((x_center, y_center))
                if len(track) > 25:
                    track.pop(0)
                a = len(track)
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=1)

                if track[-1][0] - track[a - 2][0] < 2 and track[-1][1] - track[a - 2][1] < 2:
                    frame_num +=1
                    if frame_num >= 3:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                        cv2.putText(frame, 'stop', (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), default_color, 2)
                        cv2.putText(frame, str(track_id), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    default_color, 2)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    cap.release()

if __name__ == '__main__':
    path = r'C:\Users\wqs\Desktop\byte_track-main\total - Trim.mp4'
    main(path)

