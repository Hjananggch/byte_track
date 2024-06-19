import torch
from tracker.byte_tracker import BYTETracker
from dataclasses import dataclass
import cv2
import numpy as np
from detect_type import *
from flask import Flask, request, jsonify
from collections import deque, defaultdict
import datetime

app = Flask(__name__)

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
n_mot = 0
n_pre = 0

queue_frame = deque(maxlen=20*15)
queue_frame_box = deque(maxlen=20*15)

def main(path_d,detect_type,activate,file):

    stream_data = {
        'track_history': defaultdict(list)
    }

    detect_time = {
        'chufa_num': defaultdict(list)
    }

    stop_time = {
        'time_stop': defaultdict(list)
    }
    cap = cv2.VideoCapture(path_d)

    #检测视频保存
    num_p = 0
    num_m = 0
    chu_fa = 0
    motor = []
    person = []
    filtered_detections = np.array([])
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame0 = frame
        #目标检测框
        # 图片大小
        frame_width = int(frame.shape[0] * 0.5)
        frame_height = int(frame.shape[1] * 0.5)

        # 原始画面
        resized_frame_1 = cv2.resize(frame, (frame_height, frame_width))
        frame1 = compress_frame(resized_frame_1)
        queue_frame.append(frame1)

        results = model(frame)
        # predict(frame,results,stream_data,num)

        results_1 = results.xyxy[0].cpu().numpy()

        # file_path = r"C:\Users\wqs\Desktop\daolu\26.json"
        frame1, polygon1 = read_jiso(frame,file)
        for box in results_1:
            center = Point(box[2],box[3])
            if center is not None:
                if any(polygon.contains(center) for polygon in polygon1):
                    frame, motor, person = detect_box(results, frame, n_mot, n_pre)
                    filtered_detections = results_1[results_1[:, 5] == 0]
        # 处理后画面
        resized_frame_2 = cv2.resize(frame, (frame_height, frame_width))
        frame2 = compress_frame(resized_frame_2)
        queue_frame_box.append(frame2)

        # 非机动车检测保存
        if len(motor) > 0 and motor[0][6] == 'motorcycle' :
            num_m += 1
            if num_m > 20:
                detect_motorcycle(queue_frame, queue_frame_box, frame0)
                num_m = 0
            # print('motor',num_m)
        # 行人检测保存

        if len(person) > 0 and person[0][6] == 'person':

            num_p += 1
            if num_p > 80:
                # print('人检测启动')
                detect_person( queue_frame, queue_frame_box, frame0)
                num_p = 0
            # print('person',num_p)

            # default_color = (251, 238, 1)

        if filtered_detections.size > 0:
            det_info = filtered_detections[:, :4]
            confidences = filtered_detections[:, 4]
            byte_track_input = np.hstack((det_info, confidences[:, None]))
            tracks = byte_tracker.update(byte_track_input, frame.shape, frame.shape)

            #目标id
            for detection, track in zip(filtered_detections, tracks):
                track_id = track.track_id
                bbox = track.tlbr.astype(np.int32)
                x_center, y_center = bbox[0] + (bbox[2] - bbox[0]) // 2, bbox[1] + (bbox[3] - bbox[1]) // 2

                track = stream_data['track_history'][track_id]
                track_num = detect_time['chufa_num'][track_id]
                track_stop = stop_time['time_stop'][track_id]

                track.append((x_center, y_center))

                a = len(track)
                if len(track) > 25:
                    track.pop(0)
                if len(track_num) > 1:
                    track_num.pop(0)

                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=1)

                if abs(track[-1][0] - track[a-2][0] )< 2 and abs(track[-1][1] - track[a-2][1]) < 2:
                    # print((time.time() - track_time[0]))
                    chu_fa +=1
                    track_num.append(chu_fa)
                    if len(track_num) > 0:
                        # print(track_num)
                        if track_num[0] > 50:
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                            cv2.putText(frame, 'stop', (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                            detect_cars_stop(frame , queue_frame, queue_frame_box, frame0 ,frame_height, frame_width , track_num , track_stop , track_id)
                            if track_num[0] > 260:
                                chu_fa = 50

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()

def detected(data):
    # if data['detect_type'] == 1:
    main(data['path'],data['detect_type'],data['activate'],data['file_json'])

@app.route('/tingche',methods=['POST'])
def proccess_data():
    data =request.get_json()
    detected(data)
    return jsonify({'success': 200})


if '__main__' == __name__:
    app.run(host='0.0.0.0', port=8080, debug=True)



