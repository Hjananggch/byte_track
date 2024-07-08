# 导入需要的包
import pandas as pd
import torch
import numpy as np
import cv2
from dataclasses import dataclass
from track.byte_tracker import BYTETracker
from collections import defaultdict,deque
from flask import Flask, request, jsonify
from detect_type import *
import asyncio
from threading import Thread

app = Flask(__name__)


@dataclass
class BYTETrackerArgs:
    track_thresh: float = 0.6  # 追踪阈值
    track_buffer: int = 60  # 追踪缓冲区大小
    match_thresh: float = 0.5  # 匹配阈值#3.2
    aspect_ratio_thresh: float = 3.5  # 方向比例阈值
    min_box_area: float = 1.0  # 最小框面积
    mot20: bool = False  # 是否使用MOT20指标


# 初始化跟踪器
byte_tracker = BYTETracker(BYTETrackerArgs())

model = torch.hub.load(r'./yolo_move', 'custom'
                       , path=r'./bigf.pt', source='local')  # 加载模型2
model.conf = 0.25  # 设置置信度2
model.iou = 0.45  # 设置iou|2




queue_frame = deque(maxlen=20*15)
#queue_frame_box = deque(maxlen=20*15)
queue_frame_detect = deque(maxlen=20*15)
# 主函数1
def main(path,file):
    stream_data = {'track_history': defaultdict(list)}  # 创建历史轨迹字典3
    id_data = {'id_i': defaultdict(list),'id_1': defaultdict(list),'id_2': defaultdict(list),'id_o': defaultdict(list)}
    num_m = 0
    num = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    q = 0
    qf = []
    qfd = []
    filtered_detections = np.array([])
    
    
    vio = cv2.VideoCapture(path)  # 读取视频1
    
    while True:
        if q % 1 == 0:
            ret, frame = vio.read()  # 读取视频帧1
            if ret == False:  # 如果未读取成功1
                break
            frame0 = frame
            frame_width = int(frame.shape[0] * 0.5)
            frame_height = int(frame.shape[1] * 0.5)
            resized_frame_1 = cv2.resize(frame, (frame_height, frame_width))
            frame1 = compress_frame(resized_frame_1)
            queue_frame.append(frame1)
            #frame = cv2.resize(frame, (640, 480))  # 调整帧大小1
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 调整帧颜色1
            results = model(frame)  # 使用模型预测2
            frame = detect_box(results, frame)  # 将视频文件和预测结果导入此函数进行处理2
            results = results.xyxy[0].cpu().numpy()  # 3
            
            frame1, polygon1 = read_jiso(frame,file)
            filtered_detections = results[results[:, 5] == 0]  # 选出class=0 对象的信息3
            # print(filtered_detections)
            # print(filtered_detections.size)
            if filtered_detections.size > 0:  #如果检测到目标3
                #det_info = filtered_detections[:, :4]#获取对象坐标信息3
                #print(det_info)
                #confidences = filtered_detections[:, 4]#获取置信度3
                #print(confidences)
                byte_track_input = filtered_detections[:, :5]  #获取目标坐标和置信度3.1
                #print(tiqu)
                #byte_track_input = np.hstack((det_info, confidences[:, None]))#将坐标和置信度整合进列表3
                #print(byte_track_input)
                tracks = byte_tracker.update(byte_track_input, frame.shape, frame.shape)  #向跟踪器传入目标信息和帧大小3
                #print(frame.shape)
                #print(tracks)

                for detection, track in zip(filtered_detections, tracks):  #将检测信息和轨迹信息（通过zip）进行匹配3
                    #print(detection, track)
                    track_id = track.track_id  #获取轨迹ID|3
                    #print(track_id)
                    bbox = track.tlbr.astype(np.int32)  #获取轨迹框坐标3
                    #print(bbox)
                    x_center, y_center = bbox[0] + (bbox[2] - bbox[0]) // 2, bbox[1] + (
                            bbox[3] - bbox[1]) // 2  #获取轨迹框中心坐标3
                    #print(x_center, y_center)
                    track = stream_data['track_history'][track_id]  #更新轨迹历史3
                    track.append((x_center, y_center))  #向track添加中心点坐标3
                    if len(track) > 200:  #设置轨迹历史长度3
                        track.pop(0)  #移除最早的坐标3
                    #print(track)
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))  #将轨迹历史转换为点坐标3
                    #print(points)
                    cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=1)  #绘制轨迹3
                    cv2.putText(frame, '{}'.format(track_id), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 20, 0), 2)  #为检测目标添加ID3

                    s_et = [630, 483, 735, 510]
                    nu,nu_1 = region_detect(frame,s_et,track_id,id_data['id_i'],x_center,y_center)
                    num = num + nu
                    num_1 = num_1 + nu_1
                    s_et = [550, 550, 735, 600]
                    nu, nu_1 = region_detect(frame,s_et,track_id,id_data['id_1'],x_center,y_center)
                    num_1 = num_1 - nu
                    num_2 = num_2 + nu_1
                    s_et = [420, 728, 675, 760]
                    nu, nu_1 = region_detect(frame,s_et,track_id,id_data['id_2'],x_center,y_center)
                    num_2 = num_2 - nu
                    num_3 = num_3 + nu_1
                    s_et = [225, 887, 600, 927]
                    nu, nu_1 = region_detect(frame,s_et,track_id,id_data['id_o'],x_center,y_center)
                    num = num - nu
                    num_3 = num_3 - nu

                    cv2.putText(frame, 'car:{}'.format(num), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 20, 0), 2)
                    cv2.putText(frame, 'part_{}: {}'.format(1, num_1), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 20, 0),2)
                    cv2.putText(frame, 'part_{}: {}'.format(2, num_2), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 20, 0),2)
                    cv2.putText(frame, 'part_{}: {}'.format(3, num_3), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 20, 0),2)
                   
            for box in results:
                center = Point(box[2],box[3])
                if center is not None:
                    if any(polygon.contains(center) for polygon in polygon1):
                        resized_frame_3 = cv2.resize(frame, (frame_height, frame_width))
                        frame3 = compress_frame(resized_frame_3)
                        queue_frame_detect.append(frame3)
                        num_m = num_m + 1
                        if not any(polygon.contains(center) for polygon in polygon1) or num_m==300:
                            save_car(queue_frame, queue_frame_detect, frame0)
                            num_m = 0
                            


            frame = cv2.resize(frame, (640, 480))
            cv2.imshow("frame", frame)  #播放视频1
            cv2.waitKey(1)  #延时，否则无法正常显示1
        q = q + 1


def detected(data):
    main(data['path'],data['file_json'])


# 处理来自客户端的请求
@app.route('/jishu', methods=['POST'])
def proccess_data():
    data = request.get_json()
    detected(data)
    return jsonify({'success': 200})



if __name__ == '__main__':
    path = r'./car_detecting.mp4'#输入视频路径1
    #path = r'rtmp://203.34.56.118:1938/live/c2de1ff4-e2ff-498f-ab6b-c70c180db163'
    file = r"C:\\Users\A\Desktop\\2.json"
    main(path,file)#执行主函数1

'''
if '__main__' == __name__:
    app.run(host='0.0.0.0', port=8080, debug=True)
'''