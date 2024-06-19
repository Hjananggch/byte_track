import cv2
import datetime
import json
from shapely.geometry import Point, Polygon
import numpy as np


def detect_box(results, frame,n_mot,n_pre):
    results_ = results.pandas().xyxy[0].to_numpy()
    color = (251, 238, 1)
    motor = []
    person = []

    for box in results_:
        l, t, r, b = box[:4].astype('int')
        if box[6] == 'person' or box[6] == 'motorcycle' or  box[6] == 'drip' or box[6] == 'car':
            # cv2.putText(frame, str(box[4]), (l, t), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(frame, str(box[6]), (r, b), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)

        if box[6] == 'motorcycle' and str(box[4]) > str(0.90):
            motor.append(box)

        if box[6] == 'person'and str(box[4]) > str(0.90):
            person.append(box)

    return frame,motor,person


def compress_frame(frame, quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', frame, encode_param)
    if result:
        return encimg.tobytes()
    else:
        return None


def save_video( save_path, save_path_box, queue_frame , queue_frame_box , frame0, fps=20.0):

    frame_width = int(frame0.shape[0] *0.5)
    frame_height = int(frame0.shape[1]*0.5)
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (frame_height,frame_width ))
    out1 = cv2.VideoWriter(save_path_box, fourcc, fps, (frame_height,frame_width ))

        # 写入帧到输出视频
    for frame_bytes in queue_frame:
        if frame_bytes is not None:
            # print(len(frame_bytes))
            # 解码JPEG格式的字节串到图像
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                resized_frame = cv2.resize(frame, (frame_height, frame_width))
                out.write(resized_frame)
                # print('成功')
            else:
                print('失败')
        else:
            print('None')
    out.release()

    for frame_bytes in queue_frame_box:
        if frame_bytes is not None:
            # 解码JPEG格式的字节串到图像
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                resized_frame = cv2.resize(frame, (frame_height, frame_width))
                out1.write(resized_frame)
            else:
                print('失败')
        else:
            print('None')
    out1.release()
    # 释放资源

def detect_person(queue_frame, queue_frame_box, frame0):
    time_name = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    path_p = r'C:\Users\wqs\Desktop\img\person.mp4'
    path_box_p = r'C:\Users\wqs\Desktop\img\person_box.mp4'

    save_video(path_p, path_box_p, queue_frame, queue_frame_box, frame0)


def detect_motorcycle(queue_frame,queue_frame_box,frame0):
    time_name = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    path_m = fr'C:\Users\wqs\Desktop\img\motor.mp4'
    path_box_m = fr'C:\Users\wqs\Desktop\img\motor_box.mp4'

    # print('成功启动')
    save_video(path_m, path_box_m, queue_frame, queue_frame_box, frame0)



def detect_cars_stop(frame , queue_frame, queue_frame_box, frame0 ,frame_height, frame_width , track_num , stop_time , track_id):
    time_name = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    tc_path = r'C:\Users\wqs\Desktop\img\tingche.mp4'
    tc_path_box = r'C:\Users\wqs\Desktop\img\tingche_box.mp4'
    # 移除刚添加的帧
    removed_frame_queue = queue_frame_box

    removed_frame_queue.pop()

    # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    # cv2.putText(frame, 'stop', (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 停车保存视频
    resized_frame_3 = cv2.resize(frame, (frame_height, frame_width))

    frame3 = compress_frame(resized_frame_3)
    removed_frame_queue.append(frame3)

    a = len(stop_time)
    if a > 3:
        stop_time.pop()

    if track_num[0] > 260 :
        owtime = datetime.datetime.now()
        stop_time.append(owtime)
        if a < 1:
            save_video(tc_path, tc_path_box, queue_frame, removed_frame_queue, frame0)

def plot_polygons_on_frames(frame, points_lists):
    polygons = []
    for points in points_lists:
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
        polygon = Polygon(points)
        polygons.append(polygon)

    return frame, polygons

def read_jiso(frame,file_path):
    # 打开并读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 提取坐标信息
    coordinates = data.get('shapes', [])
    for index, coord in enumerate(coordinates):
        latitude = coord.get('points')
        latitude = [latitude]
        frame1 , polygon = plot_polygons_on_frames(frame, latitude)
        return frame1, polygon



