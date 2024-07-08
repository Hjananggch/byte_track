import cv2
import datetime
import json
from shapely.geometry import Point, Polygon
import numpy as np

def detect_box(results, frame):

    car = []

    results_ = results.pandas().xyxy[0].to_numpy()
    color = (255, 20, 0)
    for box in results_: 
        l, t, r, b = box[:4].astype('int')
        cv2.rectangle(frame, (l, t), (r, b), color, 1)

    return frame
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

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)


    coordinates = data.get('shapes', [])
    for index, coord in enumerate(coordinates):
        latitude = coord.get('points')
        latitude = [latitude]
        frame1 , polygon = plot_polygons_on_frames(frame, latitude)
        return frame1, polygon

def save_video(save_path, save_path_box, qf , qfd , frame0, fps=20.0):
    frame_width = int(frame0.shape[0] *0.5)
    frame_height = int(frame0.shape[1]*0.5)


    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (frame_height,frame_width ))
    out1 = cv2.VideoWriter(save_path_box, fourcc, fps, (frame_height,frame_width ))


    for frame_bytes in qf:
        if frame_bytes is not None:
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                resized_frame = cv2.resize(frame, (frame_height, frame_width))
                out.write(resized_frame)
            else:
                print('失败')
        else:
            print('None')
    out.release()
    for frame_bytes in qfd:
        if frame_bytes is not None:
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                resized_frame = cv2.resize(frame, (frame_height, frame_width))
                out1.write(resized_frame)
            else:
                print('失败')
        else:
            print('None')
    out1.release()


def region_detect(frame,s_et,track_id,id_data,x_center,y_center):
    num = 0
    num_1 = 0
    cv2.rectangle(frame, (s_et[0], s_et[1]), (s_et[2], s_et[3]), (0, 255, 0), 3)
    if track_id not in id_data[1] and s_et[0]<x_center<s_et[2] and s_et[1]<y_center<s_et[3]:
        num = 1
        num_1 = 1
        id_data[1].append(track_id)
        if len(id_data[1]) > 4:
            id_data[1].pop(0)
    return (num,num_1)

def compress_frame(frame, quality=90):

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    result, encimg = cv2.imencode('.jpg', frame, encode_param)
    
    if result:
        return encimg.tobytes()
    else:
        return None
    
def save_car(qf, qfd, frame0):
    time_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    tc_path = f'D:/photo/video/car_num/car_num_{time_name}.mp4'
    tc_path_box = f'D:/photo/video/car_num/car_num_box_{time_name}.mp4'
    save_video(tc_path, tc_path_box, qf,qfd, frame0)