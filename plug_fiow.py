import cv2
import subprocess
import time
import torch

# 加载模型
model = torch.hub.load(r'D:/byte_track-main/model', 'custom'
                       ,path=r'D:\byte_track-main\car_best.pt',source='local')


def get_ffmpeg_command(rtsp_url):
    return [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-f', 'rawvideo',  # 输入格式为原始视频
        '-vcodec', 'rawvideo',  # 原始视频编解码器
        '-pix_fmt', 'bgr24',  # 像素格式
        '-s', '640x480',  # 图片分辨率，根据实际需要调整
        '-r', '25',  # 视频帧率
        '-i', '-',  # 从标准输入读取
        '-c:v', 'libx264',  # 视频编码器
        '-pix_fmt', 'yuv420p',  # 像素格式
        '-preset', 'ultrafast',  # 预设
        '-tune', 'zerolatency',  # 调整为零延迟
        '-f', 'rtsp',  # 输出格式
        rtsp_url  # RTSP URL
    ]


def start_ffmpeg_process(rtsp_url):
    command = get_ffmpeg_command(rtsp_url)
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def stream_video_with_detection(input_file, rtsp_url):
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_file}")
        return

    process = start_ffmpeg_process(rtsp_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # 检测处理
        results = model(frame)

        # 处理检测结果
        results_np = results.xyxy[0].cpu().numpy()# 获取检测结果中的边界框坐标信息
        for bbox in results_np:
            xmin, ymin, xmax, ymax, conf, cls = bbox
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

        # 调整图像大小
        processed_frame = cv2.resize(frame, (640, 480))

        # 显示处理后的视频帧
        cv2.imshow('Processed Stream', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 将处理后的帧写入ffmpeg的标准输入
        process.stdin.write(processed_frame.tobytes())

        # 模拟帧率（每秒25帧）
        time.sleep(1 / 25.0)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    process.stdin.close()
    process.wait()


input_file = r'rtmp://203.34.56.118:1938/live/66b10785-f541-4aeb-bc9f-144eaee31f6c'
rtsp_url = 'rtsp://127.0.0.1:8554/stream'

# 调用函数进行推流
stream_video_with_detection(input_file, rtsp_url)
