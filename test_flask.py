import json
import requests
pa = r"D:\ANACONDA/envs\\car-detect/car_detect.mp4"
file = r"C:\\Users\A\Desktop\\0.json"

data = {'path':pa,'file_json':file}

url = 'http://172.18.6.227:8080/jishu'
respson = requests.post(url, json=data)

if respson.status_code == 200:
    print('请求成功')
    print(respson.text)
else:
    print('请求失败')
    print(respson.text)