import json
import requests
pa = r"C:\Users\wqs\Desktop\byte_track-main\total.mp4"
file = r"C:\Users\wqs\Desktop\daolu\26.json"

data = {'path':pa,'detect_type':1,'activate':'True','file_json':file}

url = 'http://172.18.6.68:8080/tingche'
respson = requests.post(url, json=data)

if respson.status_code == 200:
    print('请求成功')
    print(respson.text)
else:
    print('请求失败')
    print(respson.text)
