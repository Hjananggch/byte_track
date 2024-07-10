import json
import requests
pa = r"rtmp://182.43.225.166:1935/live/175166d8-ace8-4ec6-81aa-416771fgy004"
file = [234,567,789,890]
part='in'
data = {'path':pa,'file':file,'part':part}

url = 'http://172.18.6.236:8080/jishu'
respson = requests.post(url, json=data)#请求体将以JSON格式发送之前准备的数据。

if respson.status_code == 200:
    print('请求成功')
    print(respson.text)
else:
    print('请求失败')
    print(respson.text)
# 向指定的URL（http://172.18.6.68:8080/tingche）发送一个POST请求，
# 请求体包含视频文件路径、检测类型、激活状态以及一个JSON文件路径的参数，然后根据响应状态码判断请求是否成功，并打印响应内容