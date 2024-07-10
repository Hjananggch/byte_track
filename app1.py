from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify

app = Flask(__name__)
executor = ThreadPoolExecutor(2)  # 创建一个线程池，可以调整线程数量

def detected(data):
    executor.submit(main, data['path'],data['file'],data['part'])  # 异步调用main函数

@app.route('/jishu',methods=['POST'])#当这个路由被POST请求触发时,执行下列语句
def process_data():
    data = request.get_json()  # 从请求中获取JSON格式的数据
    detected(data)
    return jsonify({'success': '200'})  # 返回状态码200，表示接受处理但尚未完成


if '__main__' == __name__:
    app.run(host='0.0.0.0', port=8080, debug=True)#服务器将在所有网络接口上监听，允许外部设备访问,端口号为8080。调试模式.