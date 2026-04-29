# Image Detection Web App

这个仓库已经清理为只保留图片识别 Web 应用。

新页面面向手机和桌面浏览器：

- 支持手机拍照识别
- 支持从图库选择图片
- 只做图像识别
- 不做目标跟踪
- 不显示轨迹和 `track_id`

## 入口

启动文件：

`run_web.py`

## 运行前准备

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 放置模型权重，二选一

方式一：
把你的 `.pt` 权重文件放到仓库根目录的 `weights/` 下。

方式二：
设置环境变量 `MODEL_WEIGHTS` 指向权重绝对路径。

```bash
export MODEL_WEIGHTS=/path/to/your/model.pt
```

## 启动页面

```bash
python run_web.py
```

默认地址：

`http://127.0.0.1:5000`

服务默认监听 `0.0.0.0`，可以直接配合 FRP、Nginx 或局域网访问。

局域网访问时，确保手机和服务端在同一网络下，然后用服务端实际 IP 打开，例如：

`http://192.168.1.10:5000`

如果使用 FRP，把外部域名或端口转发到本机 `5000` 端口即可。

## FRP 示例

HTTP 域名方式：

```ini
[image-detection]
type = http
local_ip = 127.0.0.1
local_port = 5000
custom_domains = your-domain.example.com
```

TCP 端口方式：

```ini
[image-detection]
type = tcp
local_ip = 127.0.0.1
local_port = 5000
remote_port = 15000
```

## 页面能力

- `手机拍照`：移动端优先打开后置摄像头
- `从图库选择`：选择相册或本地图片
- `置信度阈值`：控制检测严格程度
- `IOU 阈值`：控制重叠框抑制

## 手机浏览器兼容

页面使用标准文件上传控件实现拍照和图库选择，主流 iOS Safari、Android Chrome、微信内置浏览器都可以正常选择图片。

`capture="environment"` 会优先调起后置摄像头；如果浏览器不支持该属性，会自动退化为普通图片选择。

拍照上传使用的是文件选择能力，不依赖 WebRTC 摄像头 API，因此通过 FRP 后也可以正常使用。

## 接口

- `GET /`
- `GET /api/health`
- `POST /api/detect/image`

`/api/detect/image` 表单字段：

- `image`：图片文件
- `conf`：可选，默认 `0.25`
- `iou`：可选，默认 `0.45`

## 配置

- `MODEL_WEIGHTS`：模型权重路径
- `HOST`：服务监听地址，默认 `0.0.0.0`
- `PORT`：服务端口，默认 `5000`

## 说明

当前仓库里没有自带 `.pt` 权重文件，因此页面代码已经就绪，但必须先配置权重后才能真正完成识别。
