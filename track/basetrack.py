import numpy as np
from collections import OrderedDict

# 定义跟踪状态枚举类，用于表示目标的跟踪状态
class TrackState(object):
    New = 0  # 新目标
    Tracked = 1  # 正在跟踪
    Lost = 2  # 目标丢失
    Removed = 3  # 目标已移除

# 定义基础跟踪类，用于跟踪目标的状态和特征
class BaseTrack(object):
    _count = 0  # 类级变量，用于生成唯一的track_id

    track_id = 0  # 目标的唯一标识符
    is_activated = False  # 标志目标是否已激活
    state = TrackState.New  # 目标的初始状态为新目标

    history = OrderedDict()  # 用于存储目标的特征历史信息
    features = []  # 存储目标的特征集合
    curr_feature = None  # 当前帧的目标特征
    score = 0  # 目标的置信度分数
    start_frame = 0  # 目标开始跟踪的帧号
    frame_id = 0  # 当前帧的编号
    time_since_update = 0  # 自上次更新以来的时间

    # multi-camera
    location = (np.inf, np.inf)  # 目标的当前位置，初始化为无穷大表示未确定

    @property
    def end_frame(self):
        """返回目标最后出现的帧号"""
        return self.frame_id

    @staticmethod
    def next_id():
        """生成新的track_id"""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """激活目标的方法，需要子类实现"""
        raise NotImplementedError

    def predict(self):
        """目标的预测方法，需要子类实现"""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """更新目标信息的方法，需要子类实现"""
        raise NotImplementedError

    def mark_lost(self):
        """标记目标为丢失"""
        self.state = TrackState.Lost

    def mark_removed(self):
        """标记目标为已移除"""
        self.state = TrackState.Removed
