import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from track import matching
from .basetrack import BaseTrack, TrackState

# 定义一个单目标跟踪类STrack，继承自BaseTrack
class STrack(BaseTrack):
    # 共享的卡尔曼滤波器实例，用于初始化和预测
    shared_kalman = KalmanFilter()
    
    def __init__(self, tlwh, score):
        """
        初始化一个未激活的跟踪对象。
        
        参数:
        tlwh: 目标框的左上角坐标和宽高，格式为(top left x, top left y, width, height)。
        score: 目标的置信度分数。
        """
        # 初始化目标框的位置
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        # 卡尔曼滤波器相关变量，初始化为None
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        # 是否已激活的标志
        self.is_activated = False
        # 初始化目标的置信度和跟踪序列长度
        self.score = score
        self.tracklet_len = 0

    def predict(self):
        """
        使用卡尔曼滤波器进行预测，更新mean和covariance。
        
        如果跟踪状态不是Tracked，将状态第七个元素（假设为置信度）设为0。
        """
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """
        预测多个跟踪对象的位置。
        
        参数:
        stracks: 跟踪对象的列表。
        
        更新每个跟踪对象的mean和covariance。
        """
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """
        激活一个跟踪对象，使用给定的卡尔曼滤波器进行初始化。
        
        参数:
        kalman_filter: 卡尔曼滤波器实例。
        frame_id: 当前帧编号。
        """
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        重新激活一个跟踪对象。
        
        参数:
        new_track: 新的跟踪对象信息。
        frame_id: 当前帧编号。
        new_id: 是否分配新的跟踪ID。
        """
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        更新跟踪对象的信息。
        
        参数:
        new_track: 新的跟踪对象信息。
        frame_id: 当前帧编号。
        """
        """
        Update a matched tracker
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """
        获取跟踪对象的当前位置（左上角坐标、宽度、高度）。
        
        如果mean为None，则返回初始化的位置。
        """
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """
        获取跟踪对象的当前位置（左上角、右下角）。
        
        转换格式为(min x, min y, max x, max y)。
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """
        将bounding box格式转换为中心点坐标、宽高比和高度的格式。
        
        参数:
        tlwh: (top left x, top left y, width, height)格式的bounding box。
        
        返回:
        (center x, center y, aspect ratio, height)格式的坐标。
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        """
        返回跟踪对象的字符串表示，包含跟踪ID和起止帧号。
        """
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    """
    BYTETracker类用于跟踪图像序列中的目标。
    它管理着已跟踪目标、丢失目标和已移除目标的列表，并根据检测结果更新这些列表。
    """

    def __init__(self, args, frame_rate=30):
        """
        初始化跟踪器。

        :param args: 跟踪器参数，包括跟踪阈值、跟踪缓冲区大小等。
        :param frame_rate: 视频帧率，用于计算缓冲区大小。
        """
        # 初始化跟踪目标列表
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        # 初始化帧ID和参数
        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1  # 定义检测阈值
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)  # 计算缓冲区大小
        self.max_time_lost = self.buffer_size  # 定义最大丢失时间
        self.kalman_filter = KalmanFilter()  # 初始化卡尔曼滤波器

    def update(self, output_results, img_info, img_size):
        """
        根据检测结果更新跟踪目标。

        :param output_results: 检测结果，包含目标的边界框和置信度。
        :param img_info: 图像尺寸信息。
        :param img_size: 输入图像的大小。
        :return: 更新后的已跟踪目标列表。
        """
        # 更新帧ID
        self.frame_id += 1
        # 初始化各种状态的目标列表
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 处理检测结果的格式
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        # 调整边界框的尺寸以适应图像比例
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        # 根据置信度筛选出待处理的检测结果
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        # 根据筛选结果生成检测对象列表
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        # 分离未确认的跟踪目标和已激活的跟踪目标
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # 第一次关联：高置信度的检测结果与已跟踪目标或丢失目标进行匹配
        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        #print(self.args.match_thresh)

        # 更新匹配成功的跟踪目标
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 第二次关联：低置信度的检测结果与未匹配的跟踪目标进行匹配
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 处理未匹配的跟踪目标
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 处理未确认的跟踪目标与未匹配的检测结果的关联
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # 初始化新的跟踪目标
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        # 更新丢失目标的状态，如果丢失时间过长，则将其移除
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # 更新跟踪目标列表
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    """
    合并两个跟踪目标列表。

    :param tlista: 第一个跟踪目标列表。
    :param tlistb: 第二个跟踪目标列表。
    :return: 合并后的跟踪目标列表。
    """
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """
    从列表tlistb中移除与tlista中相匹配的轨迹项，返回tlistb中不匹配的轨迹项列表。
    
    :param tlista: 轨迹项列表A
    :param tlistb: 轨迹项列表B
    :return: 不在tlista中的tlistb轨迹项列表
    """
    # 初始化一个字典，用于快速查找tlista中的轨迹项
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    # 遍历tlistb，如果轨迹项在tlista中存在，则从字典中移除
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    # 将剩余的轨迹项转换为列表并返回
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    """
    移除两个轨迹项列表中的重复轨迹项，基于IOU距离判断重复。
    
    :param stracksa: 轨迹项列表A
    :param stracksb: 轨迹项列表B
    :return: 移除重复项后的stracksa和stracksb列表
    """
    # 计算两个轨迹项列表之间的IOU距离
    pdist = matching.iou_distance(stracksa, stracksb)
    # 找出IOU距离小于0.15的轨迹对
    pairs = np.where(pdist < 0.15)
    # 初始化列表，用于存储重复的轨迹项索引
    dupa, dupb = list(), list()
    # 遍历轨迹对，根据时间顺序确定重复项，并将其索引添加到相应的列表中
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    # 根据索引列表，从原轨迹项列表中移除重复项，返回处理后的列表
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb