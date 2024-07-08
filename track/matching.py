# 导入计算机视觉、numpy、scipy等库，用于图像处理和数学计算
import cv2
import numpy as np
import scipy
import lap
# 导入距离计算函数
from scipy.spatial.distance import cdist
# 导入自定义的bounding box重叠计算函数和卡尔曼滤波器
from cython_bbox import bbox_overlaps as bbox_ious
from track import kalman_filter
# 导入时间模块，用于计时
import time

# 合并匹配结果，根据给定的形状信息和两个匹配集，计算合并后的匹配情况
def merge_matches(m1, m2, shape):
    """
    合并两个匹配集。

    参数:
    m1: 第一个匹配集
    m2: 第二个匹配集
    shape: 填充矩阵的形状

    返回:
    match: 合并后的匹配列表
    unmatched_O: 第一个匹配集中未匹配的元素
    unmatched_Q: 第二个匹配集中未匹配的元素
    """
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q

# 根据索引和成本矩阵，转换为匹配结果
def _indices_to_matches(cost_matrix, indices, thresh):
    """
    将索引转换为匹配，并根据成本矩阵和阈值确定匹配与否。

    参数:
    cost_matrix: 成本矩阵
    indices: 索引对
    thresh: 匹配阈值

    返回:
    matches: 匹配的索引对
    unmatched_a: 未匹配的索引
    unmatched_b: 未匹配的索引
    """
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b

# 使用线性分配算法，对成本矩阵进行匹配
def linear_assignment(cost_matrix, thresh):
    """
    使用线性分配算法找出成本矩阵中的最佳匹配。

    参数:
    cost_matrix: 成本矩阵
    thresh: 匹配阈值

    返回:
    matches: 最佳匹配的索引对
    unmatched_a: 未匹配的索引
    unmatched_b: 未匹配的索引
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b

# 计算两个bounding box集合的IOU矩阵
def ious(atlbrs, btlbrs):
    """
    计算两个bounding box集合的IOU矩阵。

    参数:
    atlbrs: 第一个bounding box集合
    btlbrs: 第二个bounding box集合

    返回:
    ious: IOU矩阵
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

# 计算两个track集合的IOU距离矩阵
def iou_distance(atracks, btracks):
    """
    计算两个track集合的IOU距离矩阵。

    参数:
    atracks: 第一个track集合
    btracks: 第二个track集合

    返回:
    cost_matrix: IOU距离矩阵
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

# 计算两个track集合的垂直IOU距离矩阵
def v_iou_distance(atracks, btracks):
    """
    计算两个track集合的垂直IOU距离矩阵。

    参数:
    atracks: 第一个track集合
    btracks: 第二个track集合

    返回:
    cost_matrix: 垂直IOU距离矩阵
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

# 计算track和detection的嵌入距离矩阵
def embedding_distance(tracks, detections, metric='cosine'):
    """
    计算track和detection的嵌入距离矩阵。

    参数:
    tracks: track集合
    detections: detection集合
    metric: 距离度量方法，默认为'cosine'（余弦距离）

    返回:
    cost_matrix: 嵌入距离矩阵
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features

    return cost_matrix

# 根据卡尔曼滤波器的结果，计算cost矩阵的门控距离
def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """
    根据卡尔曼滤波器的结果，对cost矩阵进行门控，过滤掉与检测结果距离过远的track。

    参数:
    kf: 卡尔曼滤波器
    cost_matrix: 初始成本矩阵
    tracks: track集合
    detections: detection集合
    only_position: 是否只考虑位置信息，默认为False（同时考虑位置和方向）

    返回:
    cost_matrix: 门控后的成本矩阵
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf

    return cost_matrix

# 根据卡尔曼滤波器的结果，融合运动信息到cost矩阵中
def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """
    根据卡尔曼滤波器的结果，融合运动信息到cost矩阵中。

    参数:
    kf: 卡尔曼滤波器
    cost_matrix: 初始成本矩阵
    tracks: track集合
    detections: detection集合
    only_position: 是否只考虑位置信息，默认为False（同时考虑位置和方向）
    lambda_: 权重参数，默认为0.98

    返回:
    cost_matrix: 融合运动信息后的成本矩阵
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row]

def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """
    Fuses cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments.
        detections (list[BaseTrack]): List of detections with scores.

    Returns:
        (np.ndarray): Fused similarity matrix.
    """

    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost