import numpy as np
import scipy.linalg

# 定义95%置信区间的卡方分布逆函数值，用于后续计算
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

class KalmanFilter(object):
    """
    卡尔曼滤波器类实现。

    该类提供了初始化、预测、更新、多步预测等方法，用于估计动态系统的状态。
    """

    def __init__(self):
        """
        初始化卡尔曼滤波器的状态转移矩阵和更新矩阵等参数。
        """
        ndim, dt = 4, 1.
        # 状态转移矩阵，用于描述系统状态在时间步之间的转移
        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        # 更新矩阵，用于描述测量如何更新系统状态
        self._update_mat = np.eye(ndim, 2 * ndim)
        # 设置位置和速度的权重参数
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """
        初始化系统状态的均值和协方差矩阵。

        参数:
        measurement: 初始测量值。

        返回:
        mean: 状态均值向量。
        covariance: 状态协方差矩阵。
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        根据状态转移矩阵预测下一时刻的状态均值和协方差。

        参数:
        mean: 当前时刻的状态均值向量。
        covariance: 当前时刻的状态协方差矩阵。

        返回:
        mean: 预测的下一时刻状态均值向量。
        covariance: 预测的下一时刻状态协方差矩阵。
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # 使用状态转移矩阵更新状态均值和协方差
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        将状态从当前时刻投影到下一测量时刻。

        参数:
        mean: 当前时刻的状态均值向量。
        covariance: 当前时刻的状态协方差矩阵。

        返回:
        mean: 投影到下一测量时刻的状态均值向量。
        covariance: 投影到下一测量时刻的状态协方差矩阵。
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(mean, self._update_mat.T)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """
        对多个时间步进行预测。

        参数:
        mean: 当前时刻的状态均值向量。
        covariance: 当前时刻的状态协方差矩阵。

        返回:
        mean: 预测的多个时间步的状态均值向量。
        covariance: 预测的多个时间步的状态协方差矩阵。
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        使用测量值更新状态估计。

        参数:
        mean: 当前时刻的状态均值向量。
        covariance: 当前时刻的状态协方差矩阵。
        measurement: 当前时刻的测量值。

        返回:
        mean: 更新后的状态均值向量。
        covariance: 更新后的状态协方差矩阵。
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # 使用Cholesky分解计算卡尔曼增益
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """
        计算测量值与预测值之间的门限距离。

        参数:
        mean: 当前时刻的状态均值向量。
        covariance: 当前时刻的状态协方差矩阵。
        measurements: 测量值集合。
        only_position: 是否仅考虑位置信息，默认为False。
        metric: 距离度量方法，默认为'maha'，可选'gaussian'。

        返回:
        门限距离。
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')