# 回転行列の定義

# 標準ライブラリの読み込み
import numpy as np


# サードパーティーの読み込み


# 自作モジュールの読み込み
from constant import *      # 定数



class MyRotation:
    """
    回転行列クラス
    """
    _PITCH_THRESHOLD = 1e-4             # ピッチ角の閾値
    _ZERO_NEAR = 1e-4                   # 0近傍の閾値
    _EPSILON   = 1e-5                   # 微小値

    _ROT_MAX_VALUE =  1.0           # 回転行列の最大値
    _ROT_MIN_VALUE = -1.0           # 回転行列の最小値


    def _rot_x(self, theta):
        rotation = np.array([[1, 0,              0            ],
                             [0, np.cos(theta), -np.sin(theta)],
                             [0, np.sin(theta),  np.cos(theta)]])
        return rotation

    def _rot_y(self, theta):
        rotation = np.array([[ np.cos(theta), 0, np.sin(theta)],
                             [ 0,             1, 0            ],
                             [-np.sin(theta), 0, np.cos(theta)]])
        return rotation

    def _rot_z(self, theta):
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                             [np.sin(theta),  np.cos(theta), 0],
                             [0,              0,             1]])
        return rotation

    def rot(self, theta, axis):
        """
        回転軸に応じた回転行列の取得

        パラメータ
            theta(float): 回転角度 [rad]
            axis(str): 回転軸

        戻り値
            rotation(numpy.ndarray): 回転行列
        """
        if axis == ROTATION_X_AXIS:
            rotation = self._rot_x(theta)
        elif axis == ROTATION_Y_AXIS:
            rotation = self._rot_y(theta)
        elif axis == ROTATION_Z_AXIS:
            rotation = self._rot_z(theta)
        elif axis == ROTATION_X_NEGATIVE_AXIS:
            rotation = self._rot_x(-theta)
        elif axis == ROTATION_Y_NEGATIVE_AXIS:
            rotation = self._rot_y(-theta)
        elif axis == ROTATION_Z_NEGATIVE_AXIS:
            rotation = self._rot_z(-theta)
        else:
            raise ValueError(f"axis is abnormal. now is {axis}")

        return rotation

    def rot_from_zyx_euler(self, euler):
        """
        Z(theta1)-Y(theta2)-X(theta3)オイラー角による回転行列の計算

        パラメータ
            euler(float): Z(theta1)-Y(theta2)-X(theta3)オイラー角 [rad]

        戻り値
            rotation(numpy.ndarray): 回転行列
        """
        theta1 = euler[0]
        theta2 = euler[1]
        theta3 = euler[2]

        rot_z  = self._rot_z(theta1)
        rot_y  = self._rot_y(theta2)
        rot_x  = self._rot_x(theta3)

        rotation = np.dot(rot_z, rot_y)
        rotation = np.dot(rotation, rot_x)

        return rotation

    def rot_to_zyx_euler(self, rotation, y_cos_plus=True):
        """
        回転行列からZ(theta1)-Y(theta2)-X(theta3)オイラー角を取得

        パラメータ
            rotation(numpy.ndarray): 回転行列
            y_cos_plus(bool): Y(theta2)を+にするか

        戻り値
            euler(numpy.ndarray): Z(theta1)-Y(theta2)-X(theta3)オイラー角 [rad]
        """
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation's shape is abnormal. rotaton'shape is {rotation.shape}")

        r11, r12, r13 = rotation[0, :]
        r21, r22, r23 = rotation[1, :]
        r31, r32, r33 = rotation[2, :]

        theta2_sin = -r31
        if y_cos_plus:
            theta2_cos =  np.sqrt(r32 ** 2 + r33 ** 2)
        else:
            theta2_cos = -np.sqrt(r32 ** 2 + r33 ** 2)
        theta2 = np.arctan2(theta2_sin, theta2_cos)

        if abs(theta2 - np.pi / 2) <= self._PITCH_THRESHOLD:
            theta1 = 0.0
            theta3 = -np.arctan2(r23, r13)
        elif abs(theta2 + np.pi / 2) <= self._PITCH_THRESHOLD:
            theta1 = 0.0
            theta3 = np.arctan2(-r23, -r13)
        else:
            theta1 = np.arctan2(r21, r11)
            theta3 = np.arctan2(r32, r33)

        rpy = np.array([theta1, theta2, theta3])

        return rpy

    def rot_from_zyz_euler(self, euler):
        """
        Z(theta1)-Y(theta2)-Z(theta3)オイラー角による回転行列の計算

        パラメータ
            euler(float): Z(theta1)-Y(theta2)-Z(theta3)オイラー角 [rad]

        戻り値
            rotation(numpy.ndarray): 回転行列
        """
        theta1 = euler[0]
        theta2 = euler[1]
        theta3 = euler[2]

        rot_z1 = self._rot_z(theta1)
        rot_y  = self._rot_y(theta2)
        rot_z2 = self._rot_z(theta3)

        rotation = np.dot(rot_z1, rot_y)
        rotation = np.dot(rotation, rot_z2)

        return rotation

    def rot_to_zyz_euler(self, rotation, y_plus=True):
        """
        回転行列からZ(theta1)-Y(theta2)-Z(theta3)オイラー角を取得

        パラメータ
            rotation(numpy.ndarray): 回転行列
            y_plus(bool): Y(theta2)を+にするか

        戻り値
            euler(numpy.ndarray): Z(theta1)-Y(theta2)-Z(theta3) [rad]
        """
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation's shape is abnormal. rotaton'shape is {rotation.shape}")

        r11, r12, r13 = rotation[0, :]
        r21, r22, r23 = rotation[1, :]
        r31, r32, r33 = rotation[2, :]

        y_cos = r33
        if y_plus:
            y_sin =  np.sqrt(r31 ** 2 + r32 ** 2)
        else:
            y_sin = -np.sqrt(r31 ** 2 + r32 ** 2)
        theta2 = np.arctan2(y_sin, y_cos)

        if abs(r33 - 1) <= self._ZERO_NEAR:
            theta1 = 0.0
            theta3 = np.arctan2(r21, r11)
        elif abs(r33 + 1) <= self._ZERO_NEAR:
            theta1 = 0.0
            theta3 = -np.arctan2(-r21, -r11)
        else:
            theta3 = np.arctan2(r32 / y_sin, -r31 / y_sin)
            theta1 = np.arctan2(r23 / y_sin, r13 / y_sin)

        euler = np.array([theta1, theta2, theta3])

        return euler

    def difference_rot_matrix(self, rot1, rot2):
        """
        回転行列(rot1から見たrot2)の差分を計算

        パラメータ
            rot1(numpy.ndarray): 回転行列1 (3 * 3行列)
            rot2(numpy.ndarray): 回転行列2 (3 * 3行列)

        戻り値
            diff_rot(numpy.ndarray): 回転行列の差分
        """
        if rot1.shape != (3, 3):
            raise ValueError(f"rot1.shape is not (3, 3). rot1.shape is {rot1.shape}")

        if rot2.shape != (3, 3):
            raise ValueError(f"rot2.shape is not (3, 3). rot2.shape is {rot2.shape}")

        diff_rot = np.dot(rot1.T, rot2)

        return diff_rot

    def get_single_axis(self, init_rot, target_rot, t):
        """
        一軸回転法 (single axis rotation method) による回転角度と回転軸を算出

        パラメータ
            init_rot(numpy.ndarray): 初期姿勢
            target_rot(numpy.ndarray): 目標姿勢
            t(float): 目標姿勢への割合 (0 ~ 1)

        戻り値
            sigle_axis(numpy.ndarray): 一軸回転(回転角度(スカラー)，回転軸(3ベクトル))
        """
        single_axis = np.zeros(4)

        if t < 0 or t > 1:
            raise ValueError(f"t is abnormal. t's range is 0 to 1. t is {t}")

        diff_rot = np.dot(init_rot.T, target_rot)
        cos_rot_angle = np.clip((np.trace(diff_rot) - 1) / 2, self._ROT_MIN_VALUE, self._ROT_MAX_VALUE)

        u = np.array([diff_rot[2, 1] - diff_rot[1, 2], diff_rot[0, 2] - diff_rot[2, 0], diff_rot[1, 0] - diff_rot[0, 1]])
        sin_rot_angle = np.linalg.norm(u) / 2

        rot_angle = np.arctan2(sin_rot_angle, cos_rot_angle)

        if np.isclose(rot_angle, 0.0, atol=1e-3) or np.isclose(abs(rot_angle), np.pi, atol=1e-3):
            rot_angle = 0
            rot_axis = np.array([1, 0, 0])
        else:
            rot_axis = u / (2 * np.sin(rot_angle) + self._EPSILON)
            rot_axis = rot_axis / (np.linalg.norm(rot_axis) + self._EPSILON)

        single_axis[0]  = rot_angle * t
        single_axis[1:] = rot_axis

        return single_axis

    def _rot_from_single_axis(self, single_axis):
        """
        一軸回転法から回転行列を算出

        パラメータ
            single_axis(numpy.ndarray): 一軸回転法で算出した一軸(回転角度(スカラー)と回転軸(3ベクトル)

        戻り値
            rotation(numpy.ndarray): 回転行列
        """
        theta = single_axis[0]
        lambda_x = single_axis[1]
        lambda_y = single_axis[2]
        lambda_z = single_axis[3]

        sin = np.sin(theta)
        cos = np.cos(theta)
        one_minus_cos = 1 - cos

        r11 = one_minus_cos * lambda_x ** 2 + cos
        r12 = one_minus_cos * lambda_x * lambda_y - lambda_z * sin
        r13 = one_minus_cos * lambda_z * lambda_x + lambda_y * sin
        r21 = one_minus_cos * lambda_x * lambda_y + lambda_z * sin
        r22 = one_minus_cos * lambda_y ** 2 + cos
        r23 = one_minus_cos * lambda_y * lambda_z - lambda_x * sin
        r31 = one_minus_cos * lambda_z * lambda_x - lambda_y * sin
        r32 = one_minus_cos * lambda_y * lambda_z + lambda_x * sin
        r33 = one_minus_cos * lambda_z ** 2 + cos

        rotation = np.array([[r11, r12, r13],
                             [r21, r22, r23],
                             [r31, r32, r33]])

        return rotation

    def intermediate_rot(self, init_rot, target_rot, t):
        """
        初期姿勢と目標姿勢の中間姿勢を計算

        パラメータ
            init_rot(numpy.ndarray): 初期姿勢
            target_rot(numpy.ndarray): 目標姿勢
            t(float): 目標姿勢への割合 (0 ~ 1)

        戻り値
            rotation(numpy.ndarray): 中間姿勢
        """
        if init_rot.shape != (3, 3):
            raise ValueError(f"init_rot' shape is abnormal. init_rot.shape is {init_rot.shape}")

        if target_rot.shape != (3, 3):
            raise ValueError(f"target_rot' shape is abnormal. target_rot.shape is {target_rot.shape}")

        if t < 0 or t > 1:
            raise ValueError(f"t is abnormal. t's range is 0 to 1. t is {t}")

        single_axis = self.get_single_axis(init_rot, target_rot, t)

        sub_rotation = self._rot_from_single_axis(single_axis)

        rotation = np.dot(init_rot, sub_rotation)

        return rotation
