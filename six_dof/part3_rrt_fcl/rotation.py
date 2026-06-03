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
        """
        x軸方向にtheta[rad]回転させる回転行列

        パラメータ
            theta(float): 回転角度 [rad]

        戻り値
            rotation(numpy.ndarray): 回転行列
        """
        rotation = np.array([[1, 0,              0            ],
                             [0, np.cos(theta), -np.sin(theta)],
                             [0, np.sin(theta),  np.cos(theta)]])

        return rotation

    def _rot_y(self, theta):
        """
        y軸方向にtheta[rad]回転させる回転行列

        パラメータ
            theta(float): 回転角度 [rad]

        戻り値
            rotation(numpy.ndarray): 回転行列
        """
        rotation = np.array([[ np.cos(theta), 0, np.sin(theta)],
                             [ 0,             1, 0            ],
                             [-np.sin(theta), 0, np.cos(theta)]])

        return rotation

    def _rot_z(self, theta):
        """
        z軸方向にtheta[rad]回転させる回転行列

        パラメータ
            theta(float): 回転角度 [rad]

        戻り値
            rotation(numpy.ndarray): 回転行列
        """
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
            euler(numpy.ndarray): Z(theta1)-Y(theta2)-X(theta3)オイラー角 [rad] の順番にデータを保存
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

    def rot_to_zyz_euler(self, rotation, y_plus=True):
        """
        回転行列からZ(theta1)-Y(theta2)-Z(theta3)オイラー角を取得

        パラメータ
            rotation(numpy.ndarray): 回転行列
            y_plus(bool): Y(theta2)を+にするか

        戻り値
            euler(numpy.ndarray): Z(theta1)-Y(theta2)-Z(theta3) [rad] の順番にデータを保存
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
