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
