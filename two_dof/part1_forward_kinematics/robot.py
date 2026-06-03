# 2自由度ロボットの運動学を記載

# ライブラリの読み込み
import numpy as np

# 自作モジュールの読み込み
from constant import *              # 定数


class Robot:
    """
    ロボットのベースクラス(抽象クラス)

    プロパティ
        _links(numpy.ndarray): ロボットのリンク長 [m]

    メソッド
        public
            forward_kinematics(): 順運動学 (ロボットの関節角度からロボットの手先位置を算出)
            links(): _linksプロパティのゲッター
    """
    # 定数の定義
    _DIMENTION_POSE  = DIMENTION_NONE       # 手先位置の次元数
    _DIMENTION_THETA = DIMENTION_NONE       # 関節角度の次元数
    _DIMENTION_LINK  = DIMENTION_NONE       # リンク数

    def __init__(self, links):
        """
        コンストラクタ

        パラメータ
            links(numpy.ndarray): ロボットのリンク長 [m]
        """
        if np.size(links) != self._DIMENTION_LINK:
            # 異常
            raise ValueError(f"links's size is abnormal. correct is {self._DIMENTION_LINK}")

        # プロパティの初期化
        self._links = links

    @property
    def links(self):
        """
        _linksプロパティのゲッター
        """
        return self._links

    def forward_kinematics(self, thetas):
        """
        順運動学 (ロボットの関節角度からロボットの手先位置を算出)

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]

        戻り値
            pose(numpy.ndarray): ロボットの手先位置 (位置 + 姿勢) [m] + [rad]
        """
        raise NotImplementedError("forward_kinematics()をオーバーライドしてください.")



class Robot2DoF(Robot):
    """
    2軸ロボットクラス

    プロパティ
        _links(numpy.ndarray): ロボットのリンク長
        _rot(Rotation): 回転行列クラス

    メソッド
        public
            forward_kinematics(): 順運動学 (ロボットの関節角度からロボットの手先位置を算出)
    """
    # 定数の定義
    _DIMENTION_POSE  = DIMENTION_2D         # 手先位置の次元数
    _DIMENTION_THETA = DIMENTION_2D         # 関節角度の次元数
    _DIMENTION_LINK  = DIMENTION_2D         # リンク数

    def __init__(self, links):
        """
        コンストラクタ

        パラメータ
            links(numpy.ndarray): ロボットのリンク長 [m]
        """
        # 親クラスの初期化
        super().__init__(links)

    def forward_kinematics(self, thetas):
        """
        順運動学 (ロボットの関節角度からロボットの手先位置を算出)

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]

        戻り値
            pose(numpy.ndarray): ロボットの手先位置 (位置) [m]
        """
        # パラメータの次元数を確認
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"thetas's size is abnormal.\nthetas's size is {np.size(thetas)}")

        # あらかじめ三角関数を算出する
        sin1  = np.sin(thetas[0])
        sin12 = np.sin(thetas[0] + thetas[1])
        cos1  = np.cos(thetas[0])
        cos12 = np.cos(thetas[0] + thetas[1])

        theta1  = np.array([cos1,  sin1])
        theta12 = np.array([cos12, sin12])

        # ロボットの手先位置を算出
        pose = self._links[0] * theta1 + self._links[1] * theta12

        return pose
