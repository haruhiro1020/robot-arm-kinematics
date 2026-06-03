# 2軸ロボットアームの運動学を記載

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
            inverse_kinematics(): 逆運動学 (ロボットの手先位置からロボットの関節角度を算出)
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
        raise NotImplementedError("forward_kinematics() is necessary override.")

    def inverse_kinematics(self, pose):
        """
        逆運動学 (ロボットの手先位置からロボットの関節角度を算出)

        パラメータ
            pose(numpy.ndarray): ロボットの手先位置 (位置 + 姿勢) [m] + [rad]

        戻り値
            thetas(numpy.ndarray): ロボットの関節角度 [rad]
        """
        raise NotImplementedError("inverse_kinematics() is necessary override.")


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
    _DETERMINANT_THRESHOLD = 1e-4           # 行列式の閾値

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
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

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

    def inverse_kinematics(self, pose, upper=False):
        """
        逆運動学 (ロボットの手先位置からロボットの関節角度を算出)

        パラメータ
            pose(numpy.ndarray): ロボットの手先位置 (位置) [m]
            upper(bool): 腕が上向かどうか

        戻り値
            thetas(numpy.ndarray): ロボットの関節角度 [rad]
        """
        # パラメータの次元数を確認
        if np.size(pose) != self._DIMENTION_POSE:
            raise ValueError(f"parameter pose's size is abnormal. pose's size is {np.size(pose)}")

        # c2 = {(px ** 2 + py ** 2) - (l1 ** 2 + l2 ** 2)} / (2 * l1 * l2)
        px = pose[0]
        py = pose[1]
        l1 = self._links[0]
        l2 = self._links[1]
        cos2  = ((px ** 2 + py ** 2) - (l1 ** 2 + l2 ** 2)) / (2 * l1 * l2)
        # cosの範囲は-1以上1以下である
        if cos2 < -1 or cos2 > 1:
            # 異常
            raise ValueError(f"cos2 is abnormal. cos2 is {cos2}")

        # sinも求めて，theta2をatan2()より算出する
        sin2 = np.sqrt(1 - cos2 ** 2)
        theta2 = np.arctan2(sin2,  cos2)
        if not upper:
            theta2 = -theta2
            sin2 = np.sin(theta2)
            cos2 = np.cos(theta2)

        # 行列計算
        # [c1, s1] = [[l1 + l2 * c2, -l2 * s2], [l2 * s2, l1 + l2 * c2]] ** -1 * [px, py]
        element1 =  l1 + l2 * cos2
        element2 = -l2 * sin2
        matrix = np.array([[ element1, element2],
                           [-element2, element1]])
        # 行列式を計算
        det = np.linalg.det(matrix)
        # 0近傍の確認
        if det <= self._DETERMINANT_THRESHOLD and det >= -self._DETERMINANT_THRESHOLD:
            # 0近傍 (異常)
            raise ValueError(f"det is abnormal. det is {det}")

        # [c1, s1]の計算
        cos1_sin1 = np.dot(np.linalg.inv(matrix), pose)
        # theta1をatan2()より算出する
        theta1 = np.arctan2(cos1_sin1[1], cos1_sin1[0])

        thetas = np.array([theta1, theta2])

        return thetas
