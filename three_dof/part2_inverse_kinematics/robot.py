# ロボットアームの運動学を記載

# ライブラリの読み込み
import numpy as np

# サードパーティーの読み込み


# 自作モジュールの読み込み
from constant import *              # 定数
from rotation import Rotation       # 回転行列


class Robot:
    """
    ロボットのベースクラス(抽象クラス)

    プロパティ
        _links(numpy.ndarray): ロボットのリンク長 [m]
        _rot(Rotation): 回転行列クラス
        _objects(list): 干渉物オブジェクト
        _manager(fcl.DynamicAABBTreeCollisionManager): 干渉判定クラス

    メソッド
        public
            forward_kinematics(): 順運動学 (ロボットの関節角度からロボットの手先位置を算出)
            inverse_kinematics(): 逆運動学 (ロボットの手先位置からロボットの関節角度を算出)
            forward_kinematics_all_pos(): 順運動学で全リンクの位置を取得
            update(): 角度を与えて，各リンクの直方体を更新する
            links(): _linksプロパティのゲッター
            manager(): _managerプロパティのゲッター
    """
    # 定数の定義
    _DIMENTION_POSE  = DIMENTION_NONE       # 手先位置の次元数
    _DIMENTION_THETA = DIMENTION_NONE       # 関節角度の次元数
    _DIMENTION_LINK  = DIMENTION_NONE       # リンク数
    _DIMENTION_AXIS  = DIMENTION_NONE       # 回転軸数

    _INITIAL_THETA   = 0.0                  # 初期回転角度 [rad]


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
        self._rot = Rotation()
        self._objects = []
        self._manager = None

    @property
    def links(self):
        """
        _linksプロパティのゲッター
        """
        return self._links

    @property
    def manager(self):
        """
        _managerプロパティのゲッター
        """
        return self._manager

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

    def forward_kinematics_all_pos(self, thetas):
        """
        順運動学で全リンクの位置を取得

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]

        戻り値
            all_pose(numpy.ndarray): ロボットの全リンク位置 (位置 + 姿勢) [m] + [rad]
        """
        raise NotImplementedError("forward_kinematics() is necessary override.")

    def update(self, thetas):
        """
        角度を与えて，各リンクの直方体を更新する

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]
        """
        raise NotImplementedError("update() is necessary override.")


class Robot3DoF(Robot):
    """
    3軸ロボットクラス

    プロパティ
        _links(numpy.ndarray): ロボットのリンク長
        _rot(Rotation): 回転行列クラス
        _axiss(list): 関節の回転軸

    メソッド
        public
            forward_kinematics(): 順運動学 (ロボットの関節角度からロボットの手先位置を算出)
            inverse_kinematics(): 逆運動学 (ロボットの手先位置からロボットの関節角度を算出)
            forward_kinematics_all_pos(): 順運動学で全リンクの位置を取得
            update(): 角度を与えて，各リンクの直方体を更新する
    """
    # 定数の定義
    _DIMENTION_POSE  = DIMENTION_3D         # 手先位置の次元数
    _DIMENTION_THETA = DIMENTION_3D         # 関節角度の次元数
    _DIMENTION_LINK  = DIMENTION_3D         # リンク数

    _DETERMINANT_THRESHOLD = 1e-4           # 行列式の閾値
    _THETA1_XY_THRESHOLD   = 1e-4           # theta1算出時のx, y閾値
    _BOX_WIDTH = 1e-2                       # 各リンクの幅を定義


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
        sin2  = np.sin(thetas[1])
        sin23 = np.sin(thetas[1] + thetas[2])
        cos1  = np.cos(thetas[0])
        cos2  = np.cos(thetas[1])
        cos23 = np.cos(thetas[1] + thetas[2])

        # ロボットの手先位置を算出
        pxy_common = self._links[1] * cos2 + self._links[2] * cos23
        px = cos1 * pxy_common
        py = sin1 * pxy_common
        pz = self._links[0] + self._links[1] * sin2 + self._links[2] * sin23
        pose = np.array([px, py, pz])

        return pose

    def inverse_kinematics(self, pose, upper=False, front=True):
        """
        逆運動学 (ロボットの手先位置からロボットの関節角度を算出)

        パラメータ
            pose(numpy.ndarray): ロボットの手先位置 (位置) [m]
            upper(bool): 腕が上向かどうか
            front(bool): 正面かどうか

        戻り値
            thetas(numpy.ndarray): ロボットの関節角度 [rad]
        """
        # パラメータの次元数を確認
        if np.size(pose) != self._DIMENTION_POSE:
            raise ValueError(f"parameter pose's size is abnormal. pose's size is {np.size(pose)}")

        # はじめにtheta3を算出する
        # c3 = {(px ** 2 + py ** 2 + (pz - l1) ** 2) - (l2 ** 2 + l3 ** 2)} / (2 * l2 * l3)
        px = pose[0]
        py = pose[1]
        pz = pose[2]
        l1 = self._links[0]
        l2 = self._links[1]
        l3 = self._links[2]
        cos3  = ((px ** 2 + py ** 2 + (pz - l1) ** 2) - (l2 ** 2 + l3 ** 2)) / (2 * l2 * l3)
        # cosの範囲は-1以上1以下である
        if abs(cos3) > 1:
            # 異常
            raise ValueError(f"cos3 is abnormal. cos3 is {cos3}")

        # sinも求めて，theta3をatan2(sin2, cos2)より算出する
        sin3 = np.sqrt(1 - cos3 ** 2)
        theta3 = np.arctan2(sin3,  cos3)
        if not upper:
            # 下向きの角度のため，三角関数も更新
            theta3 = -theta3
            sin3 = np.sin(theta3)
            cos3 = np.cos(theta3)

        # 次にtheta1を算出する
        # theta1 = atan2(py, px)
        # py, pxが0近傍なら失敗
        if abs(py) <= self._THETA1_XY_THRESHOLD and abs(px) <= self._THETA1_XY_THRESHOLD:
            # 近傍のため，エラー
            raise ValueError(f"abs(py) and abs(px) is abnormal. py is {py}, px is {px}")
        if front:
            # 位置(pose)に対して正面向き
            theta1 = np.arctan2( py,  px)
        else:
            # 位置(pose)に対して後ろ向き
            theta1 = np.arctan2(-py, -px)

        # 最後にtheta2を算出する
        # 行列計算
        # [c2, s2] = [[l2 + l3 * c3, -l3 * s3], [l3 * s3, l2 + l3 * c3]] ** -1 * [root(px ** 2 + py ** 2), pz - l1]
        element1 =  l2 + l3 * cos3
        element2 = -l3 * sin3
        matrix = np.array([[ element1, element2],
                           [-element2, element1]])
        # 行列式を計算
        det = np.linalg.det(matrix)
        # 0近傍の確認
        if abs(det) <= self._DETERMINANT_THRESHOLD:
            # 0近傍 (異常)
            raise ValueError(f"det is abnormal. det is {det}")

        # 位置を保存 ([root(px ** 2 + py ** 2), pz - l1])
        position = np.array([np.sqrt(px ** 2 + py ** 2), pz - l1])

        # [c2, s2]の計算
        cos2_sin2 = np.dot(np.linalg.inv(matrix), position)
        # theta2をatan2()より算出する
        theta2 = np.arctan2(cos2_sin2[1], cos2_sin2[0])

        thetas = np.array([theta1, theta2, theta3])

        return thetas

    def forward_kinematics_all_link_pos(self, thetas):
        """
        順運動学で全リンクの位置を取得 (グラフの描画で使用する)

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]

        戻り値
            all_link_pose(numpy.ndarray): ロボットの全リンク位置 (位置 + 姿勢) [m] + [rad]
        """
        # パラメータの次元数を確認
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

        # 回転角度とリンク長をローカル変数に保存
        theta1 = thetas[0]
        theta2 = thetas[1]
        theta3 = thetas[2]
        link1  = self._links[0]
        link2  = self._links[1]
        link3  = self._links[2]

        # 三角関数を計算
        s1  = np.sin(theta1)
        c1  = np.cos(theta1)
        s2  = np.sin(theta2)
        c2  = np.cos(theta2)
        s3  = np.sin(theta3)
        c3  = np.cos(theta3)
        s12 = np.sin(theta1 + theta2)
        c12 = np.cos(theta1 + theta2)
        s23 = np.sin(theta2 + theta3)
        c23 = np.cos(theta2 + theta3)

        # 各リンクの位置を算出
        base_pos  = np.zeros(self._DIMENTION_POSE)
        link1_pos = np.array([0, 0, link1])
        link2_xy_common = link2 * c2
        link2_pos = np.array([c1 * link2_xy_common, s1 * link2_xy_common, link1 + link2 * s2])
        link3_xy_common = link2_xy_common + link3 * c23
        link3_pos = np.array([c1 * link3_xy_common, s1 * link3_xy_common, link1 + link2 * s2 + link3 * s23])

        # 全リンクの位置を算出
        all_link_pose = np.array([base_pos, link1_pos, link2_pos, link3_pos])

        return all_link_pose
