# ロボットアームの運動学を記載

# ライブラリの読み込み
import numpy as np

# サードパーティーの読み込み
import fcl

# 自作モジュールの読み込み
from constant import *              # 定数
from rotation import MyRotation       # 回転行列


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

        protected
            _calc_homogeneous_matrix(): 同時変換行列の計算
    """
    # 定数の定義
    _DIMENTION_POSE  = DIMENTION_NONE       # 手先位置の次元数
    _DIMENTION_THETA = DIMENTION_NONE       # 関節角度の次元数
    _DIMENTION_LINK  = DIMENTION_NONE       # リンク数
    _DIMENTION_AXIS  = DIMENTION_NONE       # 回転軸数

    _INITIAL_THETA   = 0.0                  # 初期回転角度 [rad]
    _HOMOGENEOU_MAT_ELEMENT = 4             # 同時変換行列の次元数
    _ROTATION_MAT_ELEMENT   = 3             # 回転行列の次元数


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
        self._rot = MyRotation()
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

    def _calc_homogeneous_matrix(self, thetas):
        """
        同時変換行列の計算

        パラメータ
            thetas(numpy.ndarray): 関節角度 [rad]

        戻り値
            homogeneou_matix(numpy.ndarray): 全リンクの同時変換行列
        """
        # パラメータの次元数を確認
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

        homogeneou_matix = np.zeros((self._DIMENTION_LINK, self._HOMOGENEOU_MAT_ELEMENT, self._HOMOGENEOU_MAT_ELEMENT))

        # 1リンク前の同時変換行列
        prev_homogeneou_matrix = np.eye(self._HOMOGENEOU_MAT_ELEMENT)
        for i in range(self._DIMENTION_THETA):
            # 4行4列の要素を1に更新
            homogeneou_matix[i, -1, -1] = 1
            # 回転行列の計算
            rotation_matrix   = self._rot.rot(thetas[i], self._axiss[i])
            # リンク間の相対位置を取得
            relative_position = self._relative_positions[i].reshape(1, -1)
            # 同時変換行列に回転行列を保存
            homogeneou_matix[i, :self._ROTATION_MAT_ELEMENT, :self._ROTATION_MAT_ELEMENT] = rotation_matrix
            # 同時変換行列に相対位置を保存
            homogeneou_matix[i, :self._ROTATION_MAT_ELEMENT,  self._ROTATION_MAT_ELEMENT] = relative_position
            # 1リンク前の同時変換行列と組み合わせる
            homogeneou_matix[i] = np.dot(prev_homogeneou_matrix, homogeneou_matix[i])
            # 1リンク前の同時変換行列の更新
            prev_homogeneou_matrix = homogeneou_matix[i]

        return homogeneou_matix


class Robot6DoF(Robot):
    """
    6軸ロボットクラス

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
    _DIMENTION_POSE  = DIMENTION_6D         # 手先位置・姿勢の次元数
    _DIMENTION_THETA = DIMENTION_6D         # 関節角度の次元数
    _DIMENTION_LINK  = DIMENTION_6D         # リンク数

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

        # (全リンクの角度を0とした時の) 各リンク間の相対位置
        self._relative_positions = np.zeros((self._DIMENTION_LINK + 1, 3))
        self._relative_positions[0] = np.array([0, 0, 0])
        self._relative_positions[1] = np.array([0, 0, self._links[0]])
        self._relative_positions[2] = np.array([0, 0, self._links[1]])
        self._relative_positions[3] = np.array([0, 0, self._links[2]])
        self._relative_positions[4] = np.array([0, 0, self._links[3]])
        self._relative_positions[5] = np.array([0, 0, self._links[4]])
        self._relative_positions[6] = np.array([0, 0, self._links[5]])

        # リンク1，リンク4，リンク6は回転軸がz軸，リンク2，リンク3，リンク5は回転軸がy軸である
        self._axiss = [ROTATION_Z_AXIS, ROTATION_Y_AXIS, ROTATION_Y_AXIS, ROTATION_Z_AXIS, ROTATION_Y_AXIS, ROTATION_Z_AXIS]

    def forward_kinematics(self, thetas):
        """
        順運動学 (ロボットの関節角度からロボットの手先位置を算出)

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]

        戻り値
            pose(numpy.ndarray): ロボットの手先位置・姿勢(RPY) [m] / [rad]
        """
        # パラメータの次元数を確認
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

        # 同時変換行列(6 * 4 * 4)を計算する
        homogeneou_matrix = self._calc_homogeneous_matrix(thetas)

        # 最終リンクの同時変換行列(最終リンク座標の位置・姿勢)より，手先位置を計算する
        final_link_matrix = homogeneou_matrix[self._DIMENTION_LINK - 1]
        # 最終リンクから手先位置までの相対位置(4ベクトル)を定義
        relative_pos = np.ones(self._HOMOGENEOU_MAT_ELEMENT)
        relative_pos[:self._HOMOGENEOU_MAT_ELEMENT - 1] = self._relative_positions[-1]
        # 手先位置(x, y, z)・姿勢(Z-Y-Xオイラー角)を保存する
        pose = np.zeros(self._DIMENTION_POSE)
        # 手先位置(x, y, z)を保存
        pose[:DIMENTION_3D] = np.dot(final_link_matrix, relative_pos)[:DIMENTION_3D]
        # 最終リンクの回転行列からZ-Y-Xオイラー角を計算
        zyx_euler = self._rot.rot_to_zyx_euler(final_link_matrix[:self._ROTATION_MAT_ELEMENT, :self._ROTATION_MAT_ELEMENT])
        # 手先位置姿勢を保存(手先姿勢は最終リンクと同じ姿勢と仮定する)
        pose[DIMENTION_3D:] = zyx_euler

        return pose

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

        # 同時変換行列(6 * 4 * 4)を計算する
        homogeneou_matix = self._calc_homogeneous_matrix(thetas)

        # 全リンクの座標系の原点を取得
        all_link_pose = np.zeros((self._DIMENTION_LINK + 1, self._DIMENTION_POSE))
        for i, matrix in enumerate(homogeneou_matix):
            # 同時変換行列から位置を取得
            all_link_pose[i, :DIMENTION_3D] = matrix[:self._ROTATION_MAT_ELEMENT, self._ROTATION_MAT_ELEMENT].reshape(1, -1)
            # 同時変換行列から回転行列を取得して，Z-Y-Xオイラー角に変換する
            zyx_euler = self._rot.rot_to_zyx_euler(matrix[:self._ROTATION_MAT_ELEMENT, :self._ROTATION_MAT_ELEMENT])
            all_link_pose[i, DIMENTION_3D:] = zyx_euler

        # 最後のリンクの座標
        # 最後のリンクの座標系の原点から，手先の位置を計算する
        pos = np.ones(self._HOMOGENEOU_MAT_ELEMENT)
        pos[:DIMENTION_3D] = self._relative_positions[-1]
        all_link_pose[-1, :DIMENTION_3D] = np.dot(homogeneou_matix[-1], pos)[:DIMENTION_3D].reshape(1, -1)
        # 手先姿勢は最終リンクの姿勢と一致していると考え，最終リンクの回転行列をZ-Y-Xオイラー角に変換する
        zyx_euler = self._rot.rot_to_zyx_euler(homogeneou_matix[-1, :self._ROTATION_MAT_ELEMENT, :self._ROTATION_MAT_ELEMENT])
        all_link_pose[-1, DIMENTION_3D:] = zyx_euler

        return all_link_pose
