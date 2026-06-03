# ロボットアームの運動学を記載

# ライブラリの読み込み
import numpy as np

# サードパーティーの読み込み
import fcl

# 自作モジュールの読み込み
from constant import *              # 定数
from rotation import MyRotation     # 回転行列


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
        return self._links

    @property
    def manager(self):
        return self._manager

    def forward_kinematics(self, thetas):
        raise NotImplementedError("forward_kinematics() is necessary override.")

    def inverse_kinematics(self, pose):
        raise NotImplementedError("inverse_kinematics() is necessary override.")

    def forward_kinematics_all_pos(self, thetas):
        raise NotImplementedError("forward_kinematics() is necessary override.")

    def update(self, thetas):
        raise NotImplementedError("update() is necessary override.")

    def _calc_homogeneous_matrix(self, thetas):
        """
        同時変換行列の計算

        パラメータ
            thetas(numpy.ndarray): 関節角度 [rad]

        戻り値
            homogeneou_matix(numpy.ndarray): 全リンクの同時変換行列
        """
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

        homogeneou_matix = np.zeros((self._DIMENTION_LINK, self._HOMOGENEOU_MAT_ELEMENT, self._HOMOGENEOU_MAT_ELEMENT))

        prev_homogeneou_matrix = np.eye(self._HOMOGENEOU_MAT_ELEMENT)
        for i in range(self._DIMENTION_THETA):
            homogeneou_matix[i, -1, -1] = 1
            rotation_matrix   = self._rot.rot(thetas[i], self._axiss[i])
            relative_position = self._relative_positions[i].reshape(1, -1)
            homogeneou_matix[i, :self._ROTATION_MAT_ELEMENT, :self._ROTATION_MAT_ELEMENT] = rotation_matrix
            homogeneou_matix[i, :self._ROTATION_MAT_ELEMENT,  self._ROTATION_MAT_ELEMENT] = relative_position
            homogeneou_matix[i] = np.dot(prev_homogeneou_matrix, homogeneou_matix[i])
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
        super().__init__(links)
        self._objects = []
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
        # 初期角度
        initial_thetas = np.zeros(self._DIMENTION_THETA)
        # 順運動学により，全リンクの位置を計算
        all_link_pose = self.forward_kinematics_all_link_pos(initial_thetas)

        # 1つ前のリンクの回転行列を更新
        prev_rotation = np.eye(self._ROTATION_MAT_ELEMENT)

        # ロボットの各リンクを直方体として定義する
        for i in range(self._DIMENTION_THETA):
            rotation = self._rot.rot(initial_thetas[i], self._axiss[i])
            rotation = np.dot(prev_rotation, rotation)
            center = (all_link_pose[i + 1, :DIMENTION_3D] - all_link_pose[i, :DIMENTION_3D]) / 2 + all_link_pose[i, :DIMENTION_3D]
            x, y, z = [self._BOX_WIDTH * 2, self._BOX_WIDTH * 2, self._links[i]]
            box = fcl.Box(x, y, z)
            translation = fcl.Transform(rotation, center)
            obj = fcl.CollisionObject(box, translation)
            self._objects.append(obj)
            prev_rotation = rotation

        # DynamicAABBTreeCollisionManager に登録
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.registerObjects(self._objects)
        self._manager.setup()

    def forward_kinematics(self, thetas):
        """
        順運動学 (ロボットの関節角度からロボットの手先位置を算出)

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]

        戻り値
            pose(numpy.ndarray): ロボットの手先位置・姿勢(Z-Y-Xオイラー角) [m] / [rad]
        """
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

        homogeneou_matrix = self._calc_homogeneous_matrix(thetas)

        final_link_matrix = homogeneou_matrix[-1]
        relative_pos = np.ones(self._HOMOGENEOU_MAT_ELEMENT)
        relative_pos[:self._HOMOGENEOU_MAT_ELEMENT - 1] = self._relative_positions[-1]
        pose = np.zeros(self._DIMENTION_POSE)
        pose[:DIMENTION_3D] = np.dot(final_link_matrix, relative_pos)[:DIMENTION_3D]
        zyx_euler = self._rot.rot_to_zyx_euler(final_link_matrix[:self._ROTATION_MAT_ELEMENT, :self._ROTATION_MAT_ELEMENT])
        pose[DIMENTION_3D:] = zyx_euler

        return pose

    def inverse_kinematics(self, pose, right=False, front=True, above=True):
        """
        逆運動学 (ロボットの手先位置からロボットの関節角度を算出)

        パラメータ
            pose(numpy.ndarray): ロボットの手先位置([m])・姿勢([rad])
            right(bool): 腕が右向きかどうか
            front(bool): 正面かどうか
            above(bool): 上向きかどうか

        戻り値
            thetas(numpy.ndarray): ロボットの関節角度 [rad]
        """
        if np.size(pose) != self._DIMENTION_POSE:
            raise ValueError(f"parameter pose's size is abnormal. pose's size is {np.size(pose)}")

        pos = pose[:DIMENTION_3D]
        zyx_euler = pose[DIMENTION_3D:]

        thetas = np.zeros(self._DIMENTION_THETA)

        thetas[:DIMENTION_3D] = self._inverse_kinematics_123(pos, zyx_euler, right=right, front=front)
        thetas[DIMENTION_3D:] = self._inverse_kinematics_456(zyx_euler, thetas[:DIMENTION_3D], above=above)

        return thetas


    def _inverse_kinematics_123(self, pos, zyx_euler, right=False, front=True):
        """
        逆運動学で関節1, 2, 3を計算

        パラメータ
            pos(numpy.ndarray): ロボットの手先位置 [m]
            zyx_euler(numpy.ndarray): ロボットの手先姿勢(Z-Y-Xオイラー角) [rad]
            right(bool): 肘が右向きどうか
            front(bool): 正面かどうか

        戻り値
            theta123(numpy.ndarray): ロボットの関節角度(関節1, 2, 3) [rad]
        """
        l1  = self._links[0]
        l2  = self._links[1]
        l34 = self._links[2] + self._links[3]
        l56 = self._links[4] + self._links[5]

        rotation = self._rot.rot_from_zyx_euler(zyx_euler)
        rotation_z = rotation[:, 2]

        pos5 = pos - rotation_z * l56
        px = pos5[0]
        py = pos5[1]
        pz = pos5[2]

        cos3 = ((px ** 2 + py ** 2 + (pz - l1) ** 2) - (l2 ** 2 + l34 ** 2)) / (2 * l2 * l34)
        if abs(cos3) > 1:
            raise ValueError(f"cos3 is abnormal. cos3 is {cos3}")

        if right:
            sin3 =  np.sqrt(1 - cos3 ** 2)
        else:
            sin3 = -np.sqrt(1 - cos3 ** 2)
        theta3 = np.arctan2(sin3, cos3)

        if abs(px) <= self._THETA1_XY_THRESHOLD and abs(py) <= self._THETA1_XY_THRESHOLD:
            theta1 = 0.0
        else:
            if front:
                theta1 = np.arctan2( py,  px)
            else:
                theta1 = np.arctan2(-py, -px)

        element1 = l2  + l34 * cos3
        element2 = l34 * sin3
        matrix = np.array([[ element1, element2],
                           [-element2, element1]])

        det = np.linalg.det(matrix)
        if abs(det) <= self._DETERMINANT_THRESHOLD:
            raise ValueError(f"det is abnormal. det is {det}")

        position = np.array([np.sqrt(px ** 2 + py ** 2), pz - l1])

        sin2_cos2 = np.dot(np.linalg.inv(matrix), position)
        theta2 = np.arctan2(sin2_cos2[0], sin2_cos2[1])

        theta123 = np.array([theta1, theta2, theta3])

        return theta123

    def _inverse_kinematics_456(self, zyx_euler, theta123, above=True):
        """
        逆運動学で関節4, 5, 6を計算

        パラメータ
            zyx_euler(numpy.ndarray): ロボットの手先姿勢(Z-Y-Xオイラー角) [rad]
            theta123(numpy.ndarray): 関節1, 2, 3の角度 [rad]
            above(bool): 手首が上かどうか

        戻り値
            theta456(numpy.ndarray): ロボットの関節角度(関節4, 5, 6) [rad]
        """
        thetas = np.zeros(self._DIMENTION_THETA)
        thetas[:3] = theta123

        rot_we = self._rot.rot_from_zyx_euler(zyx_euler)

        homogeneou_matrix = self._calc_homogeneous_matrix(thetas)
        rot_w3 = homogeneou_matrix[2, :3, :3]

        rot_36 = np.dot(rot_w3.T, rot_we)

        theta456 = self._rot.rot_to_zyz_euler(rot_36, above)

        return theta456

    def forward_kinematics_all_link_pos(self, thetas):
        """
        順運動学で全リンクの位置を取得 (グラフの描画で使用する)

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]

        戻り値
            all_link_pose(numpy.ndarray): ロボットの全リンク位置 (位置 + 姿勢) [m] + [rad]
        """
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

        homogeneou_matix = self._calc_homogeneous_matrix(thetas)

        all_link_pose = np.zeros((self._DIMENTION_LINK + 1, self._DIMENTION_POSE))
        for i, matrix in enumerate(homogeneou_matix):
            all_link_pose[i, :DIMENTION_3D] = matrix[:self._ROTATION_MAT_ELEMENT, self._ROTATION_MAT_ELEMENT].reshape(1, -1)
            zyx_euler = self._rot.rot_to_zyx_euler(matrix[:self._ROTATION_MAT_ELEMENT, :self._ROTATION_MAT_ELEMENT])
            all_link_pose[i, DIMENTION_3D:] = zyx_euler

        pos = np.ones(self._HOMOGENEOU_MAT_ELEMENT)
        pos[:DIMENTION_3D] = self._relative_positions[-1]
        all_link_pose[-1, :DIMENTION_3D] = np.dot(homogeneou_matix[-1], pos)[:DIMENTION_3D].reshape(1, -1)
        zyx_euler = self._rot.rot_to_zyx_euler(homogeneou_matix[-1, :self._ROTATION_MAT_ELEMENT, :self._ROTATION_MAT_ELEMENT])
        all_link_pose[-1, DIMENTION_3D:] = zyx_euler

        return all_link_pose

    def update(self, thetas):
        """
        角度を与えて，各リンクの直方体を更新する

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]
        """
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

        all_link_pose = self.forward_kinematics_all_link_pos(thetas)

        for i in range(self._DIMENTION_THETA):
            rotation = self._rot.rot_from_zyx_euler(all_link_pose[i, DIMENTION_3D:])
            center = (all_link_pose[i + 1, :DIMENTION_3D] - all_link_pose[i, :DIMENTION_3D]) / 2 + all_link_pose[i, :DIMENTION_3D]
            translation = fcl.Transform(rotation, center)
            self._objects[i].setTransform(translation)

        self._manager.update()
