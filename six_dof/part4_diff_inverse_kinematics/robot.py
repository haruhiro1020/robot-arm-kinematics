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
    _HOMOGENEOU_MAT_ELEMENT = 4             # 同時変換行列の次元数
    _ROTATION_MAT_ELEMENT   = 3             # 回転行列の次元数

    # ヤコビ行列に関する定数
    _JACOV_DELTA_TIME = 0.2                 # ヤコビの1時刻
    _JACOV_NEAR_POS   = 5e-3                # 目標位置との近傍距離 [m]
    _JACOV_NEAR_ORI   = 5e-2                # 目標位置との近傍姿勢 [rad]
    _JACOV_MAX_COUNT  = 200                 # ヤコビの最大回数
    _JACOV_POS_ERROR_WEIGHT = 1.0           # 微分逆運動学での位置誤差への重み
    _JACOV_ORI_ERROR_WEIGHT = 0.2           # 微分逆運動学での姿勢誤差への重み
    _LEVENBERG_MARUQUARDT_LAMBDA = 0.5      # Levenberg-Marquardt法で使用するパラメータ

    # その他定数
    _INITIAL_THETA   = 0.0                  # 初期回転角度 [rad]
    _DETERMINANT_THRESHOLD = 1e-4           # 行列式の閾値
    _BOX_WIDTH = 1e-2                       # 各リンクの幅を定義


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
        self._jacov_thetas = []
        self._lambda = self._LEVENBERG_MARUQUARDT_LAMBDA

    def _reset_jacov_thetas(self):
        """
        _jacov_thetasプロパティのリセット
        """
        if len(self._jacov_thetas) != 0:
            self._jacov_thetas.clear()

    @property
    def links(self):
        return self._links

    @property
    def manager(self):
        return self._manager

    @property
    def jacov_thetas(self):
        return self._jacov_thetas

    def forward_kinematics(self, thetas):
        raise NotImplementedError("forward_kinematics() is necessary override.")

    def inverse_kinematics(self, pose):
        raise NotImplementedError("inverse_kinematics() is necessary override.")

    def forward_kinematics_all_pos(self, thetas):
        raise NotImplementedError("forward_kinematics() is necessary override.")

    def update(self, thetas):
        raise NotImplementedError("update() is necessary override.")

    def _get_base_jacobian(self, rot_axis, pos_robot, homogeneou_matrix):
        """
        基礎ヤコビ行列の1列分を取得

        パラメータ
            rot_axis(str): 回転軸
            pos_robot(numpy.ndarray): ロボットの手先位置
            homogeneou_matrix(numpy.ndarray): 同時変換行列 (4 * 4行列)

        戻り値
            base_jacobian(numpy.ndarray): 基礎ヤコビ行列の1列分(6 * 1行列)
        """
        if homogeneou_matrix.shape != (4, 4):
            raise ValueError(f"homogeneou_matrix is abnormal. homogeneou_matrix.shape is {homogeneou_matrix.shape}")

        base_jacobian = np.zeros((DIMENTION_6D, 1))

        if rot_axis == ROTATION_X_AXIS:
            axis = np.array([1, 0, 0])
        elif rot_axis == ROTATION_Y_AXIS:
            axis = np.array([0, 1, 0])
        elif rot_axis == ROTATION_Z_AXIS:
            axis = np.array([0, 0, 1])
        else:
            raise ValueError(f"rot_axis is abnormal. rot_axis is {rot_axis}")

        word_axis = np.dot(homogeneou_matrix[:self._ROTATION_MAT_ELEMENT, :self._ROTATION_MAT_ELEMENT], axis)
        position = homogeneou_matrix[:self._ROTATION_MAT_ELEMENT, self._ROTATION_MAT_ELEMENT]

        difference = pos_robot - position.reshape(-1)
        cross_data = np.cross(word_axis, difference)
        base_jacobian[:DIMENTION_3D] = cross_data.reshape(-1, 1)
        base_jacobian[DIMENTION_3D:] = word_axis.reshape(-1, 1)

        return base_jacobian

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
            differential_inverse_kinematics(): 微分逆運動学
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
        initial_thetas = np.zeros(self._DIMENTION_THETA)
        all_link_pose = self.forward_kinematics_all_link_pos(initial_thetas)

        prev_rotation = np.eye(self._ROTATION_MAT_ELEMENT)

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

    def differential_inverse_kinematics(self, thetas, target_pos):
        """
        2点間の微分逆運動学

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]
            target_pos(numpy.ndarray): 目標位置・姿勢 (Z-Y-Xオイラー角) ([m] / [rad])

        戻り値
            target_thetas(numpy.ndarray): 目標位置の関節角度 [rad]
        """
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"theta's size is abnormal. theta's size is {np.size(thetas)}")

        current_thetas = np.copy(thetas)

        self._reset_jacov_thetas()
        self._jacov_thetas.append(current_thetas)

        target_position = target_pos[:DIMENTION_3D]
        target_rotation = self._rot.rot_from_zyx_euler(target_pos[DIMENTION_3D:])
        error_pos_rot = np.zeros(DIMENTION_6D)

        for i in range(self._JACOV_MAX_COUNT):
            jacovian = self._jacobian(current_thetas)

            current_pos = self.forward_kinematics(current_thetas)
            dP = target_position - current_pos[:DIMENTION_3D]
            current_rotation = self._rot.rot_from_zyx_euler(current_pos[DIMENTION_3D:])
            single_axis = self._rot.get_single_axis(current_rotation, target_rotation, t=1)
            dR = single_axis[1:] * single_axis[0]
            error_pos_rot[:DIMENTION_3D] = dP * self._JACOV_POS_ERROR_WEIGHT
            error_pos_rot[DIMENTION_3D:] = dR * self._JACOV_ORI_ERROR_WEIGHT

            new_jacobian = np.dot(jacovian.T, jacovian) + self._lambda * np.eye(jacovian.shape[0])
            dTheta = np.dot(np.dot(np.linalg.inv(new_jacobian), jacovian.T), error_pos_rot)

            current_thetas = current_thetas + dTheta * self._JACOV_DELTA_TIME

            self._jacov_thetas.append(current_thetas)

            current_pos = self.forward_kinematics(current_thetas)
            pos_dist = np.linalg.norm(target_pos[:DIMENTION_3D] - current_pos[:DIMENTION_3D])

            current_rotation = self._rot.rot_from_zyx_euler(current_pos[DIMENTION_3D:])
            single_axis = self._rot.get_single_axis(current_rotation, target_rotation, t=1)
            # print(f"pos_dist = {pos_dist}")
            # print(f"single_axis[0] = {single_axis[0]}")
            if pos_dist <= self._JACOV_NEAR_POS and single_axis[0] <= self._JACOV_NEAR_ORI:
                break

        target_thetas = current_thetas

        return target_thetas

    def _jacobian(self, thetas):
        """
        ヤコビ行列の計算

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]

        戻り値
            jacovian(numpy.ndarray): ヤコビ行列
        """
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"theta's size is abnormal. theta's size is {np.size(thetas)}")

        jacovian = np.zeros((DIMENTION_6D, self._DIMENTION_THETA))
        homogeneou_matrix = self._calc_homogeneous_matrix(thetas)
        robot_pos = self.forward_kinematics(thetas)[:DIMENTION_3D]
        for idx in range(self._DIMENTION_THETA):
            jacovian[:, idx] = self._get_base_jacobian(self._axiss[idx], robot_pos, homogeneou_matrix[idx]).reshape(-1)

        return jacovian

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
        """
        thetas = np.zeros(self._DIMENTION_THETA)
        thetas[:3] = theta123

        rot_we = self._rot.rot_from_zyx_euler(zyx_euler)

        homogeneou_matrix = self._calc_homogeneous_matrix(thetas)
        rot_w3 = homogeneou_matrix[2, :3, :3]

        rot_36 = np.dot(rot_w3.T, rot_we)

        theta456 = self._rot.rot_to_zyz_euler(rot_36)

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
