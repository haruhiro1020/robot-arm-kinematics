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
        _jacov_thetas(list): 微分逆行列で取得した角度を保存

    メソッド
        public
            forward_kinematics(): 順運動学 (ロボットの関節角度からロボットの手先位置を算出)
            inverse_kinematics(): 逆運動学 (ロボットの手先位置からロボットの関節角度を算出)
            forward_kinematics_all_pos(): 順運動学で全リンクの位置を取得
            update(): 角度を与えて，各リンクの直方体を更新する
            differential_inverse_kinematics(): 微分逆運動学
            links(): _linksプロパティのゲッター
            manager(): _managerプロパティのゲッター
            jacov_thetas(): _jacov_thetasプロパティのゲッター

        protected
            _calc_homogeneous_matrix(): 同時変換行列の計算
            _jacobian(): ヤコビ行列
    """
    # 定数の定義
    _DIMENTION_POSE  = DIMENTION_NONE       # 手先位置の次元数
    _DIMENTION_THETA = DIMENTION_NONE       # 関節角度の次元数
    _DIMENTION_LINK  = DIMENTION_NONE       # リンク数
    _DIMENTION_AXIS  = DIMENTION_NONE       # 回転軸数

    _INITIAL_THETA   = 0.0                  # 初期回転角度 [rad]
    _HOMOGENEOU_MAT_ELEMENT = 4             # 同時変換行列の次元数
    _ROTATION_MAT_ELEMENT   = 3             # 回転行列の次元数

    _DETERMINANT_THRESHOLD = 1e-6           # 行列式の閾値
    _BOX_WIDTH = 1e-2                       # 各リンクの幅を定義
    _JACOV_DELTA_TIME = 0.1                 # ヤコビの1時刻
    _JACOV_NEAR_POS   = 1e-10               # 目標位置との近傍距離 [m]
    _JACOV_MAX_COUNT  = 100                 # ヤコビの最大回数

    _LEVENBERG_MARUQUARDT_LAMBDA = 1        # Levenberg-Marquardt法で使用するパラメータ


    def __init__(self, links, levenberg=True):
        """
        コンストラクタ

        パラメータ
            links(numpy.ndarray): ロボットのリンク長 [m]
            levenberg(bool): レーベンバーグ・マルカート法の実装有無
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
        self._levenberg_maruquardt = levenberg

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

    def differential_inverse_kinematics(self, thetas, target_pos):
        raise NotImplementedError("differential_inverse_kinematics() is necessary override.")

    def _jacobian(self, thetas):
        raise NotImplementedError("_jacobian() is necessary override.")

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


class Robot2DoF(Robot):
    """
    2軸ロボットクラス

    プロパティ
        _links(numpy.ndarray): ロボットのリンク長
        _rot(Rotation): 回転行列クラス
        _axiss(list): 関節の回転軸

    メソッド
        public
            forward_kinematics(): 順運動学 (ロボットの関節角度からロボットの手先位置を算出)
    """
    # 定数の定義
    _DIMENTION_POSE  = DIMENTION_2D         # 手先位置の次元数
    _DIMENTION_THETA = DIMENTION_2D         # 関節角度の次元数
    _DIMENTION_LINK  = DIMENTION_2D         # リンク数


    def __init__(self, links, levenberg=True):
        """
        コンストラクタ

        パラメータ
            links(numpy.ndarray): ロボットのリンク長 [m]
            levenberg(bool): レーベンバーグ・マルカート法の実装有無
        """
        super().__init__(links, levenberg)
        self._objects = []
        # (全リンクの角度を0とした時の) 各リンク間の相対位置
        self._relative_positions = np.zeros((self._DIMENTION_POSE + 1, 3))
        self._relative_positions[0] = np.array([0, 0, 0])
        self._relative_positions[1] = np.array([self._links[0], 0, 0])
        self._relative_positions[2] = np.array([self._links[1], 0, 0])

        # リンク1とリンク2は回転軸がz軸である
        self._axiss = [ROTATION_Z_AXIS, ROTATION_Z_AXIS]
        # 初期角度
        initial_thetas = np.zeros(self._DIMENTION_THETA)
        # 順運動学により，全リンクの位置を計算
        all_link_pose = self.forward_kinematics_all_link_pos(initial_thetas)

        prev_rotation = np.eye(self._ROTATION_MAT_ELEMENT)

        for i in range(self._DIMENTION_THETA):
            rotation = self._rot.rot(initial_thetas[i], self._axiss[i])
            rotation = np.dot(prev_rotation, rotation)
            center = np.zeros(DIMENTION_3D)
            center[:self._DIMENTION_POSE] = all_link_pose[i + 1] / 2 + all_link_pose[i]
            box = fcl.Box(self._relative_positions[i + 1, 0], 2 * self._BOX_WIDTH, 2 * self._BOX_WIDTH)
            translation = fcl.Transform(rotation, center)
            obj = fcl.CollisionObject(box, translation)
            self._objects.append(obj)
            prev_rotation = rotation

        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.registerObjects(self._objects)
        self._manager.setup()

        self._jacov_thetas = []

    def forward_kinematics(self, thetas):
        """
        順運動学 (ロボットの関節角度からロボットの手先位置を算出)

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]

        戻り値
            pose(numpy.ndarray): ロボットの手先位置 (位置) [m]
        """
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

        sin1  = np.sin(thetas[0])
        cos1  = np.cos(thetas[0])
        cos1_sin1 = np.array([cos1, sin1])
        sin12 = np.sin(thetas[0] + thetas[1])
        cos12 = np.cos(thetas[0] + thetas[1])
        cos12_sin12 = np.array([cos12, sin12])
        pose  = self._links[0] * cos1_sin1 + self._links[1] * cos12_sin12

        return pose

    def differential_inverse_kinematics(self, thetas, target_pos):
        """
        2点間の微分逆運動学

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]
            target_pos(numpy.ndarray): 目標位置 (位置[m]・姿勢[rad])

        戻り値
            target_thetas(numpy.ndarray): 目標位置の関節角度 [rad]
        """
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"theta's size is abnormal. theta's size is {np.size(thetas)}")

        current_thetas = np.copy(thetas)

        self._reset_jacov_thetas()
        self._jacov_thetas.append(current_thetas)

        for _ in range(self._JACOV_MAX_COUNT):
            jacovian = self._jacobian(current_thetas)

            current_pos = self.forward_kinematics(current_thetas)
            dP = target_pos - current_pos

            if self._levenberg_maruquardt:
                # 特異点対応として，レーベンバーグ・マルカート法を実施
                new_jacobian = np.dot(jacovian.T, jacovian) + self._lambda * np.eye(jacovian.shape[0])
                dTheta = np.dot(np.dot(np.linalg.inv(new_jacobian), jacovian.T), dP)
            else:
                # 特異点対応の未実施
                det = np.linalg.det(jacovian)
                if abs(det) <= self._DETERMINANT_THRESHOLD:
                    raise ValueError(f"abs(det) is near 0. abs(det) is {abs(det)}")

                dTheta = np.dot(np.linalg.inv(jacovian), dP)

            # 関節角度の更新
            # print(f"id(current_thetas) = {id(current_thetas)}")   # アドレス確認
            # 位置の差分が大きいほど，角度の更新量を小さくしたい
            coefficient = 1 / max(0.5, np.linalg.norm(dP))
            coefficient = 1
            current_thetas = current_thetas + dTheta * coefficient * self._JACOV_DELTA_TIME
            # 範囲 [-π, π] に正規化（任意）
            current_thetas = (current_thetas + np.pi) % (2 * np.pi) - np.pi

            self._jacov_thetas.append(current_thetas)

            current_pos = self.forward_kinematics(current_thetas)
            distance = np.linalg.norm(target_pos - current_pos)
            if distance <= self._JACOV_NEAR_POS:
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

        l1 = self._links[0]
        l2 = self._links[1]

        sin1  = np.sin(thetas[0])
        cos1  = np.cos(thetas[0])
        sin12 = np.sin(thetas[0] + thetas[1])
        cos12 = np.cos(thetas[0] + thetas[1])

        jacovian = np.array([[-l1 * sin1 - l2 * sin12, -l2 * sin12],
                             [ l1 * cos1 + l2 * cos12,  l2 * cos12]])

        return jacovian

    def inverse_kinematics(self, pose, upper=False):
        """
        逆運動学 (ロボットの手先位置からロボットの関節角度を算出)

        パラメータ
            pose(numpy.ndarray): ロボットの手先位置 (位置) [m]
            upper(bool): 腕が上向かどうか

        戻り値
            thetas(numpy.ndarray): ロボットの関節角度 [rad]
        """
        if np.size(pose) != self._DIMENTION_POSE:
            raise ValueError(f"parameter pose's size is abnormal. pose's size is {np.size(pose)}")

        px = pose[0]
        py = pose[1]
        l1 = self._links[0]
        l2 = self._links[1]
        cos2  = ((px ** 2 + py ** 2) - (l1 ** 2 + l2 ** 2)) / (2 * l1 * l2)
        if cos2 < -1 or cos2 > 1:
            raise ValueError(f"cos2 is abnormal. cos2 is {cos2}")

        sin2 = np.sqrt(1 - cos2 ** 2)
        theta2 = np.arctan2(sin2, cos2)
        if not upper:
            theta2 = -theta2
            sin2 = np.sin(theta2)
            cos2 = np.cos(theta2)

        element1 =  l1 + l2 * cos2
        element2 = -l2 * sin2
        matrix = np.array([[ element1, element2],
                           [-element2, element1]])
        det = np.linalg.det(matrix)
        if det <= self._DETERMINANT_THRESHOLD and det >= -self._DETERMINANT_THRESHOLD:
            raise ValueError(f"det is abnormal. det is {det}")

        cos1_sin1 = np.dot(np.linalg.inv(matrix), pose)
        theta1 = np.arctan2(cos1_sin1[1], cos1_sin1[0])

        thetas = np.array([theta1, theta2])

        return thetas

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
            pos = matrix[:self._DIMENTION_POSE, self._ROTATION_MAT_ELEMENT].reshape(1, -1)
            all_link_pose[i] = pos

        pos = np.ones(self._HOMOGENEOU_MAT_ELEMENT)
        pos[:DIMENTION_3D] = self._relative_positions[-1]
        all_link_pose[-1]  = np.dot(homogeneou_matix[-1], pos)[:self._DIMENTION_POSE].reshape(1, -1)

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
        prev_rotation = np.eye(self._ROTATION_MAT_ELEMENT)

        for i in range(self._DIMENTION_THETA):
            rotation = self._rot.rot(thetas[i], self._axiss[i])
            rotation = np.dot(prev_rotation, rotation)
            center = np.zeros(DIMENTION_3D)
            center[:self._DIMENTION_POSE] = (all_link_pose[i + 1] - all_link_pose[i]) / 2 + all_link_pose[i]
            translation = fcl.Transform(rotation, center)
            self._objects[i].setTransform(translation)
            prev_rotation = rotation

        self._manager.update()
