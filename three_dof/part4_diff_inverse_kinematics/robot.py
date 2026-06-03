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
        _jacov_thetas(list): 微分逆行列で取得した角度を保存

    メソッド
        public
            forward_kinematics(): 順運動学 (ロボットの関節角度からロボットの手先位置を算出)
            inverse_kinematics(): 逆運動学 (ロボットの手先位置からロボットの関節角度を算出)
            forward_kinematics_all_pos(): 順運動学で全リンクの位置を取得
            update(): 角度を与えて，各リンクの直方体を更新する
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

    _DETERMINANT_THRESHOLD = 1e-4           # 行列式の閾値
    _BOX_WIDTH = 1e-2                       # 各リンクの幅を定義
    _JACOV_DELTA_TIME = 0.10                # ヤコビの1時刻
    _JACOV_NEAR_POS   = 1e-6                # 目標位置との近傍距離 [m]
    _JACOV_MAX_COUNT  = 100                 # ヤコビの最大回数


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

    def _reset_jacov_thetas(self):
        """
        _jacov_thetasプロパティのリセット
        """
        if len(self._jacov_thetas) != 0:
            self._jacov_thetas.clear()

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

    @property
    def jacov_thetas(self):
        """
        _jacov_thetasプロパティのゲッター
        """
        return self._jacov_thetas

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

    def differential_inverse_kinematics(self, thetas, target_pos):
        """
        2点間の微分逆運動学

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]
            target_pos(numpy.ndarray): 目標位置 (位置[m]・姿勢[rad])

        戻り値
            target_thetas(numpy.ndarray): 目標位置の関節角度 [rad]
        """
        raise NotImplementedError("update() is necessary override.")

    def _jacobian(self, thetas):
        """
        ヤコビ行列の計算

        パラメータ
            thetas(numpy.ndarray): ロボットの関節角度 [rad]

        戻り値
            jacovian(numpy.ndarray): ヤコビ行列
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

    _THETA1_XY_THRESHOLD   = 1e-4           # theta1算出時のx, y閾値


    def __init__(self, links):
        """
        コンストラクタ

        パラメータ
            links(numpy.ndarray): ロボットのリンク長 [m]
        """
        # 親クラスの初期化
        super().__init__(links)
        # ロボットの各リンクを直方体として定義する
        self._objects = []
        # (全リンクの角度を0とした時の) 各リンク間の相対位置
        self._relative_positions = np.zeros((self._DIMENTION_POSE + 1, 3))
        self._relative_positions[0] = np.array([0, 0, 0])
        self._relative_positions[1] = np.array([0,              0, self._links[0]])
        self._relative_positions[2] = np.array([self._links[1], 0, 0             ])
        self._relative_positions[3] = np.array([self._links[2], 0, 0             ])

        # リンク1は回転軸がz軸，リンク2とリンク3は回転軸がy軸である
        self._axiss = [ROTATION_Z_AXIS, ROTATION_Y_NEGATIVE_AXIS, ROTATION_Y_NEGATIVE_AXIS]
        # 初期角度
        initial_thetas = np.zeros(self._DIMENTION_THETA)
        # 順運動学により，全リンク(ベースリンク，リンク1，リンク2，リンク3)の位置を計算
        all_link_pose = self.forward_kinematics_all_link_pos(initial_thetas)

        # 1つ前のリンクの回転行列を更新
        prev_rotation = np.eye(self._ROTATION_MAT_ELEMENT)

        # 各リンクの角度が全部0の時のx, y, zを定義
        box_xyz = [[self._BOX_WIDTH * 2, self._BOX_WIDTH * 2, self._links[0]     ],
                   [self._links[1],      self._BOX_WIDTH * 2, self._BOX_WIDTH * 2],
                   [self._links[2],      self._BOX_WIDTH * 2, self._BOX_WIDTH * 2]]

        # ロボットの各リンクを直方体として定義する
        for i in range(self._DIMENTION_THETA):
            # 各リンクの回転行列を定義
            rotation = self._rot.rot(initial_thetas[i], self._axiss[i])
            rotation = np.dot(prev_rotation, rotation)
            # 各リンクの中心位置 (x, y, z) を定義
            center = (all_link_pose[i + 1] - all_link_pose[i]) / 2 + all_link_pose[i]
            # 直方体の定義 (x, y, zの長さを保存)
            x, y, z = box_xyz[i]
            box = fcl.Box(x, y, z)
            # 直方体の中心を定義 (位置・姿勢)
            translation = fcl.Transform(rotation, center)
            obj = fcl.CollisionObject(box, translation)
            # モデルを追加
            self._objects.append(obj)
            # 1つ前のリンクの回転行列を更新
            prev_rotation = rotation

        # 直方体をAABBとして，定義
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
            pose(numpy.ndarray): ロボットの手先位置 (位置) [m]
        """
        # パラメータの次元数を確認
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

        # 同時変換行列(3 * 4 * 4)を計算する
        homogeneou_matrix = self._calc_homogeneous_matrix(thetas)

        # 最終リンクの同時変換行列(最終リンク座標の位置・姿勢)より，手先位置を計算する
        final_link_matrix = homogeneou_matrix[self._DIMENTION_LINK - 1]
        # 最終リンクから手先位置までの早退位置(4ベクトル)を定義
        relative_pos = np.ones(self._HOMOGENEOU_MAT_ELEMENT)
        relative_pos[:self._HOMOGENEOU_MAT_ELEMENT - 1] = self._relative_positions[-1]
        pose = np.dot(final_link_matrix, relative_pos)
        # 手先位置(x, y)を取得
        pose = pose[:self._DIMENTION_POSE]

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
        # 引数の確認
        if np.size(thetas) != self._DIMENTION_THETA:
            # 異常
            raise ValueError(f"theta's size is abnormal. theta's size is {np.size(thetas)}")

        # 現在の関節角度を保存 (異なるアドレスにデータを保存するためにnp.copy()を採用)
        current_thetas = np.copy(thetas)

        # 目標値の関節角度が見つかったどうか
        success_flg = False

        # 微分逆行列で取得したデータを初期化
        self._reset_jacov_thetas()
        # 計算した角度を保存する
        self._jacov_thetas.append(current_thetas)

        # 目標位置に近づくまでループ
        for i in range(self._JACOV_MAX_COUNT):
            # ヤコビ行列の取得
            jacovian = self._jacobian(current_thetas)

            # dP(位置の差分) = J(ヤコビ行列) * dTheta(角度の差分)
            # dTheta = J^(-1)(ヤコビ行列の逆行列) * dP
            # 行列式が0近傍であるかの確認(ヤコビ行列の逆行列の存在確認)
            det = np.linalg.det(jacovian)
            if abs(det) <= self._DETERMINANT_THRESHOLD:
                # 0近傍であるため，逆行列が存在しない
                raise ValueError(f"abs(det) is near 0. abs(det) is {abs(det)}")

            # 現在の手先位置を計算
            current_pos = self.forward_kinematics(current_thetas)
            # 位置の差分を計算
            dP = target_pos - current_pos
            # dTheta = J^(-1)(ヤコビ行列の逆行列) * dP
            dTheta = np.dot(np.linalg.inv(jacovian), dP)

            # 関節角度の更新 (+=とするとcurrent_thetasが常に同じアドレスになるため，current_thetasを異なるアドレスとしたい)
            # print(f"id(current_thetas) = {id(current_thetas)}")   # アドレス確認
            # current_thetas = current_thetas + dTheta * self._JACOV_DELTA_TIME
            # 位置の差分が大きいほど，角度の更新量を小さくしたい
            coefficient = 1 / max(0.1, np.linalg.norm(dP))
            current_thetas = current_thetas + dTheta * coefficient * self._JACOV_DELTA_TIME
            # print(f"id(current_thetas) = {id(current_thetas)}")   # アドレス確認

            # 計算した角度を保存する
            self._jacov_thetas.append(current_thetas)

            # 更新後の手先位置が目標位置の近傍であれば，処理終了とする
            current_pos = self.forward_kinematics(current_thetas)
            distance = np.linalg.norm(target_pos - current_pos)
            # print(f"distance = {distance}")
            if distance <= self._JACOV_NEAR_POS:
                # 近傍のため，処理終了
                success_flg = True
                break

        if not success_flg:
            # 目標位置の関節角度が見つからない
            raise ValueError(f"target_pos's theta is not found. please change target_pos")

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
        # 引数の確認
        if np.size(thetas) != self._DIMENTION_THETA:
            # 異常
            raise ValueError(f"theta's size is abnormal. theta's size is {np.size(thetas)}")

        # jacovian = [[-l1 * sin(theta1) - l2 * sin(theta12), -l2 * sin(theta12)], [l1 * cos(theta1) + l2 * cos(theta12), l2 * cos(theta12)]]
        # 各リンクの長さをローカル変数に保存
        l1 = self._links[0]
        l2 = self._links[1]
        l3 = self._links[2]

        # 三角関数の計算
        sin1  = np.sin(thetas[0])
        cos1  = np.cos(thetas[0])
        sin2  = np.sin(thetas[1])
        cos2  = np.cos(thetas[1])
        sin23 = np.sin(thetas[1] + thetas[2])
        cos23 = np.cos(thetas[1] + thetas[2])

        # ヤコビ行列
        jacovian =    np.array([[-sin1 * (l2 * cos2 + l3 * cos23), -cos1 * (l2 * sin2 + l3 * sin23), -cos1 * l3 * sin23],
                                [ cos1 * (l2 * cos2 + l3 * cos23), -sin1 * (l2 * sin2 + l3 * sin23), -sin1 * l3 * sin23],
                                [ 0                              ,          l2 * cos2 + l3 * cos23 ,         l3 * cos23]])

        return jacovian

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

        # 同時変換行列(3 * 4 * 4)を計算する
        homogeneou_matix = self._calc_homogeneous_matrix(thetas)

        # 全リンクの座標系の原点を取得
        all_link_pose = np.zeros((self._DIMENTION_LINK + 1, self._DIMENTION_POSE))
        for i, matrix in enumerate(homogeneou_matix):
            # 同時変換行列から位置を取得
            pos = matrix[:self._DIMENTION_POSE, self._ROTATION_MAT_ELEMENT].reshape(1, -1)
            all_link_pose[i] = pos

        # 最後のリンクの座標
        # 最後のリンクの座標系の原点から，手先の位置を計算する
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
        # パラメータの次元数を確認
        if np.size(thetas) != self._DIMENTION_THETA:
            raise ValueError(f"thetas's size is abnormal. thetas's size is {np.size(thetas)}")

        # 順運動学により，全リンク(ベースリンク，リンク1，リンク2，リンク3)の位置を計算
        all_link_pose = self.forward_kinematics_all_link_pos(thetas)
        # 1つ前のリンクの回転行列を定義
        prev_rotation = np.eye(self._ROTATION_MAT_ELEMENT)

        # ロボットの各リンクを直方体として定義する
        for i in range(self._DIMENTION_THETA):
            # 各リンクの回転行列を定義
            rotation = self._rot.rot(thetas[i], self._axiss[i])
            # 1つ前のリンクの回転も考慮する
            rotation = np.dot(prev_rotation, rotation)
            # 各リンクの中心位置 (x, y, z) を定義
            center = (all_link_pose[i + 1] - all_link_pose[i]) / 2 + all_link_pose[i]
            # 直方体の中心を定義 (位置・姿勢)
            translation = fcl.Transform(rotation, center)
            # モデルの位置を更新
            self._objects[i].setTransform(translation)
            # 1つ前のリンクの回転行列を更新
            prev_rotation = rotation

        # AABBを更新
        self._manager.update()
