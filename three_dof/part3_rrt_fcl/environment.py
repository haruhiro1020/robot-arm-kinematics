# 環境の作成

# 標準ライブラリの読み込み
import numpy as np

# サードパーティーの読み込み
import fcl

# 自作モジュールの読み込み
from constant import *              # 定数
from rotation import Rotation       # 回転行列


class Base3DEnv:
    """
    経路生成時の3次元環境

    プロパティ
        _distance(float): 干渉物との最短距離
        _inteferences(dist): 干渉物の情報 (名称(球や直方体など) + 形状(半径，中心点など))

    メソッド
        public
            inteferences(): _inteferencesプロパティのゲッター
            distance(): _distanceプロパティのゲッター
            is_collision(): 干渉判定
            is_collision_dist(): 干渉判定 + 最短距離
    """
    # 定数の定義
    _INITIAL_MARGIN = 0.0   # 干渉物とのマージン [m]

    def __init__(self):
        """
        コンストラクタ
        """
        objects = []
        self._interferences = {}

        # 球の干渉物を定義 (球の位置 (x, y, z) と半径を定義)
        balls = [[0.8, 0.8, 0.8, 0.3], [1.2, 0.8, 0.8, 0.3], [1.2, 1.2, 1.2, 0.3], [0.8, 1.2, 1.2, 0.3]]
        for x, y, z, radius in balls:
            # 球の半径を設定
            ball = fcl.Sphere(radius)
            # 球の中心点を平行移動
            translation = fcl.Transform(np.array([x, y, z]))
            obj = fcl.CollisionObject(ball, translation)
            # モデルを追加
            objects.append(obj)
        # 球の干渉物を保存
        self._interferences[INTERFERENCE.BALL] = balls

        # # 直方体の干渉物を定義
        # # 直方体の長さ (x, y, z) と中心点の位置 (x, y, z)
        # cuboids = [[0.2, 0.3, 0.3, 0.6, 0.6, 0.6], [0.4, 0.5, 0.6, 0.8, 1.0, 1.0], [1.0, 1.0, 1.0, 0.8, 0.8, 0.8]]
        # for x, y, z, center_x, center_y, center_z in cuboids:
        #     # 直方体の各辺の長さを設定
        #     cuboid = fcl.Box(x, y, z)
        #     # 長方形の中心点を設定．無回転とする
        #     translation = fcl.Transform(np.array([center_x, center_y, center_z]))
        #     obj = fcl.CollisionObject(cuboid, translation)
        #     # モデルを追加
        #     objects.append(obj)
        # # 直方体の干渉物を保存
        # self._interferences[INTERFERENCE.CUBOID] = cuboids

        # DynamicAABBTreeCollisionManager に登録
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.registerObjects(objects)
        self._manager.setup()

        # 最短距離を更新
        self._distance = 0.0

    @property
    def distance(self):
        """
        _distanceプロパティのゲッター
        """
        return self._distance

    @property
    def interferences(self):
        """
        _interferencesプロパティのゲッター (描画で使用する)
        """
        return self._interferences

    def is_collision(self, obj):
        """
        干渉判定

        パラメータ
            obj(fcl.CollisionObject): 干渉物

        戻り値
            is_collision(bool): True / False = 干渉あり / 干渉なし
        """
        # 衝突リクエスト
        # num_max_contacts ... 最大の衝突点数
        # enable_contact ... 衝突点情報の有無
        request  = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
        col_data = fcl.CollisionData(request=request)

        # 本環境と干渉物の干渉判定
        self._manager.collide(obj, col_data, fcl.defaultCollisionCallback)

        return col_data.result.is_collision

    def is_collision_dist(self, obj, margin=_INITIAL_MARGIN):
        """
        干渉判定 + 最短距離

        パラメータ
            obj(fcl.CollisionObject): 干渉物
            margin(float): 干渉判定のマージン [m]

        戻り値
            collision(bool): True / False = 干渉あり / 干渉なし
        """
        # 引数確認
        if margin < 0:
            margin = self._INITIAL_MARGIN

        dist_data   = fcl.DistanceData()

        # 最短距離の結果を保存する
        self._manager.distance(obj, dist_data, fcl.defaultDistanceCallback)
        min_dist = dist_data.result.min_distance
        # 最短距離の更新
        self._distance = min_dist
        # print(f"min_dist = {min_dist}")

        if min_dist > margin:
            # 干渉なし
            collision = False
        else:
            # 干渉あり
            collision = True

        return collision


class Robot3DEnv(Base3DEnv):
    """
    経路生成時の3次元環境 (ロボット用)

    プロパティ
        _distance(float): 干渉物との最短距離
        _inteferences(dist): 干渉物の情報 (名称(円や長方形など) + 形状(半径，中心点など))

    メソッド
        public
            inteferences(): _inteferencesプロパティのゲッター
            distance(): _distanceプロパティのゲッター
            is_collision(): 干渉判定
            is_collision_dist(): 干渉判定 + 最短距離
    """
    # 定数の定義

    def __init__(self):
        """
        コンストラクタ
        """
        objects = []
        self._interferences = {}
        self._rot = Rotation()

        # 円形の干渉物を定義 (円の位置 (x, y, z) と半径を定義)
        balls = [[0.5, 1.5, 1.5, 0.5], [1.5, 0.3, 1.5, 0.3], [1.5, 0.5, 1.0, 0.5]]
        for x, y, z, radius in balls:
            # fclに円が定義されていないため，球とする
            ball = fcl.Sphere(radius)
            # 円の中心点が原点であるため，中心点を平行移動させる
            translation = fcl.Transform(np.array([x, y, z]))
            obj = fcl.CollisionObject(ball, translation)
            # モデルを追加
            objects.append(obj)
        # 円形の干渉物を保存
        self._interferences[INTERFERENCE.BALL] = balls

        # 直方体の干渉物を定義
        # 直方体の長さ (x, y, z) と中心点の位置 (x, y, z)，x,y,z軸の回転角度 [deg]
        cuboids = [[0.4, 0.3, 0.8, 1.4, -0.6, 1.5, 0, 0, 0], [0.6, 0.8, 1.0, 1.2, 0.7, 2.0, 0, 0, 0]]
        for x, y, z, center_x, center_y, center_z, angle_x, angle_y, angle_z in cuboids:
            # 直方体の長さ (x, y, z) を設定
            cuboid = fcl.Box(x, y, z)
            # 回転行列の作成
            rotation_x = self._rot.rot(np.deg2rad(angle_x), ROTATION_X_AXIS)
            rotation_y = self._rot.rot(np.deg2rad(angle_y), ROTATION_Y_AXIS)
            rotation_z = self._rot.rot(np.deg2rad(angle_z), ROTATION_Z_AXIS)
            rotation = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
            # 長方形の中心点，回転行列を設定
            translation = fcl.Transform(rotation, np.array([center_x, center_y, center_z]))
            obj = fcl.CollisionObject(cuboid, translation)
            # モデルを追加
            objects.append(obj)
        # 長方形の干渉物を保存
        self._interferences[INTERFERENCE.CUBOID] = cuboids

        # DynamicAABBTreeCollisionManager に登録
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.registerObjects(objects)
        self._manager.setup()

        # 最短距離を更新
        self._distance = 0.0
