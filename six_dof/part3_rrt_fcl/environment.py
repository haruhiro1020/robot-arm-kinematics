# python-fclによる干渉判定チェック

# 標準ライブラリの読み込み
import numpy as np

# サードパーティーの読み込み
import fcl

# 自作モジュールの読み込み
from constant import *              # 定数
from rotation import MyRotation       # 回転行列


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
            ball = fcl.Sphere(radius)
            translation = fcl.Transform(np.array([x, y, z]))
            obj = fcl.CollisionObject(ball, translation)
            objects.append(obj)
        self._interferences[INTERFERENCE.BALL] = balls

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
        request  = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
        col_data = fcl.CollisionData(request=request)

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
        if margin < 0:
            margin = self._INITIAL_MARGIN

        dist_data   = fcl.DistanceData()

        self._manager.distance(obj, dist_data, fcl.defaultDistanceCallback)
        min_dist = dist_data.result.min_distance
        self._distance = min_dist
        # print(f"min_dist = {min_dist}")

        if min_dist > margin:
            collision = False
        else:
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
    def __init__(self):
        """
        コンストラクタ
        """
        objects = []
        self._interferences = {}
        self._rot = MyRotation()

        # 円形の干渉物を定義 (円の位置 (x, y, z) と半径を定義)
        balls = [[1.5, 0.3, 1.5, 0.3], [0.5, -0.5, 2.0, 0.3], [1.5, 0.5, 1.0, 0.5], [0.5, 1.5, 1.5, 0.5]]
        for x, y, z, radius in balls:
            ball = fcl.Sphere(radius)
            translation = fcl.Transform(np.array([x, y, z]))
            obj = fcl.CollisionObject(ball, translation)
            objects.append(obj)
        self._interferences[INTERFERENCE.BALL] = balls

        # 直方体の干渉物を定義
        # 直方体の長さ (x, y, z) と中心点の位置 (x, y, z)，x,y,z軸の回転角度 [deg]
        cuboids = [[0.4, -0.6, 0.5, 1.4, -0.6, 1.5, 0, 0, 0], [0.6, -0.2, 1.0, 1.2, 0.7, 2.0, 0, 0, 0]]
        for x, y, z, center_x, center_y, center_z, angle_x, angle_y, angle_z in cuboids:
            cuboid = fcl.Box(x, y, z)
            rotation_x = self._rot.rot(np.deg2rad(angle_x), ROTATION_X_AXIS)
            rotation_y = self._rot.rot(np.deg2rad(angle_y), ROTATION_Y_AXIS)
            rotation_z = self._rot.rot(np.deg2rad(angle_z), ROTATION_Z_AXIS)
            rotation = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
            translation = fcl.Transform(rotation, np.array([center_x, center_y, center_z]))
            obj = fcl.CollisionObject(cuboid, translation)
            objects.append(obj)
        self._interferences[INTERFERENCE.CUBOID] = cuboids

        # DynamicAABBTreeCollisionManager に登録
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.registerObjects(objects)
        self._manager.setup()

        # 最短距離を更新
        self._distance = 0.0
