# 環境の作成

# 標準ライブラリの読み込み
import numpy as np

# サードパーティーの読み込み
import fcl

# 自作モジュールの読み込み
from constant import *              # 定数
from rotation import Rotation       # 回転行列


class Base2DEnv:
    """
    経路生成時の2次元環境

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
    _Z_VALUE = 0.0          # z軸方向の値

    def __init__(self):
        """
        コンストラクタ
        """
        objects = []
        self._interferences = {}
        self._rot = Rotation()

        # 円形の干渉物を定義 (円の位置 (x, y) と半径を定義)
        circles = [[0.8, 0.8, 0.3], [1.2, 0.8, 0.3], [1.2, 1.2, 0.3], [0.8, 1.2, 0.3]]
        for x, y, radius in circles:
            # fclに円が定義されていないため，楕円のz軸を0として円とする
            circle = fcl.Ellipsoid(radius, radius, self._Z_VALUE)
            # 円の中心点が原点であるため，中心点を平行移動させる
            translation = fcl.Transform(np.array([x, y, self._Z_VALUE]))
            obj = fcl.CollisionObject(circle, translation)
            # モデルを追加
            objects.append(obj)
        # 円形の干渉物を保存
        self._interferences[INTERFERENCE.CIRCLE] = circles

        # # 長方形の干渉物を定義
        # 長方形の長さ (x, y) と中心点の位置 (x, y)
        # rectangles = [[0.2, 0.3, 0.6, 0.6], [0.4, 0.5, 0.8, 1.0], [1.0, 1.0, 0.8, 0.8]]
        # for x, y, center_x, center_y in rectangles:
        #     # 直方体の長さ (x, y, z) を設定
        #     rectangle = fcl.Box(x, y, self._Z_VALUE)
        #     # 長方形の中心点を設定．無回転とする
        #     translation = fcl.Transform(np.array([center_x, center_y, self._Z_VALUE]))
        #     obj = fcl.CollisionObject(rectangle, translation)
        #     # モデルを追加
        #     objects.append(obj)
        # # 長方形の干渉物を保存
        # self._interferences[INTERFERENCE.RECTANGLE] = rectangles

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

    def is_collision_dist(self, obj, margin=0.0):
        """
        干渉判定 + 最短距離

        パラメータ
            obj(fcl.CollisionObject): 干渉物
            margin(float): 干渉判定のマージン

        戻り値
            collision(bool): True / False = 干渉あり / 干渉なし
        """
        dist_data   = fcl.DistanceData()

        # 最短距離の結果を保存する
        self._manager.distance(obj, dist_data, fcl.defaultDistanceCallback)
        min_dist = dist_data.result.min_distance
        # 最短距離の更新
        self._distance = min_dist

        if min_dist > margin:
            # 干渉なし
            collision = False
        else:
            # 干渉あり
            collision = True

        return collision


class Robot2DEnv(Base2DEnv):
    """
    経路生成時の2次元環境 (ロボット用)

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

        # 円形の干渉物を定義 (円の位置 (x, y) と半径を定義)
        circles = [[1.5, 0.3, 0.3], [0.5, -0.5, 0.3]]
        for x, y, radius in circles:
            # fclに円が定義されていないため，球とする
            circle = fcl.Sphere(radius)
            # 円の中心点が原点であるため，中心点を平行移動させる
            translation = fcl.Transform(np.array([x, y, self._Z_VALUE]))
            obj = fcl.CollisionObject(circle, translation)
            # モデルを追加
            objects.append(obj)
        # 円形の干渉物を保存
        self._interferences[INTERFERENCE.CIRCLE] = circles

        # 長方形の干渉物を定義
        # 長方形の長さ (x, y) と中心点の位置 (x, y)，角度 [deg]
        rectangles = [[0.4, 0.6, 1.4, -0.6, 0], [0.6, 0.2, 1.2, 0.7, 0]]
        for x, y, center_x, center_y, angle in rectangles:
            # 直方体の長さ (x, y, z) を設定
            rectangle = fcl.Box(x, y, self._Z_VALUE)
            # 回転行列の作成
            rotation = self._rot.rot(np.deg2rad(angle), ROTATION_Z_AXIS)
            # 長方形の中心点，回転行列を設定
            translation = fcl.Transform(rotation, np.array([center_x, center_y, self._Z_VALUE]))
            obj = fcl.CollisionObject(rectangle, translation)
            # モデルを追加
            objects.append(obj)
        # 長方形の干渉物を保存
        self._interferences[INTERFERENCE.RECTANGLE] = rectangles

        # DynamicAABBTreeCollisionManager に登録
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.registerObjects(objects)
        self._manager.setup()

        # 最短距離を更新
        self._distance = 0.0
