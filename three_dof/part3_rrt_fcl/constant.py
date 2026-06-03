# 複数ファイルで使用する定数の定義
from enum import Enum
from enum import auto


# 次元数を定義
DIMENTION_NONE  = -1    # 未定義
DIMENTION_2D    =  2    # 2次元
DIMENTION_3D    =  3    # 3次元

# ノードの最短ノード要素を定義
RRT_NEAR_NODE_IDX = -1    # RRTでの最短ノード要素

# 回転軸
ROTATION_X_AXIS          = "rot_x"      # x軸周りに回転
ROTATION_Y_AXIS          = "rot_y"      # y軸周りに回転
ROTATION_Z_AXIS          = "rot_z"      # z軸周りに回転
ROTATION_Y_NEGATIVE_AXIS = "rot_neg_y"  # y軸周りに逆回転
ROTATION_Z_NEGATIVE_AXIS = "rot_neg_z"  # z軸周りに逆回転

# ツリーの初期ノードの親ノード
INITIAL_NODE_NEAR_NODE    = -1          # 初期ノードに親ノードが存在しないから-1

# 0割を防ぐための定数
EPSILON                   = 1e-6

# 干渉物の名称を定義
class INTERFERENCE(Enum):
    """
    干渉物の名称を定義
    """
    NONE         = ""          # 未定義
    # 2次元の干渉物 ↓
    CIRCLE       = "circle"    # 円形の干渉物
    RECTANGLE    = "rectangle" # 長方形の干渉物
    LINE2D       = "line2D"    # 2次元の直線
    # 2次元の干渉物 ↑
    # 3次元の干渉物 ↓
    BALL         = "ball"      # 球の干渉物
    CUBOID       = "cuboid"    # 直方体の干渉物
    LINE3D       = "line3D"    # 3次元の直線
    # 3次元の干渉物 ↑

# 補間方法の定義
class INTERPOLATION(Enum):
    """
    補間方法
    """
    JOINT     = 10      # 関節補間
    POSITION  = auto()  # 位置補間
    LINEAR    = auto()  # 直線補間
    CUBIC     = auto()  # 3次補間
    QUINTIC   = auto()  # 5次補間
