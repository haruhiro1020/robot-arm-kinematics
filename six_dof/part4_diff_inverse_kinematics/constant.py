# 複数ファイルで使用する定数の定義
from enum import Enum
from enum import auto


# 次元数を定義
DIMENTION_NONE  = -1    # 未定義
DIMENTION_2D    =  2    # 2次元
DIMENTION_3D    =  3    # 3次元
DIMENTION_6D    =  6    # 6次元

# 回転軸
ROTATION_X_AXIS = "rot_x"   # x軸周りに回転
ROTATION_Y_AXIS = "rot_y"   # y軸周りに回転
ROTATION_Z_AXIS = "rot_z"   # z軸周りに回転
ROTATION_X_NEGATIVE_AXIS = "rot_neg_x"  # x軸周りに逆回転
ROTATION_Y_NEGATIVE_AXIS = "rot_neg_y"  # y軸周りに逆回転
ROTATION_Z_NEGATIVE_AXIS = "rot_neg_z"  # z軸周りに逆回転

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

# 0割を防ぐための定数
EPSILON                   = 1e-6
