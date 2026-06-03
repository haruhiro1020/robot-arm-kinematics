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

# 0割を防ぐための定数
EPSILON                   = 1e-6

# RRTで使用するノードの最短ノード要素番号 (ノード末尾に親ノード番号を格納する)
RRT_NEAR_NODE_IDX         = -1

# 始点ノードの親ノード番号 (始点には親ノードが存在しない)
INITIAL_NODE_NEAR_NODE    = -1

# 干渉物の名称を定義
class INTERFERENCE(Enum):
    """
    干渉物の名称を定義
    """
    NONE         = ""          # 未定義
    CIRCLE       = "circle"    # 円形の干渉物
    RECTANGLE    = "rectangle" # 長方形の干渉物
    BALL         = "ball"      # 球の干渉物
    CUBOID       = "cuboid"    # 直方体の干渉物

# 補間方法の定義
class INTERPOLATION(Enum):
    """
    補間方法の定義
    """
    JOINT    = 10   # 関節空間での補間
    POSITION = 11   # 位置空間での補間
