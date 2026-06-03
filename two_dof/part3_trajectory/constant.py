# 複数ファイルで使用する定数の定義
from enum import Enum
from enum import auto


# 次元数を定義
DIMENTION_NONE            = -1          # 未定義
DIMENTION_2D              =  2          # 2次元
DIMENTION_3D              =  3          # 3次元

# 補間時の分割する時の距離
DEVIDED_DISTANCE_JOINT      = 0.1      # 関節補間時の距離 [rad]
DEVIDED_DISTANCE_POSIION    = 0.1      # 位置補間時の距離 [m]

# プロットする時にグラフ(静止画)とするかアニメーションを定義
PLOT_NONE                 = 20          # プロットしない
PLOT_GRAPH                = 21          # グラフ
PLOT_ANIMATION            = 22          # アニメーション

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
