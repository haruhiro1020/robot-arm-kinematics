# メイン処理

# ライブラリの読み込み
import numpy as np                  # 数値計算
import matplotlib.pyplot as plt     # 描画

# 自作モジュールの読み込み
from constant import *              # 定数
from robot import Robot3DoF         # ロボットクラス


LINE_WIDTH = 3                      # プロット時の線の太さ
GRAPH_FILE_NAME = "plot_3drob_forward_kinematics.png"


def main():
    """
    メイン処理
    """
    # 3軸ロボットのリンク長
    links = np.array([1.0, 1.0, 1.0])
    # 3軸ロボットのインスタンスを作成
    robot = Robot3DoF(links)

    # 関節角度 [deg] を定義
    deg_thetas = [0, 30, 60]
    # [deg]から[rad]へ変換
    candidate  = np.deg2rad(deg_thetas)
    all_thetas = np.array([[theta1, theta2, theta3] for theta1 in candidate for theta2 in candidate for theta3 in candidate])

    # 3次元グラフの作成
    fig = plt.figure()
    axis = fig.add_subplot(111, projection="3d")

    # 順運動学により，位置を算出
    for thetas in all_thetas:
        plot(robot, thetas, axis)

    # X軸，Y軸，Z軸の名称
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")

    axis.grid()
    axis.set_aspect("equal")
    plt.legend(fontsize=5, loc='center left', bbox_to_anchor=(0.8, .5))
    # # 凡例を見切れないようにするために，plt.tight_layout()
    plt.tight_layout()
    plt.savefig(GRAPH_FILE_NAME)
    plt.show()


def plot(robot, thetas, axis, label=True):
    """
    ロボットをプロット (将来的にはクラス化する)

    パラメータ
        robot(Robot3DoF): ロボット
        thetas(numpy.ndarray): 回転角度
        axis(figure.add_subplot): 3次元の描画軸
        label(bool): 描画時のラベルの有無
    """
    # 全リンクの位置を算出
    all_link_pos = robot.forward_kinematics_all_link_pos(thetas)

    # 線プロット
    if label:
        axis.plot(all_link_pos[:, 0], all_link_pos[:, 1], all_link_pos[:, 2], linewidth=LINE_WIDTH, label=f"theta1={np.rad2deg(thetas[0]):.1f} [deg], theta2={np.rad2deg(thetas[1]):.1f} [deg], theta3={np.rad2deg(thetas[2]):.1f} [deg]")
    else:
        axis.plot(all_link_pos[:, 0], all_link_pos[:, 1], all_link_pos[:, 2], linewidth=LINE_WIDTH)

    # 点プロット
    axis.scatter(all_link_pos[:, 0], all_link_pos[:, 1], all_link_pos[:, 2])


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
