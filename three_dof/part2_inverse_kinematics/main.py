# メイン処理

# ライブラリの読み込み
import numpy as np                  # 数値計算
import matplotlib.pyplot as plt     # 描画

# 自作モジュールの読み込み
from constant import *              # 定数
from robot import Robot3DoF         # ロボットクラス


FORWARD_KINE = False                 # 順運動学の計算フラグ (True / False = 順運動学 / 逆運動学)
LINE_WIDTH = 3                      # プロット時の線の太さ
GRAPH_FORWARD_FILE_NAME = "plot_3drob_forward_kinematics.png"
GRAPH_INVERSE_FILE_NAME = "plot_3drob_inverse_kinematics.png"


def forward(robot, thetas, axis, plot_flg=True, label=True):
    """
    順運動学

    パラメータ
        robot(Robot3DoF): ロボットクラス
        thetas(numpy.ndarray): 関節角度 [rad]
        axis(figure.add_subplot): 3次元の描画軸
        plot_flg(bool): 描画の有無 (True / False = 描画する / しない)
        label(bool): 描画時のラベルの有無

    戻り値
        pose(numpy.ndarray): 手先位置
    """
    # 順運動学により，位置を算出
    pose = robot.forward_kinematics(thetas)

    if plot_flg:
        plot(robot, thetas, axis, label=label)
    else:
        print(f"forward theta = {thetas}")
        print(f"forward pose  = {pose}")
        pass

    return pose


def inverse(robot, pose, axis, plot_flg=True, upper=True, front=True, label=True):
    """
    逆運動学

    パラメータ
        robot(Robot3DoF): ロボットクラス
        pose(numpy.ndarray): 手先位置 [m]
        axis(figure.add_subplot): 3次元の描画軸
        plot_flg(bool): 描画の有無 (True / False = 描画する / しない)
        upper(bool): 関節が上向きとするかどうか (True / False = Yes / No)
        front(bool): 正面向きとするかどうか (True / False = Yes / No)
        label(bool): 描画時のラベルの有無

    戻り値
        thetas(list): 関節角度
    """
    # 順運動学により，位置を算出
    thetas = robot.inverse_kinematics(pose, upper=upper, front=front)
    if plot_flg:
        plot(robot, thetas, axis)
    else:
        print(f"inverse pose  = {pose}")
        print(f"inverse theta = {thetas}")
        pass

    return thetas


def main():
    """
    メイン処理
    """
    # 3軸ロボットのリンク長
    links = np.array([1.0, 1.0, 1.0])
    # 3軸ロボットのインスタンスを作成
    robot = Robot3DoF(links)
    # プロット有無
    plot_flg = True
    # 逆運動学の上向き解とするかの有無
    uppers = [True, False]

    # 順運動学で使用する関節角度 [deg] を定義
    deg_thetas = [0, 150]
    candidate = np.deg2rad(deg_thetas)
    all_thetas = np.array([[theta1, theta2, theta3] for theta1 in candidate for theta2 in candidate for theta3 in candidate])

    # 3次元グラフの作成
    fig = plt.figure()
    axis = fig.add_subplot(111, projection="3d")

    if FORWARD_KINE:
        # 順運動学
        file_name = GRAPH_FORWARD_FILE_NAME
        for thetas in all_thetas:
            pose = forward(robot, thetas, axis, plot_flg=plot_flg)
    else:
        # 逆運動学
        file_name = GRAPH_INVERSE_FILE_NAME
        for thetas in all_thetas:
            for upper in uppers:
                # 順運動学で位置を取得
                pose1  = forward(robot, thetas, axis, plot_flg=False)
                # 取得した位置で逆運動学
                theta  = inverse(robot, pose1,  axis, plot_flg=plot_flg, upper=upper)
                # 逆運動学の結果が正しいかを順運動学で計算
                pose2  = forward(robot, theta,  axis, plot_flg=False)
                print()
            print()
            print()

    if plot_flg:
        show(axis, file_name)


def show(axis, graph_file_name):
    """
    描画する

    パラメータ
        axis(figure.add_subplot): 3次元の描画軸
        graph_file_name(str): 描画ファイル名
    """
    # X軸，Y軸，Z軸の名称
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")

    axis.grid()
    axis.set_aspect("equal")
    plt.legend(fontsize=5, loc='center left', bbox_to_anchor=(0.8, .5))
    # # 凡例を見切れないようにするために，plt.tight_layout()
    plt.tight_layout()
    plt.savefig(graph_file_name)
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
