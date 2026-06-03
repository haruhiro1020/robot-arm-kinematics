# メイン処理

# ライブラリの読み込み
import numpy as np                  # 数値計算
import matplotlib.pyplot as plt     # 描画

# 自作モジュールの読み込み
from robot import Robot2DoF         # ロボットクラス


FORWARD_KINE = False                 # 順運動学の計算フラグ (True / False = 順運動学 / 逆運動学)
LINE_WIDTH = 3                      # プロット時の線の太さ
GRAPH_FORWARD_FILE_NAME = "plot_2drob_forward_kinematics.png"
GRAPH_INVERSE_FILE_NAME = "plot_2drob_inverse_kinematics.png"


def forward(links, robot, thetas, plot_flg=True):
    """
    順運動学

    パラメータ
        links(numpy.ndarray): ロボットのリンク長 [m]
        robot(Robot2DoF): ロボットクラス
        thetas(numpy.ndarray): 関節角度 [rad]
        plot_flg(bool): 描画の有無 (True / False = 描画する / しない)

    戻り値
        pose(numpy.ndarray): 手先位置
    """
    # 順運動学により，位置を算出
    pose = robot.forward_kinematics(thetas)

    if plot_flg:
        plot(robot, thetas)

    return pose


def inverse(links, robot, pose, plot_flg=True, upper=True):
    """
    逆運動学

    パラメータ
        links(numpy.ndarray): ロボットのリンク長 [m]
        robot(Robot2DoF): ロボットクラス
        pose(numpy.ndarray): 手先位置 [m]
        plot_flg(bool): 描画の有無 (True / False = 描画する / しない)
        upper(bool): 関節が上向きとするかどうか (True / False = Yes / No)

    戻り値
        thetas(list): 関節角度
    """
    # 順運動学により，位置を算出
    thetas = robot.inverse_kinematics(pose, upper=upper)
    if plot_flg:
        plot(robot, thetas)

    return thetas


def main():
    """
    メイン処理
    """
    # 2軸ロボットのリンク長
    links = np.array([1.0, 1.0])
    # 2軸ロボットのインスタンスを作成
    robot = Robot2DoF(links)
    # プロット有無
    plot_flg = True
    # 逆運動学の上向き解とするかの有無
    uppers = [True, False]

    # 順運動学で使用する関節角度 [deg] を定義
    deg_thetas = [30, 120]
    candidate = np.deg2rad(deg_thetas)
    all_thetas = np.array([[theta1, theta2] for theta1 in candidate for theta2 in candidate])

    if FORWARD_KINE:
        # 順運動学
        file_name = GRAPH_FORWARD_FILE_NAME
        for thetas in all_thetas:
            pose = forward(links, robot, thetas, plot_flg=plot_flg)
    else:
        # 逆運動学
        file_name = GRAPH_INVERSE_FILE_NAME
        for thetas in all_thetas:
            for upper in uppers:
                # 順運動学で位置を取得
                pose1  = forward(links, robot, thetas, plot_flg=False)
                # 取得した位置で逆運動学
                theta  = inverse(links, robot, pose1,  plot_flg=plot_flg, upper=upper)
                # 逆運動学の結果が正しいかを順運動学で計算
                pose2  = forward(links, robot, theta,  plot_flg=False)
                print(f"pose1 = {pose1[0]:.3f}, {pose1[1]:.3f}")
                print(f"pose2 = {pose2[0]:.3f}, {pose2[1]:.3f}")
            print()

    if plot_flg:
        show(file_name)


def show(graph_file_name):
    """
    描画する

    パラメータ
        graph_file_name(str): 描画ファイル名
    """
    # X軸，Y軸の名称
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")

    # X軸，Y軸の描画範囲
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)

    plt.grid()
    plt.legend(fontsize=6, loc='center left', bbox_to_anchor=(1., .5))
    # 凡例を見切れないようにするために，plt.tight_layout()
    plt.tight_layout()
    plt.savefig(graph_file_name)
    plt.show()


def plot(robot, thetas):
    """
    ロボットをプロット (将来的にはクラス化する)

    パラメータ
        robot(Robot2DoF): ロボット
        thetas(numpy.ndarray): 回転角度
    """
    # 回転角度とリンク長をローカル変数に保存
    theta1 = thetas[0]
    theta2 = thetas[1]
    link1  = robot.links[0]
    link2  = robot.links[1]

    # 三角関数を計算
    s1  = np.sin(theta1)
    c1  = np.cos(theta1)
    s2  = np.sin(theta2)
    c2  = np.cos(theta2)
    s12 = np.sin(theta1 + theta2)
    c12 = np.cos(theta1 + theta2)

    # 各リンクの位置を算出
    base_pos  = np.zeros(2)
    link1_pos = link1 * np.array([c1, s1])
    link2_pos = link1_pos + link2 * np.array([c12, s12])

    # 全リンクの位置を算出
    all_link_pos = np.array([base_pos, link1_pos, link2_pos])
    # 全リンクの回転角度を算出
    all_link_theta = np.array([theta1, theta2])

    # 線プロット
    plt.plot(all_link_pos[:, 0], all_link_pos[:, 1], linewidth=LINE_WIDTH, label=f"theta1={np.rad2deg(theta1):.1f} [deg], theta2={np.rad2deg(theta2):.1f} [deg]")
    # 点プロット
    plt.scatter(all_link_pos[:, 0], all_link_pos[:, 1])


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
