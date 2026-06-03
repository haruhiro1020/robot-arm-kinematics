# メイン処理

# ライブラリの読み込み
import numpy as np                  # 数値計算
import matplotlib.pyplot as plt     # 描画

# 自作モジュールの読み込み
from robot import Robot2DoF         # ロボットクラス


LINE_WIDTH = 3                      # プロット時の線の太さ


def main():
    """
    メイン処理
    """
    # 2軸ロボットのリンク長
    links = np.array([1.0, 1.0])
    # 2軸ロボットのインスタンスを作成
    robot = Robot2DoF(links)

    # 関節角度 [deg] を定義
    deg_thetas = [0, 30, 60, 90]
    # [deg]から[rad]へ変換
    candidate  = np.deg2rad(deg_thetas)
    all_thetas = np.array([[theta1, theta2] for theta1 in candidate for theta2 in candidate])

    # 順運動学により，位置を算出
    for thetas in all_thetas:
        pose = robot.forward_kinematics(thetas)
        plot(robot, thetas)

    # X軸，Y軸の名称
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")

    # X軸，Y軸の描画範囲
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)

    plt.grid()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.legend(fontsize=6)
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

    # 線プロット (小数点1桁まで記載)
    plt.plot(all_link_pos[:, 0], all_link_pos[:, 1], linewidth=LINE_WIDTH, label=f"theta1={np.rad2deg(theta1):.1f} [deg], theta2={np.rad2deg(theta2):.1f} [deg]")
    # 点プロット
    plt.scatter(all_link_pos[:, 0], all_link_pos[:, 1])


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
