# メイン処理

# ライブラリの読み込み
import numpy as np                  # 数値計算
import matplotlib.pyplot as plt     # 描画


# 自作モジュールの読み込み
from constant import *              # 定数
from rotation import MyRotation     # 回転行列



N_MIDDLE = 101          # 中間姿勢の数
COLOR = ["red", "blue", "black", "orange", "purple", "green", "magenta", "lime", "cyan"]
FILE_NAME = "rotation_trajectory.png"


def main():
    """
    メイン処理
    """
    rot = MyRotation()
    # 初期角度と目標角度より，初期姿勢と目標姿勢を算出
    init_thetas = np.deg2rad(np.array([0, 0, 0]))
    target_thetas = np.deg2rad(np.array([90, 60, 30]))
    init_rot = rot.rot_from_zyx_euler(init_thetas)
    target_rot = rot.rot_from_zyx_euler(target_thetas)

    # 回転行列の軌道を保存
    plot_rotation = np.zeros((N_MIDDLE, 3, 3))

    # 一軸回転法
    for i in range(N_MIDDLE):
        t = i / 100
        # print(f"t = {t}")
        middle_rot = rot.intermediate_rot(init_rot, target_rot, t)
        plot_rotation[i] = middle_rot

    # print(f"diff_rot = {np.dot(init_rot.T, target_rot)}")
    # print(f"plot_rotation[-1] = {plot_rotation[-1]}")

    # 回転行列の各要素を描画
    plt.plot(plot_rotation[:, 0, 0], label="r11", color=COLOR[0])
    plt.plot(plot_rotation[:, 0, 1], label="r12", color=COLOR[1])
    plt.plot(plot_rotation[:, 0, 2], label="r13", color=COLOR[2])
    plt.plot(plot_rotation[:, 1, 0], label="r21", color=COLOR[3])
    plt.plot(plot_rotation[:, 1, 1], label="r22", color=COLOR[4])
    plt.plot(plot_rotation[:, 1, 2], label="r23", color=COLOR[5])
    plt.plot(plot_rotation[:, 2, 0], label="r31", color=COLOR[6])
    plt.plot(plot_rotation[:, 2, 1], label="r32", color=COLOR[7])
    plt.plot(plot_rotation[:, 2, 2], label="r33", color=COLOR[8])

    # 初期値をプロット
    plot_scatter(init_rot, 0)
    # 目標位置をプロット
    plot_scatter(target_rot, N_MIDDLE)

    plt.legend()
    plt.grid()

    # ファイル保存
    plt.savefig(FILE_NAME)

    plt.show()


def plot_scatter(rot, x):
    """
    各要素を点プロット

    パラメータ
        rot(numpy.ndarray): 回転行列
        x(int): x軸の位置
    """
    plt.scatter(x, rot[0, 0], color=COLOR[0])
    plt.scatter(x, rot[0, 1], color=COLOR[1])
    plt.scatter(x, rot[0, 2], color=COLOR[2])
    plt.scatter(x, rot[1, 0], color=COLOR[3])
    plt.scatter(x, rot[1, 1], color=COLOR[4])
    plt.scatter(x, rot[1, 2], color=COLOR[5])
    plt.scatter(x, rot[2, 0], color=COLOR[6])
    plt.scatter(x, rot[2, 1], color=COLOR[7])
    plt.scatter(x, rot[2, 2], color=COLOR[8])


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
