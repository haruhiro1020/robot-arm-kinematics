# メイン処理

# ライブラリの読み込み
import numpy as np                  # 数値計算
import matplotlib.pyplot as plt     # 描画

# 自作モジュールの読み込み
from constant import *              # 定数
from robot import Robot3DoF         # ロボットクラス
from animation import RRTRobotAnimation    # ロボットのアニメーション


LINE_WIDTH = 3                      # プロット時の線の太さ


def main():
    """
    メイン処理
    """
    # 3軸ロボットのリンク長
    links = np.array([1.0, 1.0, 1.0])
    # 3軸ロボットのインスタンスを作成
    robot = Robot3DoF(links)

    # 始点
    # start_pos = np.array([1.0, 0.0])
    start_pos = np.array([1.0, -1.0, 1.0])
    # 終点
    # end_pos = np.array([0.0, 1.0])
    end_pos = np.array([-1.0, 1.0, 2.0])

    try:
        # 始点と終点の逆運動学
        start_theta = robot.inverse_kinematics(start_pos)
        end_theta   = robot.inverse_kinematics(end_pos)
    except Exception as e:
        # 逆運動学の解が存在しない (始点または終点が異常な値)
        raise ValueError(f"please start_pos or end_pos is change. pos is singularity")

    # 目標位置の角度を取得
    target_theta = robot.differential_inverse_kinematics(start_theta, end_pos)
    print(f"target_theta = {target_theta}")
    print(f"end_theta = {end_theta}")
    print(f"robot.jacov_thetas[-1] = {np.array(robot.jacov_thetas[-1])}")
    print(f"robot.forward_kinematics(target_theta) = {robot.forward_kinematics(target_theta)}")

    # アニメーション作成
    robotAnime = RRTRobotAnimation()

    # 関節空間による RRT 経路生成
    # ファイル名
    file_name = "jacov_robot_anime.gif"
    robotAnime.plot_Animation(DIMENTION_3D, robot, np.array(robot.jacov_thetas), None, file_name)


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
