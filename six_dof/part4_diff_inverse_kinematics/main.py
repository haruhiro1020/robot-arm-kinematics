# メイン処理

# ライブラリの読み込み
import numpy as np                  # 数値計算

# 自作モジュールの読み込み
from constant import *              # 定数
from robot import Robot6DoF         # ロボットクラス
from animation import RRTRobotAnimation    # ロボットのアニメーション
from rotation import MyRotation


LINE_WIDTH = 3                      # プロット時の線の太さ


def main():
    """
    メイン処理
    """
    # 6軸ロボットのリンク長
    links = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    # 6軸ロボットのインスタンスを作成
    robot = Robot6DoF(links)

    # 始点 (位置(x, y, z), 姿勢(z, y, x))
    start_pos = np.array([ 1.0, -1.0, 1.0, 0, 0      , 0])
    # 終点
    end_pos   = np.array([-1.0,  1.0, 2.0, 0, np.pi/2, 0])

    try:
        # 始点と終点の逆運動学
        start_theta = robot.inverse_kinematics(start_pos)
        end_theta   = robot.inverse_kinematics(end_pos)
    except Exception as e:
        # 逆運動学の解が存在しない (始点または終点が異常な値)
        raise ValueError(f"please start_pos or end_pos is change. pos is singularity")

    # 目標位置の角度を取得
    target_theta = robot.differential_inverse_kinematics(start_theta, end_pos)
    target_pos = robot.forward_kinematics(target_theta)
    end_pos = robot.forward_kinematics(end_theta)
    # print(f"robot.forward_kinematics(target_theta) = {target_pos}")
    # print(f"robot.forward_kinematics(end_theta) = {end_pos}")

    # 回転行列の計算
    target_rotation = MyRotation().rot_from_zyx_euler(target_pos[3:])
    end_rotation    = MyRotation().rot_from_zyx_euler(end_pos[3:])
    # print(f"target_rotation = {target_rotation}")
    # print(f"end_rotation    = {end_rotation}")

    # アニメーション作成
    robotAnime = RRTRobotAnimation()

    # ファイル名
    file_name = "jacovian_robot_pos_anime.gif"
    robotAnime.plot_Animation(DIMENTION_6D, robot, np.array(robot.jacov_thetas), None, file_name)


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
