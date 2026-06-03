# メイン処理

# ライブラリの読み込み
import numpy as np                  # 数値計算
import matplotlib.pyplot as plt     # 描画

# 自作モジュールの読み込み
from constant import *              # 定数
from robot import Robot2DoF         # ロボットクラス
from animation import RobotAnimation    # ロボットのアニメーション


def main():
    """
    メイン処理
    """
    # レーベンバーグ・マルカート法の実装有無
    # levenberg = True
    levenberg = False

    # 2軸ロボットのリンク長
    links = np.array([1.0, 1.0])
    # 2軸ロボットのインスタンスを作成
    robot = Robot2DoF(links, levenberg=levenberg)

    # 始点
    start_pos = np.array([-1.9, 0.1])
    # 終点
    end_pos   = np.array([ 1.9, 0.1])

    try:
        # 始点の逆運動学
        start_theta = robot.inverse_kinematics(start_pos)
    except Exception as e:
        # 逆運動学の解が存在しない
        raise ValueError(f"please start_pos is change. start_pos is singularity")

    # 目標位置の角度を取得
    target_theta = robot.differential_inverse_kinematics(start_theta, end_pos)
    # print(f"target_theta = {target_theta}")
    # print(f"robot.jacov_thetas[-1] = {robot.jacov_thetas[-1]}")
    # print(f"forwrad_kinematics = {robot.forward_kinematics(target_theta)}")

    # アニメーション作成
    robotAnime = RobotAnimation()

    # ファイル名
    file_name = "jacov_robot_anime.gif"
    # robot.jacov_thetasはlist型であり，numpy.ndarray型に変換するためにnp.array()を採用
    robotAnime.plot_Animation(DIMENTION_2D, robot, np.array(robot.jacov_thetas), None, file_name)


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
