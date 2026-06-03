# メイン処理

# ライブラリの読み込み
import numpy as np                  # 数値計算
import matplotlib.pyplot as plt     # 描画

# 自作モジュールの読み込み
from constant import *              # 定数
from robot import Robot2DoF         # ロボットクラス
from animation import RobotAnimation    # ロボットのアニメーション
from rrt import RRTRobot            # 経路生成
from environment import Robot2DEnv  # 経路生成の環境


def main():
    """
    メイン処理
    """
    # 2軸ロボットのリンク長
    links = np.array([1.0, 1.0])
    # 2軸ロボットのインスタンスを作成
    robot = Robot2DoF(links)
    # 始点
    # start_pos = np.array([1.0, 0.0])
    start_pos = np.array([1.0, -1.0])
    # 終点
    # end_pos = np.array([0.0, 1.0])
    end_pos   = np.array([1.0,  1.0])

    try:
        # 始点と終点の逆運動学
        start_theta = robot.inverse_kinematics(start_pos)
        end_theta   = robot.inverse_kinematics(end_pos)
    except Exception as e:
        # 逆運動学の解が存在しない (始点または終点が異常な値)
        raise ValueError(f"please start_pos or end_pos is change. pos is singularity")

    # 環境
    environment = Robot2DEnv()
    # 補間の種類 (位置空間/関節空間)
    interpolations = [INTERPOLATION.JOINT, INTERPOLATION.POSITION]
    for interpolation in interpolations:
        # RRTによる経路生成
        rrt = RRTRobot()
        if interpolation == INTERPOLATION.POSITION:
            # 位置空間での経路生成の実行
            planning_result = rrt.planning(start_pos,   end_pos,   environment, robot, interpolation)
        else:
            # 関節空間での経路生成の実行
            planning_result = rrt.planning(start_theta, end_theta, environment, robot, interpolation)

        if not planning_result:
            # 経路生成の失敗
            return

        # アニメーション作成
        robotAnime = RobotAnimation()

        if interpolation == INTERPOLATION.POSITION:
            # 位置空間による RRT 経路生成
            # 逆運動学による関節を取得
            thetas = np.zeros((rrt.pathes.shape[0], DIMENTION_2D))
            for i, pos in enumerate(rrt.pathes):
                # 逆運動学
                theta = robot.inverse_kinematics(pos)
                thetas[i] = theta
            # ファイル名
            file_name = "rrt_robot_pos_anime.gif"
            robotAnime.plot_Animation(robot,     thetas, environment, file_name)
        else:
            # 関節空間による RRT 経路生成
            # ファイル名
            file_name = "rrt_robot_joint_anime.gif"
            robotAnime.plot_Animation(robot, rrt.pathes, environment, file_name)


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
