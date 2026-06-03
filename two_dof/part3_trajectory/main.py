# メイン処理

# ライブラリの読み込み
import numpy as np                  # 数値計算

# 自作モジュールの読み込み
from constant import *              # 定数
from robot import Robot2DoF         # ロボットクラス
from animation import RobotAnimation  # アニメーションクラス


# 終端時刻
T_FINAL = 1.0
# 分割数
N_STEPS = 50


def linear_interpolation(start, end, n_steps):
    """
    直線補間

    パラメータ
        start(numpy.ndarray): 初期値
        end(numpy.ndarray): 終端値
        n_steps(int): 分割数

    戻り値
        trajectory(numpy.ndarray): 軌道
    """
    t_final = T_FINAL
    times = np.linspace(0, t_final, n_steps)
    a = (end - start) / t_final
    b = start
    trajectory = np.array([a * t + b for t in times])
    return trajectory


def cubic_interpolation(start, end, n_steps):
    """
    3次多項式による補間 (初期/終端速度=0)

    パラメータ
        start(numpy.ndarray): 初期値
        end(numpy.ndarray): 終端値
        n_steps(int): 分割数

    戻り値
        trajectory(numpy.ndarray): 軌道
    """
    t_final = T_FINAL
    times = np.linspace(0, t_final, n_steps)
    delta = end - start
    a = -2 * delta / (t_final ** 3)
    b =  3 * delta / (t_final ** 2)
    c = np.zeros_like(start)
    d = start
    trajectory = np.array([a * t**3 + b * t**2 + c * t + d for t in times])
    return trajectory


def quintic_interpolation(start, end, n_steps):
    """
    5次多項式による補間 (初期/終端速度=0, 初期/終端加速度=0)

    パラメータ
        start(numpy.ndarray): 初期値
        end(numpy.ndarray): 終端値
        n_steps(int): 分割数

    戻り値
        trajectory(numpy.ndarray): 軌道
    """
    t_final = T_FINAL
    times = np.linspace(0, t_final, n_steps)
    delta = end - start
    a =   6 * delta / (t_final ** 5)
    b = -15 * delta / (t_final ** 4)
    c =  10 * delta / (t_final ** 3)
    d = np.zeros_like(start)
    e = np.zeros_like(start)
    f = start
    trajectory = np.array([a * t**5 + b * t**4 + c * t**3 + d * t**2 + e * t + f for t in times])
    return trajectory


def joint_space_trajectory(robot, start_pose, end_pose, interpolation_func, anime_file_name):
    """
    関節空間内での軌道生成 (パターン1)

    パラメータ
        robot(Robot2DoF): ロボットクラス
        start_pose(numpy.ndarray): 初期手先位置 [m]
        end_pose(numpy.ndarray): 終端手先位置 [m]
        interpolation_func: 補間関数
        anime_file_name(str): アニメーションファイル名
    """
    # 逆運動学により，初期/終端位置から関節角度を算出
    start_thetas = robot.inverse_kinematics(start_pose, upper=False)
    end_thetas   = robot.inverse_kinematics(end_pose,   upper=False)

    # 関節空間内で補間
    thetas_trajectory = interpolation_func(start_thetas, end_thetas, N_STEPS)

    # アニメーション作成
    robot_animation = RobotAnimation()
    robot_animation.plot_Animation(robot, thetas_trajectory, anime_file_name)


def position_space_trajectory(robot, start_pose, end_pose, interpolation_func, anime_file_name):
    """
    位置空間内での軌道生成 (パターン2)

    パラメータ
        robot(Robot2DoF): ロボットクラス
        start_pose(numpy.ndarray): 初期手先位置 [m]
        end_pose(numpy.ndarray): 終端手先位置 [m]
        interpolation_func: 補間関数
        anime_file_name(str): アニメーションファイル名
    """
    # 位置空間内で補間
    pose_trajectory = interpolation_func(start_pose, end_pose, N_STEPS)

    # 逆運動学により，各位置から関節角度を算出
    thetas_trajectory = np.array([robot.inverse_kinematics(pose, upper=False) for pose in pose_trajectory])

    # アニメーション作成
    robot_animation = RobotAnimation()
    robot_animation.plot_Animation(robot, thetas_trajectory, anime_file_name)


def main():
    """
    メイン処理
    """
    # 2軸ロボットのリンク長
    links = np.array([1.0, 1.0])
    # 2軸ロボットのインスタンスを作成
    robot = Robot2DoF(links)

    # 初期位置と終端位置を定義 [m]
    start_pose = np.array([1.0, 0.5])
    end_pose   = np.array([-1.0, 0.5])

    # パターン1: 関節空間内での軌道生成
    joint_space_trajectory(robot, start_pose, end_pose, linear_interpolation, "robot_forward_anime_linear.gif")
    joint_space_trajectory(robot, start_pose, end_pose, cubic_interpolation,  "robot_forward_anime_cubic.gif")
    joint_space_trajectory(robot, start_pose, end_pose, quintic_interpolation, "robot_forward_anime_quintic.gif")

    # パターン2: 位置空間内での軌道生成
    position_space_trajectory(robot, start_pose, end_pose, linear_interpolation, "robot_inverse_anime_linear.gif")
    position_space_trajectory(robot, start_pose, end_pose, cubic_interpolation,  "robot_inverse_anime_cubic.gif")
    position_space_trajectory(robot, start_pose, end_pose, quintic_interpolation, "robot_inverse_anime_quintic.gif")


if __name__ == "__main__":
    # 本ファイルがメインで呼ばれた時の処理
    main()
