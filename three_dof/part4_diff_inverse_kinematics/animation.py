# ロボットのアニメーションを実施

# ライブラリの読み込み
import numpy as np      # 数値計算
import matplotlib.pyplot as plt     # 描画用
import matplotlib.animation as ani  # アニメーション用
import matplotlib.patches as patches    # 2次元形状の描画
from mpl_toolkits.mplot3d.art3d import Poly3DCollection     # 3次元形状の描画

# 自作モジュールの読み込み
from constant import *      # 定数
from rotation import MyRotation   # 回転行列


class RRTRobotAnimation:
    """
    RRTによるロボットの経路生成アニメーション

    プロパティ
        _figure: 描画枠
        _axis: 描画内容

    publicメソッド (全てのクラスから参照可能)
        plot_Animation(): アニメーション作成

    protectedメソッド (自クラスまたは子クラスが参照可能)
        _reset2D(): 2次元データのリセット
    """
    # 定数の定義
    _ANIMATION_NAME = "robot_animation.gif"
    _PLOT_NAME      = "robot_plot.gif"
    _STRICT_POS     = 0.2

    def __init__(self):
        """
        コンストラクタ
        """
        self._rot = MyRotation()

    def _reset2D(self):
        """
        2次元データのリセット
        """
        self._figure = plt.Figure()
        self._axis = self._figure.add_subplot(111)

        # X/Y軸に文字を記載
        self._axis.set_xlabel("X")
        self._axis.set_ylabel("Y")

        self._axis.grid()
        self._axis.set_aspect("equal")

    def _plot_circle(self, x, y, radius):
        """
        円の描画

        パラメータ
            x(float): 中心点 (x)
            y(float): 中心点 (y)
            radius(float): 半径
        """
        circle = patches.Circle((x, y), radius, color="gray", alpha=0.5)
        self._axis.add_patch(circle)

    def _plot_rectangle(self, center, width, height, angle):
        """
        長方形の描画

        パラメータ
            center(numpy.ndarray): 中心の座標 (x, y)
            width(float): 幅
            height(float): 高さ
            angle(float): 角度 [deg]
        """
        # 左下隅の座標
        xy = np.array([center[0] - width / 2, center[1] - height / 2])
        rect = patches.Rectangle(xy, width, height, angle=angle, color="gray", alpha=0.5)
        # 長方形を軸に追加
        self._axis.add_patch(rect)

    def _plot_ball(self, center, radius):
        """
        円の描画

        パラメータ
            center(np.ndarray): 中心位置 (x, y, z)
            radius(float): 半径
        """
        self._axis.plot_wireframe(self._x * radius + center[0], self._y * radius + center[1], self._z * radius + center[2], color="gray", alpha=0.5)

    def _plot_cuboid(self, center, x, y, z, rotation):
        """
        直方体の描画

        パラメータ
            center(numpy.ndarray): 中心位置 (x, y, z)
            x(float): 直方体のx軸の長さ
            y(float): 直方体のy軸の長さ
            z(float): 直方体のz軸の長さ
            rotation(numpy.ndarray): 回転行列
        """
        # 直方体の頂点を算出する
        points =  np.array([[center[0] - x / 2, center[1] - y / 2, center[2] - z / 2],
                            [center[0] + x / 2, center[1] - y / 2, center[2] - z / 2],
                            [center[0] + x / 2, center[1] + y / 2, center[2] - z / 2],
                            [center[0] - x / 2, center[1] + y / 2, center[2] - z / 2],
                            [center[0] - x / 2, center[1] - y / 2, center[2] + z / 2],
                            [center[0] + x / 2, center[1] - y / 2, center[2] + z / 2],
                            [center[0] + x / 2, center[1] + y / 2, center[2] + z / 2],
                            [center[0] - x / 2, center[1] + y / 2, center[2] + z / 2]])

        # 頂点の4点から面を算出する
        verts =    [[points[0], points[1], points[2], points[3]],
                    [points[4], points[5], points[6], points[7]],
                    [points[0], points[1], points[5], points[4]],
                    [points[2], points[3], points[7], points[6]],
                    [points[1], points[2], points[6], points[5]],
                    [points[4], points[7], points[3], points[0]]]

        # 直方体の描画
        self._axis.add_collection3d(Poly3DCollection(verts, facecolors='gray', edgecolors='gray', alpha=0.5))

    def _plot_2DAnimation(self, robot, all_link_thetas, environment, anime_file_name=""):
        """
        2次元アニメーションの作成

        パラメータ
            dimention(int): 次元数
            robot(Robot2DoF): ロボットクラス
            all_link_thetas(numpy.ndarray): 全リンクの回転角度
            environment(Robot2DEnv): 経路生成時の環境
            anime_file_name(str): アニメーションのファイル名
        """
        # データをリセットする
        self._reset2D()

        # 全画像を保存する
        imgs = []

        # 手先位置の軌跡を保存
        position_trajectory = np.zeros((all_link_thetas.shape[0], DIMENTION_2D))

        # 始点と終点をプロット
        # 始点位置を取得
        start_pos   = robot.forward_kinematics(all_link_thetas[0])
        start_image = self._axis.scatter(start_pos[0], start_pos[1], color="cyan")
        end_pos   = robot.forward_kinematics(all_link_thetas[-1])
        end_image = self._axis.scatter(end_pos[0], end_pos[1], color="red")

        # 軌道生成
        for i, thetas in enumerate(all_link_thetas):
            path_images = []
            # 順運動学により，全リンク (ベースリンク, リンク1，手先位置) の位置を計算
            all_link_pos = robot.forward_kinematics_all_link_pos(thetas)
            # 線プロット
            image = self._axis.plot(all_link_pos[:, 0], all_link_pos[:, 1], color="blue")
            path_images.extend(image)
            # 点プロット
            image = self._axis.scatter(all_link_pos[:, 0], all_link_pos[:, 1], color="black", alpha=0.5)
            path_images.extend([image])

            # 手先位置を保存
            position_trajectory[i] = all_link_pos[-1]
            # 手先位置の軌跡をプロット
            image = self._axis.plot(position_trajectory[:i + 1, 0], position_trajectory[:i + 1, 1], color="lime")
            path_images.extend(image)

            # 始点と終点の画像を保存
            path_images.extend([start_image])
            path_images.extend([end_image])

            # 画像を1枚にまとめて保存
            imgs.append(path_images)

        # アニメーション作成
        animation = ani.ArtistAnimation(self._figure, imgs)
        if anime_file_name:
            # ファイル名が存在する
            animation.save(anime_file_name, writer='imagemagick')
        else:
            # ファイル名が存在しない
            animation.save(self._ANIMATION_NAME, writer='imagemagick')
        plt.show()

    def _reset3D(self):
        """
        3次元データのリセット
        """
        self._figure = plt.figure()
        self._axis = self._figure.add_subplot(111, projection="3d")

        # 0 ~ 2piまでの範囲とする
        theta_1_0 = np.linspace(0, np.pi * 2, 20)
        theta_2_0 = np.linspace(0, np.pi * 2, 20)
        theta_1, theta_2 = np.meshgrid(theta_1_0, theta_2_0)

        # x, y, zの曲座標表示 (中心点が原点である半径1の球)
        self._x = np.cos(theta_2) * np.sin(theta_1)
        self._y = np.sin(theta_2) * np.sin(theta_1)
        self._z = np.cos(theta_1)

    def _set_3DAxis(self, robot):
        """
        3次元データのラベルや範囲を設定
        """
        # X/Y/Z軸に文字を記載
        self._axis.set_xlabel("X")
        self._axis.set_ylabel("Y")
        self._axis.set_zlabel("Z")

        self._axis.grid()
        self._axis.set_aspect("equal")

    def _update_3Ddata(self, i, robot, all_link_thetas, all_link_poses, environment):
        """
        3D(3次元)各データの更新

        パラメータ
            i(int): フレーム番号
            robot(Robot3DoF): ロボットクラス
            all_link_thetas(numpy.ndarray): 始点から終点までの全角度
            all_link_poses(numpy.ndarray): 始点から終点までの全位置
            environment(Robot3DEnv): 経路生成時の環境
        """
        # 以前のプロットをクリアする
        self._axis.clear()
        self._set_3DAxis(robot)

        # 始点と終点をプロット
        # 始点位置を取得
        start_pos = robot.forward_kinematics(all_link_thetas[0])
        self._axis.scatter(start_pos[0], start_pos[1], start_pos[2], color="cyan")
        end_pos   = robot.forward_kinematics(all_link_thetas[-1])
        self._axis.scatter(end_pos[0], end_pos[1], end_pos[2], color="red")

        # 順運動学により，全リンク (ベースリンク, リンク1，手先位置) の位置を計算
        all_link_pos = robot.forward_kinematics_all_link_pos(all_link_thetas[i])
        # 線プロット
        self._axis.plot(all_link_pos[:, 0], all_link_pos[:, 1], all_link_pos[:, 2], color="blue")
        # 点プロット
        self._axis.scatter(all_link_pos[:, 0], all_link_pos[:, 1], all_link_pos[:, 2], color="black", alpha=0.5)
        # 手先位置の軌跡をプロット
        self._axis.plot(all_link_poses[:i + 1, 0], all_link_poses[:i + 1, 1], all_link_poses[:i + 1, 2], color="lime")

    def _plot_3DAnimation(self, robot, all_link_thetas, environment, anime_file_name):
        """
        3次元アニメーションの作成

        パラメータ
            robot(Robot3DoF): ロボットクラス
            all_link_thetas(numpy.ndarray): 全リンクの回転角度
            environment(Robot3DEnv): 経路生成時の環境
            anime_file_name(str): アニメーションのファイル名
        """
        # データをリセットする
        self._reset3D()

        # 全位置を計算する
        all_link_poses = np.zeros((all_link_thetas.shape[0], DIMENTION_3D))
        for i, thetas in enumerate(all_link_thetas):
            # 順運動学による位置の計算
            poses = robot.forward_kinematics(thetas)
            all_link_poses[i] = poses

        # アニメーションのフレーム数
        n_frame = all_link_thetas.shape[0]
        animation = ani.FuncAnimation(self._figure, self._update_3Ddata, fargs=(robot, all_link_thetas, all_link_poses, environment), interval=100, frames=n_frame)

        # アニメーション
        if anime_file_name:
            animation.save(anime_file_name, writer="imagemagick")
        else:
            animation.save(self._ANIMATION_NAME, writer="imagemagick")
        plt.show()

    def plot_Animation(self, dimention, robot, all_link_thetas, environment, anime_file_name=""):
        """
        アニメーション作成

        パラメータ
            dimention(int): 次元数
            robot(Robot2DoF): ロボットクラス
            all_link_thetas(numpy.ndarray): 全リンクの回転角度
            environment(Robot2DEnv): 経路生成時の環境
            anime_file_name(str): アニメーションのファイル名
        """
        # 引数の確認
        if all_link_thetas.size == 0:
            raise ValueError(f"all_link_thetas's size is abnormal. all_link_thetas's size is {all_link_thetas.size}")

        if dimention == DIMENTION_2D:
            # 2次元アニメーション
            self._plot_2DAnimation(robot, all_link_thetas, environment, anime_file_name)
        elif dimention == DIMENTION_3D:
            # 3次元アニメーション
            self._plot_3DAnimation(robot, all_link_thetas, environment, anime_file_name)
        else:
            # 異常
            raise ValueError(f"dimention is abnormal. dimention is {dimention}")
