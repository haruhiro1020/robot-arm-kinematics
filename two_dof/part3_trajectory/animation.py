# ロボットのアニメーションを実施

# ライブラリの読み込み
import numpy as np      # 数値計算
import matplotlib.pyplot as plt     # 描画用
import matplotlib.animation as ani  # アニメーション用

# 自作モジュールの読み込み
from constant import *      # 定数


class RobotAnimation:
    """
    ロボットのアニメーション作成

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

    def __init__(self):
        """
        コンストラクタ
        """
        pass

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

    def plot_Animation(self, robot, all_link_thetas, anime_file_name=""):
        """
        アニメーション作成

        パラメータ
            robot(Robot2DoF): ロボットクラス
            all_link_thetas(numpy.ndarray): 全リンクの回転角度
            anime_file_name(str): アニメーションのファイル名
        """
        # 引数の確認
        if all_link_thetas.size == 0:
            raise ValueError(f"all_link_thetas's size is abnormal. all_link_thetas's size is {all_link_thetas.size}")

        # データをリセットする
        self._reset2D()

        # 全画像を保存する
        imgs = []

        # 手先位置の軌跡を保存
        position_trajectory = np.zeros((all_link_thetas.shape[0], DIMENTION_2D))

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
