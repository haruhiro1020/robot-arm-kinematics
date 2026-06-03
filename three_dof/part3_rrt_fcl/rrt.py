# 経路生成手法であるRRT (Rapidly-exploring Random Tree) の実装

# ライブラリの読み込み
import numpy as np
import time

# 自作モジュールの読み込み
from constant import *              # 定数
from environment import Robot3DEnv  # 環境


class Tree:
    """
    ツリークラス

    プロパティ
        _nodes(numpy.ndarray): ノード
        _near_node_idx(int): _nodes内の最短ノードを保存している列番号

    メソッド
        public
            nodes(): _nodesプロパティのゲッター
            add_node(): ノードの追加
            get_near_node(): 最短距離のノードを取得
    """
    # 定数の定義

    def __init__(self, near_node_idx):
        """
        コンストラクタ
        """
        # プロパティの初期化
        self._nodes = []
        self._near_node_idx = near_node_idx

    @property
    def nodes(self):
        """
        _nodesプロパティのゲッター
        """
        return self._nodes

    def reset(self):
        """
        データの初期化
        """
        if len(self._nodes) != 0:
            # 何かしらのデータが保存
            del self._nodes
        self._nodes  =[]

    def add_node(self, node):
        """
        ノードの追加

        パラメータ
            node(numpy.ndarray): ノード
        """
        if len(self._nodes) == 0:       # 初回だけ実行
            self._nodes = node
        else:
            self._nodes = np.append(self._nodes, node, axis=0)

    def _chk_node_exist(self):
        """
        ノードが存在するかの確認
        """
        if len(self._nodes) == 0:
            # 存在しない
            raise ValueError("self_nodes is not exist")

    def get_near_node(self, pos):
        """
        最短距離のノードを取得

        パラメータ
            pos(numpy.ndarray): 位置

        戻り値
            min_dist_idx(int): 最短距離のノード番号
        """
        # ノードの存在確認
        self._chk_node_exist()

        # ノードから位置を取得
        nodes_pos  = self._nodes[:, :self._near_node_idx]
        # 差分を計算
        difference = nodes_pos - pos
        # 距離を計算
        distance   = np.linalg.norm(difference, axis=1)
        # 最短距離ノードを取得
        min_dist_idx = np.argmin(distance)

        return min_dist_idx

    def get_near_node_list(self, pos, radius):
        """
        ノードと近傍ノードをリストで取得

        パラメータ
            pos(numpy.ndarray): ノード位置
            radius(float): 半径

        戻り値
            near_node_list(list): 近傍ノードリスト
        """
        # ノードの存在確認
        self._chk_node_exist()

        near_node_list = []
        # ツリー内全ノード位置を取得
        all_node_pos = self._nodes[:, :self._near_node_idx]
        # ノードとツリー内全ノードの差分を計算
        difference   = all_node_pos - pos
        # 差分から距離(ユークリッド距離)を計算
        distance     = np.linalg.norm(difference, axis=1)

        # 距離が一定以内のノードだけを保存
        near_node_list = [idx for idx, dist in enumerate(distance) if dist <= radius]

        return near_node_list


class RRTRobot:
    """
    RRTにロボットを追加したクラス

    プロパティ
        _pathes(numpy.ndarray): 始点から終点までの経路
        _start_tree(Tree): 始点ツリー
        _start_random_tree(Tree): ランダムな位置を保存した始点ツリー
        _strict_min_pos(numpy.ndarray): 探索の最小範囲
        _strict_max_pos(numpy.ndarray): 探索の最大範囲
        _environment(Base2DEnv): 経路生成したい環境
        _debug(bool): デバッグフラグ (グラフやアニメーションでデータを確認したいかどうか)
        _robot(Robot): ロボットクラス
        _moving_value(float): 1回あたりの移動量 [rad] or [m]

    メソッド
        public
            planning(): 経路生成の実装
            pathes(): _pathesプロパティのゲッター
            start_tree(): _treeプロパティのゲッター
            start_random_tree(): _random_treeプロパティのゲッター
            strict_min_pos(): _strict_min_posプロパティのゲッター
            strict_max_pos(): _strict_max_posプロパティのゲッター
            environment(): _environmentプロパティのゲッター

        protected
            _reset(): データの初期化
            _add_node(): ツリーにノードを追加する
            _chk_end_pos_dist(): 終点との距離が一定範囲内であるかの確認
            _fin_planning(): 経路生成の終了処理
            _calc_new_pos(): 最短ノードからランダムな値方向へ新しいノード(位置)を作成
            _strict_planning_pos(): 探索範囲を制限する
            _get_random_pos(): ランダムな位置を取得
            _line_interference(): 2点間の干渉チェック
            _is_interference_pos(): 干渉判定処理
            _save(): データの保存
    """
    # 定数の定義
    # ファイル名の定義
    _FILE_NAME_PATHES = "rrt_pathes.csv"                        # _pathesプロパティを保存するファイル名
    _FILE_NAME_START_TREE = "rrt_start_tree.csv"                # _start_treeプロパティを保存するファイル名
    _FILE_NAME_START_RANDOM_TREE = "rrt_start_random_tree.csv"  # _start_random_treeプロパティを保存するファイル名

    # ツリーの要素番号を定義
    _NODE_NEAR_NODE_IDX = RRT_NEAR_NODE_IDX     # ノード内の最短ノード要素

    # 干渉判定の定義
    _DEVIDED_DISTANCE_POS   = 0.05      # 2点間を分割する際の基準距離 [m]
    _DEVIDED_DISTANCE_JOINT = 0.01      # 2点間を分割する際の基準距離 [rad]
    _INTERFERENCE_MARGIN    = 0.2       # 干渉物とのマージン (最短距離がマージンよりも小さかったら干渉ありとする) [m]

    # 探索に関する定義
    _MOVING_VALUE_JOINT = 0.1         # 1回の移動量 [rad] (ロボットの関節空間)
    _MOVING_VALUE_POS   = 0.2         # 1回の移動量 [m] (ロボットの位置空間)
    _STRICT_PLANNING_ROB_JOINT = 1.5  # 探索範囲の制限 [rad] (ロボットの関節空間)
    _STRICT_PLANNING_ROB_POS   = 0.5  # 探索範囲の制限 [m] (ロボットの位置空間)

    _TIMEOUT_VALUE = 10         # タイムアウト時間 [second]
    _GOAL_SAMPLE_RATE = 0.1     # ランダムな値を取るときに，終点を選択する確率


    def __init__(self):
        """
        コンストラクタ
        """
        self._start_tree = Tree(self._NODE_NEAR_NODE_IDX)
        self._start_random_tree = Tree(self._NODE_NEAR_NODE_IDX)
        self._pathes = []
        self._environment = None
        self._debug = False
        self._robot = None
        self._interpolation = INTERPOLATION.JOINT
        self._moving_value  = self._MOVING_VALUE_JOINT

    @property
    def start_tree(self):
        """
        _start_treeプロパティのゲッター
        """
        return self._start_tree

    @property
    def start_random_tree(self):
        """
        _start_random_treeプロパティのゲッター
        """
        return self._start_random_tree

    @property
    def pathes(self):
        """
        _pathesプロパティのゲッター
        """
        return self._pathes

    @property
    def strict_min_pos(self):
        """
        _strict_min_posプロパティのゲッター
        """
        return self._strict_min_pos

    @property
    def strict_max_pos(self):
        """
        _strict_max_posプロパティのゲッター
        """
        return self._strict_max_pos

    def _add_node_start_tree(self, pos, random_pos, near_node):
        """
        始点ツリーにノードを追加

        パラメータ
            pos(numpy.ndarray): 位置
            random_pos(numpy.ndarray): ランダムな位置
            near_node(int): 最短ノード
        """
        # _start_treeにノードを追加
        node = np.append(pos, near_node).reshape(1, -1)
        self._start_tree.add_node(node)

        # デバッグ有効の時しか，ランダムなツリーにデータを保存したくない (処理時間の短縮)
        if self._debug:
            # _start_random_treeにノードを追加
            random_node = np.append(random_pos, near_node).reshape(1, -1)
            self._start_random_tree.add_node(random_node)

    def _reset(self):
        """
        データの初期化
        """
        self._start_tree.reset()
        self._start_random_tree.reset()

        if len(self._pathes) != 0:
            # 何かしらのデータが保存
            del self._pathes
        self._pathes = []

        if self._environment is None:
            del self._environment
        self._environment = None

        if self._robot is None:
            del self._robot
        self._robot = None
        self._interpolation = INTERPOLATION.JOINT

    def _set_robot(self, robot, interpolation):
        """
        経路生成したいロボット情報を設定

        パラメータ
            robot(Robot): ロボットクラス
            interpolation(int): 補間の種類 (関節補間/位置補間)
        """
        self._robot = robot
        if interpolation == INTERPOLATION.POSITION:
            # 位置空間
            self._interpolation = INTERPOLATION.POSITION
            self._moving_value  = self._MOVING_VALUE_POS
        else:
            # 関節空間
            self._interpolation = INTERPOLATION.JOINT
            self._moving_value  = self._MOVING_VALUE_JOINT

    def planning(self, start_pos, end_pos, environment, robot, interpolation, debug=False):
        """
        経路生成

        パラメータ
            start_pos(list): 経路生成の始点
            end_pos(list): 経路生成の終点
            environment(Base2DEnv): 環境
            robot(Robot): ロボットクラス
            interpolation(int): 補間方法 (関節空間/位置空間)
            debug(bool): デバッグフラグ

        戻り値
            result(bool): True / False = 成功 / 失敗
        """
        # 処理結果
        result = False

        # デバッグフラグの更新
        self._debug = debug
        # データの初期化
        self._reset()

        # 始点と終点の次元数が一致しているかの確認
        start_pos_dim = np.size(start_pos)
        end_pos_dim   = np.size(end_pos)
        if start_pos_dim != end_pos_dim:
            # 次元数が異なるので異常
            raise ValueError(f"start_pos_dim and end_pos_dim are not matched. start_pos_dim is {start_pos_dim}, end_pos_dim is {end_pos_dim}")

        self._dim = start_pos_dim

        # 環境を設定
        self._environment = environment
        # ロボットの設定
        self._set_robot(robot, interpolation)

        # 始点と終点の干渉判定
        if self._interference_pos(start_pos):
            # 干渉ありまたは逆運動学の解が求まらない
            raise ValueError(f"start_pos is interference. please change start_pos")

        if self._interference_pos(end_pos):
            # 干渉ありまたは逆運動学の解が求まらない
            raise ValueError(f"end_pos is interference. please change end_pos")

        # 始点ノードを設定
        self._add_node_start_tree(start_pos, start_pos, INITIAL_NODE_NEAR_NODE)
        # 探索する範囲を設定
        self._strict_planning_pos(start_pos, end_pos)

        # 経路生成は一定時間内に終了させる
        start_time = time.time()

        # 終点までの経路生成が終わるまでループ
        while True:
        # for _ in range(1000):
            # ランダムな値を取得
            random_pos = self._get_random_pos(end_pos)
            # ランダムな値と最短ノードを計算
            near_node = self._start_tree.get_near_node(random_pos)
            # 最短ノードの位置
            near_node_pos = self._start_tree.nodes[near_node, :self._NODE_NEAR_NODE_IDX]
            # 最短ノードからランダムな値方向へ新しいノード(位置)を作成
            new_pos = self._calc_new_pos(random_pos, near_node_pos)

            # 新規ノードと最短ノード間での干渉判定
            is_interference = self._line_interference(new_pos, near_node_pos)
            if not is_interference:
                # 干渉なしのため，ノードを設定する
                self._add_node_start_tree(new_pos, random_pos, near_node)

                # 終点との距離が一定範囲内であるかの確認
                if self._chk_end_pos_dist(new_pos, end_pos):
                    # 一定範囲内のため，経路生成の完了
                    result = True
                    break

            now_time = time.time()
            if now_time - start_time >= self._TIMEOUT_VALUE:
                # タイムアウト
                break

        if result:
            # 経路生成に成功のため，経路生成の終了処理
            self._fin_planning(start_pos, end_pos)
            # ファイルに保存する
            self._save()

        return result

    def _line_interference(self, pos1, pos2):
        """
        2点間の干渉チェック

        パラメータ
            pos1(numpy.ndarray): 位置1/関節1
            pos2(numpy.ndarray): 位置2/関節2

        戻り値
            is_interference(bool): True / False = 干渉あり / 干渉なし
        """
        is_interference = False

        # 差分を計算
        difference = pos2 - pos1
        # 距離を計算
        distance = np.linalg.norm(difference)

        if self._interpolation == INTERPOLATION.POSITION:
            # 位置空間 (分割時の距離を保存)
            devided_dist = self._DEVIDED_DISTANCE_POS
        else:
            # 関節空間 (分割時の距離を保存)
            devided_dist = self._DEVIDED_DISTANCE_JOINT

        # 2点間の分割数を算出
        n_devided = max(int(distance / devided_dist), 1)
        for i in range(n_devided + 1):
            direct_pos  = i / n_devided * difference
            devided_pos = pos1 + direct_pos
            if self._interference_pos(devided_pos):
                # 干渉あり
                is_interference = True
                break

        return is_interference

    def _interference_pos(self, pos):
        """
        干渉判定

        パラメータ
            pos(numpy.ndarray): 位置/関節

        戻り値
            is_interference(bool): True / False = 干渉あり / 干渉なし
        """
        is_interference = False

        if self._interpolation == INTERPOLATION.POSITION:
            # 位置空間の場合は，逆運動学を計算
            try:
                thetas = self._robot.inverse_kinematics(pos)
                # ロボットの位置を更新
                self._robot.update(thetas)
                # 干渉判定
                if self._environment.is_collision_dist(self._robot.manager, margin=self._INTERFERENCE_MARGIN):
                    # 干渉あり
                    is_interference = True

            except Exception as e:
                # 逆運動学の解が得られないから，干渉ありとする
                is_interference = True

        else:
            # 関節空間
            # ロボットの位置を更新
            self._robot.update(pos)
            # 干渉判定
            if self._environment.is_collision_dist(self._robot.manager, margin=self._INTERFERENCE_MARGIN):
                # 干渉あり
                is_interference = True

        return is_interference

    def _chk_end_pos_dist(self, pos, end_pos):
        """
        終点との距離が一定範囲内であるかの確認

        パラメータ
            pos(numpy.ndarray): ノード位置
            end_pos(numpy.ndarray): 経路生成の終点

        戻り値
            is_near(bool): True / False = 一定範囲内である / でない
        """
        is_near = False
        # 距離を計算
        dist = np.linalg.norm(end_pos - pos)
        # 一定範囲内であるかの確認
        if dist <= self._moving_value:
            if not self._line_interference(pos, end_pos):
                # 干渉しない
                is_near = True

        return is_near

    def _fin_planning(self, start_pos, end_pos):
        """
        経路生成の終了処理

        パラメータ
            start_pos(list): 経路生成の始点
            end_pos(list): 経路生成の終点
        """
        # 始点から終点までの経路に関係するノードを選択
        revers_path = end_pos.reshape(1, -1)
        near_node = -1

        while True:
            # 終点から始点方向へノードを取得
            node = self._start_tree.nodes[near_node]
            pos  = node[:self._NODE_NEAR_NODE_IDX].reshape(1, -1)
            # 浮動小数型になっているので，整数型に型変換
            near_node = int(node[self._NODE_NEAR_NODE_IDX])
            revers_path = np.append(revers_path, pos, axis=0)
            if near_node == INITIAL_NODE_NEAR_NODE:
                # 始点ノードまで取得できたため，処理終了
                break

        # 経路が終点からの順番になっているため，始点から終点とする
        self._pathes = revers_path[::-1]
        # ツリーに終点を追加 (要素番号を指定するため -1)
        self._add_node_start_tree(end_pos, end_pos, self._start_tree.nodes.shape[0] - 1)

    def _calc_new_pos(self, random_pos, near_node_pos):
        """
        最短ノードからランダムな値方向へ新しいノード(位置)を作成

        パラメータ
            random_pos(numpy.ndarray): ランダムな位置
            near_node_pos(numpy.ndarray): 最短ノード位置

        戻り値
            new_pos(numpy.ndarray): 新しいノード
        """
        # 方向を計算
        direction = random_pos - near_node_pos
        # 方向の大きさを1にする
        norm_direction = direction / (np.linalg.norm(direction) + EPSILON)
        # 新しいノードを作成
        new_pos = near_node_pos + norm_direction * self._moving_value

        return new_pos

    def _strict_planning_pos(self, start_pos, end_pos):
        """
        探索範囲を制限する

        パラメータ
            start_pos(numpy.ndarray): 始点
            end_pos(numpy.ndarray): 終点
        """
        all_pos = np.array([start_pos, end_pos])
        # 各列の最大/最小値を取得
        min_pos = np.min(all_pos, axis=0)
        max_pos = np.max(all_pos, axis=0)

        if self._interpolation == INTERPOLATION.POSITION:
            # 位置空間の探索
            strict_planning_pos = self._STRICT_PLANNING_ROB_POS
            self._strict_min_pos = min_pos - strict_planning_pos
            self._strict_max_pos = max_pos + strict_planning_pos
        else:
            # 関節空間の探索
            strict_planning_pos = self._STRICT_PLANNING_ROB_JOINT
            self._strict_min_pos = min_pos - strict_planning_pos
            self._strict_max_pos = max_pos + strict_planning_pos
            # self._strict_min_pos = np.ones(self._dim) * np.pi * -1
            # self._strict_max_pos = np.ones(self._dim) * np.pi

        # print(f"self._strict_min_pos = {self._strict_min_pos}")
        # print(f"self._strict_max_pos = {self._strict_max_pos}")

    def _get_random_pos(self, target_pos):
        """
        ランダムな位置を取得

        パラメータ
            target_pos(numpy.ndarray): 目標点

        戻り値
            random_pos(numpy.ndarray): ランダムな位置
        """
        # 乱数を取って，目標点を選択するかランダムを選択するか
        select_goal = np.random.rand()
        if select_goal < self._GOAL_SAMPLE_RATE:
            # 目標点を選択s
            random_pos = target_pos
        else:
            random_pos = np.random.uniform(self._strict_min_pos, self._strict_max_pos)

        return random_pos

    def _save(self):
        """
        生成した経路をファイル保存
        """
        if self._debug:
            np.savetxt(self._FILE_NAME_PATHES,              self._pathes)
            np.savetxt(self._FILE_NAME_START_TREE,          self._start_tree.nodes)
            np.savetxt(self._FILE_NAME_START_RANDOM_TREE,   self._start_random_tree.nodes)
