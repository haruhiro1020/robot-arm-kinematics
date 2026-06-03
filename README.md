# Robot Arm Kinematics — ソースコード概要

ロボットアーム運動学を段階的に学ぶための学習教材です。  
2自由度のロボットアームに対し、順運動学から RRT 経路計画まで体系的に実装されています。

---

## ディレクトリ構成

```
src/
├── two_dof/                         # 2自由度ロボットアーム
│   ├── part1_forward_kinematics/    # 順運動学
│   ├── part2_inverse_kinematics/    # 逆運動学
│   ├── part3_trajectory/            # 軌道生成
│   ├── part4_rrt_fcl/               # RRT 経路計画（FCL 干渉判定）
│   └── part5_diff_inverse_kinematics/ # 微分逆運動学
│
├── three_dof/                       # 3自由度ロボットアーム
│   ├── part1_forward_kinematics/
│   ├── part2_inverse_kinematics/
│   ├── part3_rrt_fcl/
│   └── part4_diff_inverse_kinematics/
│
└── six_dof/                         # 6自由度ロボットアーム
    ├── part1_forward_kinematics/
    ├── part2_inverse_kinematics/
    ├── part3_rrt_fcl/
    └── part4_diff_inverse_kinematics/
```

---

## 使用言語・ライブラリ

| ライブラリ | バージョン | 用途 |
|---|---|---|
| Python | 3.10.9 | メイン言語 |
| NumPy | 1.23.5 | 行列演算・三角関数 |
| Matplotlib | 3.7.0 | 2D/3D 可視化・GIF アニメーション生成 |
| python-fcl | 0.7.0.8 | 干渉判定・最短距離計算 |
| ImageMagick | — | GIF 保存（Matplotlib 経由） |

---

## 主な機能

| Part | 機能 | 概要 |
|---|---|---|
| part1 | 順運動学 | 関節角度 → 手先位置を計算・可視化 |
| part2 | 逆運動学 | 手先位置 → 関節角度を解析的に計算 |
| part3 | 軌道生成 | 直線・3次・5次多項式補間による滑らかな軌道 |
| part4 | RRT 経路計画 | FCL 干渉判定を用いた衝突回避経路の生成 |
| part5 | 微分逆運動学 | ヤコビ行列による反復計算（Levenberg-Marquardt 法対応） |

---

## クラス設計

### Robot クラス階層

```
Robot（抽象基底クラス）
├── Robot2DoF
├── Robot3DoF
└── Robot6DoF
```

### 主なメソッド（Robot 基底クラス）

| メソッド | 説明 |
|---|---|
| `forward_kinematics(thetas)` | 関節角度から手先位置を計算 |
| `inverse_kinematics(pose)` | 手先位置から関節角度を計算 |
| `forward_kinematics_all_link_pos(thetas)` | 全リンク位置を取得 |
| `differential_inverse_kinematics(thetas, target_pos)` | 微分逆運動学で角度更新 |
| `update(thetas)` | FCL オブジェクトの位置を更新 |
| `_calc_homogeneous_matrix(thetas)` | 4×4 同次変換行列を計算 |
| `_jacobian(thetas)` | ヤコビ行列を計算 |

### その他の主要クラス

| クラス | ファイル | 役割 |
|---|---|---|
| `Rotation` / `MyRotation` | `rotation.py` | x/y/z 軸周りの回転行列を生成 |
| `RRTRobot` | `rrt.py` | RRT アルゴリズムによる経路計画 |
| `RobotAnimation` | `animation.py` | Matplotlib アニメーション生成・GIF 保存 |
| `Robot2DEnv` | `environment.py` | FCL 円形障害物の定義・干渉判定 |

---

## 主な計算式

### 2軸ロボット 順運動学

```
P = l1 * [cos(θ1), sin(θ1)] + l2 * [cos(θ1+θ2), sin(θ1+θ2)]
```

### 2軸ロボット 逆運動学

```
cos(θ2) = (px² + py² - l1² - l2²) / (2 * l1 * l2)
θ2 = atan2(sin(θ2), cos(θ2))
θ1 = atan2(sin(θ1), cos(θ1))  ← 行列計算で導出
```

### ヤコビ行列 / 微分逆運動学

```
ΔP = J(θ) × ΔΘ  →  ΔΘ = J⁻¹ × ΔP
```

特異点近傍では Levenberg-Marquardt 法を使用:

```
ΔΘ = (JᵀJ + λI)⁻¹ Jᵀ ΔP
```

---

## 実行方法

各 `main.py` を直接実行します。

```bash
# 例: 2軸ロボットの順運動学
cd src/two_dof/part1_forward_kinematics
python main.py
```

### 出力の種類

| 出力 | 形式 | 説明 |
|---|---|---|
| 静止グラフ | PNG | 関節角度ごとの手先位置プロット |
| アニメーション | GIF | 軌道・経路計画のアニメーション |
| デバッグデータ | CSV | RRT のノード情報など |

---

## ファイル間の依存関係（2軸アームの例）

```
main.py
├── robot.py        ← Robot2DoF（順/逆/微分運動学）
├── constant.py     ← 補間方法・回転軸などの Enum 定義
├── rotation.py     ← 回転行列クラス
├── animation.py    ← GIF アニメーション生成
├── environment.py  ← FCL 障害物環境
└── rrt.py          ← RRT 経路計画（part4 のみ）
```

---
