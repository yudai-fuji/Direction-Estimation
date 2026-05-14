# ---端末の姿勢推定を行う---

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import pandas as pd
import japanize_matplotlib

# CSVファイルからデータを読み込む
f = 'EPSON.csv'
df = pd.read_csv(f)

# GameRoだけ抽出して時刻順に整列
df = df[df['Sensor'] == 'GameRo'].copy()
df = df.sort_values('Timestamp').reset_index(drop=True)

# 元のtimestamps（ns）
timestamps = df['Timestamp'].values

# (サンプル数,4)の2次元配列
rotation_vectors = df[['X', 'Y', 'Z', 'W']].values

# 大きさを返す．axis=(0,1)=(列，行)で計算．今回は行ごとに大きさを計算する．keepdimsは配列の次元を残すかどうか(Trueは残す)
norm = np.linalg.norm(rotation_vectors, axis=1, keepdims=True)

# normが0ならnorm = 1.0, 0でないならnorm = norm
rotation_vectors = rotation_vectors / np.where(norm == 0, 1.0, norm)

# 回転そのものを表すもの(回転行列ではない)をサンプル数の数だけ作成
rots = R.from_quat(rotation_vectors)

# ★追加：初期姿勢 rot0 の逆回転を左から掛けて.inv()、初期を恒等回転にする（frame=0で軸一致）
rot0 = rots[0]
rots_rel = rot0.inv()*rots  

# タイムスタンプを0から始まる0.2刻みの新しい値に置き換え
new_timestamps = np.arange(0, len(timestamps) * 0.2, 0.2)

# 配列の長さを調整（生成した配列がtimestampを超える場合）
if len(new_timestamps) > len(timestamps):
    new_timestamps = new_timestamps[:len(timestamps)]
timestamps = new_timestamps

# 3Dプロットの設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

origin = np.array([0, 0, 0])  # ★追加：世界座標の原点(0,0,0)を固定
x = np.array([1, 0, 0])       # ★追加：世界座標のX軸単位ベクトル
y = np.array([0, 1, 0])       # ★追加：世界座標のY軸単位ベクトル
z = np.array([0, 0, 1])       # ★追加：世界座標のZ軸単位ベクトル

# 世界座標を「実線 -」で描画
world_x, = ax.plot([origin[0], x[0]], [origin[1], x[1]], [origin[2], x[2]],
                   linestyle='-', color='red', alpha=0.6, label='-Wx')
world_y, = ax.plot([origin[0], y[0]], [origin[1], y[1]], [origin[2], y[2]],
                   linestyle='-', color='blue', alpha=0.6, label='-Wy')
world_z, = ax.plot([origin[0], z[0]], [origin[1], z[1]], [origin[2], z[2]],
                   linestyle='-', color='green', alpha=0.6, label='-Wz')


# デバイス座標を「破線 --」で描画
# 初期設定
line_x, = ax.plot([], [], [], linestyle='--', color='r', label='--Dx')
line_y, = ax.plot([], [], [], linestyle='--', color='b', label='--Dy')
line_z, = ax.plot([], [], [], linestyle='--', color='g', label='--Dz')

"""
line_x, = ax.plot([], [], [], linestyle='--', color='black', label='--Dx',marker='o')
line_y, = ax.plot([], [], [], linestyle='--', color='black', label='--Dy',marker='o')
line_z, = ax.plot([], [], [], linestyle='--', color='black', label='--Dz',marker='o')
"""

ax.legend(loc='upper left')

# アニメーションの初期化関数
def init():
    line_x.set_data([], [])
    line_x.set_3d_properties([])
    line_y.set_data([], [])
    line_y.set_3d_properties([])
    line_z.set_data([], [])
    line_z.set_3d_properties([])
    return line_x, line_y, line_z

# アニメーションのアップデート関数
def update(frame):
    if frame >= len(rots_rel):
        return line_x, line_y, line_z
    
    R_matrix = rots_rel[frame].as_matrix()

    #回転後のベクトルを計算
    x_rot = R_matrix @ x
    y_rot = R_matrix @ y
    z_rot = R_matrix @ z

    #set_dataはx・yベクトルの始点,終点をセット、3DはZ軸も考慮
    line_x.set_data([origin[0], x_rot[0]], [origin[1], x_rot[1]])
    line_x.set_3d_properties([origin[2], x_rot[2]])
    
    line_y.set_data([origin[0], y_rot[0]], [origin[1], y_rot[1]])
    line_y.set_3d_properties([origin[2], y_rot[2]])
    
    line_z.set_data([origin[0], z_rot[0]], [origin[1], z_rot[1]])
    line_z.set_3d_properties([origin[2], z_rot[2]])

    return line_x, line_y, line_z

print('Gif画像を生成しています...')

# ---アニメーションの設定---
# fig:アニメーションを描くFigure
# func:各フレームで呼ばれる更新関数
# frames:フレームの数
# init_func:アニメーション開始前に1回だけ呼ばれる初期化関数
# blit:再描画を最小限にして高速化するか
# interval:フレーム感覚[ms]
ani = FuncAnimation(fig, update, frames=len(timestamps), init_func=init, blit=True, interval=20)

#アニメーションをgif画像として保存
ani.save(f.replace(".csv", "") + "-Pose.gif", writer='pillow')
print('Gif画像を生成しました')
plt.clf()
plt.close()
