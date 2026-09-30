#---PCAを用いて第一主成分軸の動きを直線で追うGif画像の出力---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import japanize_matplotlib
from pathlib import Path

print("ライブラリのインポートが完了しました。")

# --- 1. 定数と設定 ---
WINDOW_SIZE = 20
PCA_AXIS_HALF_LENGTH_RATIO = 0.8
FILE_PATH = 'PC1.csv'
OUTPUT_GIF_DIR = Path(r"C:\書類\宇都宮大学大学院\藤井研究室_研究\yudai_データ分析\Python & CSVファイル\GIF画像")
OUTPUT_GIF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_GIF_PATH = OUTPUT_GIF_DIR / f'{Path(FILE_PATH).stem}-{WINDOW_SIZE}.gif'

# --- 2. データの読み込み ---
try:
    df = pd.read_csv(FILE_PATH)
    print(f"'{FILE_PATH}' を読み込みました。データ数: {len(df)}")
except FileNotFoundError:
    print(f"エラー: ファイル '{FILE_PATH}' が見つかりません。")
    exit()

# accとGameRoをフィルタリングしてコピー
acc_df = df[df['Sensor'] == 'Lacc'].copy() #dfで返したいため.copy()
gamerot_df = df[df['Sensor'] == 'GameRo'].copy()

# accを基準として、最も近い時刻のGameRoの値を結合する
# GameRoのデータはTimestamp, X, Y, Z, Wのみを選択してマージする
merged_df = pd.merge_asof(
    acc_df.sort_values('Timestamp'),
    gamerot_df.sort_values('Timestamp'),
    on='Timestamp',direction='backward',suffixes=('_acc', '_ro'))

# 結合後の列名を分かりやすく設定
# 加速度
acc_x = merged_df['X_acc']
acc_y = merged_df['Y_acc']
acc_z = merged_df['Z_acc']
# クォータニオン
gx = merged_df['X_ro']
gy = merged_df['Y_ro']
gz = merged_df['Z_ro']
gw = merged_df['W_ro']

gx0, gy0, gz0, gw0 = gamerot_df.iloc[0]['X'], gamerot_df.iloc[0]['Y'], gamerot_df.iloc[0]['Z'], gamerot_df.iloc[0]['W']
    
# 共役な四元数を求める
def kyoyaku(gx, gy, gz, gw):
    return (-gx, -gy, -gz, gw)

# 1行目の四元数の共役を取得
Gx0, Gy0, Gz0, Gw0 = kyoyaku(gx0, gy0, gz0, gw0)

# --- 相対クォータニオンの計算（q_relative = q * q_0_conjugate）---
# 相対化のための四元数積
gwc = gw*Gw0 - gx*Gx0 - gy*Gy0 - gz*Gz0
gxc = gw*Gx0 + gx*Gw0 - gy*Gz0 + gz*Gy0
gyc = gw*Gy0 + gx*Gz0 + gy*Gw0 - gz*Gx0
gzc = gw*Gz0 - gx*Gy0 + gy*Gx0 + gz*Gw0

# --- 回転行列の適用（加速度ベクトルの回転 P' = q_relative * P * q_relative_conjugate）---
# 相対化した回転行列の計算
Ax = (2*gwc*gwc + 2*gxc*gxc - 1)*acc_x + (2*gxc*gyc - 2*gzc*gwc)*acc_y + (2*gxc*gzc + 2*gyc*gwc)*acc_z
Ay = (2*gxc*gyc + 2*gzc*gwc)*acc_x + (2*gwc*gwc + 2*gyc*gyc - 1)*acc_y + (2*gyc*gzc - 2*gxc*gwc)*acc_z
Az = (2*gxc*gzc - 2*gyc*gwc)*acc_x + (2*gyc*gzc + 2*gxc*gwc)*acc_y + (2*gwc*gwc + 2*gzc*gzc - 1)*acc_z

merged_df['X_rotated'] = Ax
merged_df['Y_rotated'] = Ay
merged_df['Z_rotated'] = Az

df = merged_df.dropna(subset=['X_rotated','Y_rotated','Z_rotated']).reset_index(drop=True)

time_data = df['Timestamp'].values

#最初の時刻
time_start_ns = time_data[0]

#最後の時刻
time_end_ns = time_data[-1]

#総時間(s)
total_duration_sec = (time_end_ns - time_start_ns) / 1000000000

#総フレーム数
total_frame = len(df)

#fpsを計算.1秒当たりのフレーム数
real_time_fps = total_frame / total_duration_sec

#fを書くことで文字列ではなく変数として認識
print(f"計測時間: {total_duration_sec:.2f} 秒")
print(f"リアルタイムFPS: {real_time_fps:.2f} ")

x_data = df['X_rotated'].values
y_data = df['Y_rotated'].values
z_data = df['Z_rotated'].values

#グラフの初期設定と軸範囲の固定
absolute_acceleration = np.concatenate((np.abs(x_data), np.abs(y_data)))
detail_limit = max(np.percentile(absolute_acceleration, 99.5) * 1.1, 1.0)
overview_limit = np.max(absolute_acceleration) * 1.1
if overview_limit == 0:
    overview_limit = 1.0

fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xlim(-detail_limit, detail_limit)
ax.set_ylim(-detail_limit, detail_limit)
ax.set_aspect('equal', adjustable='box')
ax.grid(True)

truth_arrow = ax.quiver(
    0, 0,          # 始点 (x, y)
    0, 5,          # ベクトル (u, v)
    angles='xy',
    scale_units='xy',
    scale=1,
    color='black',
    alpha=0.7,
    width=0.005,
    headwidth=5,
    headlength=7,
    headaxislength=6,
    zorder=10
)

#アニメーション要素の初期化
points, = ax.plot(
    [],
    [],
    marker='o',
    linestyle='None',
    markersize=4,
    alpha=1.0
)

pca_line, = ax.plot(
    [],
    [],
    color='red',
    linestyle='-',
    linewidth=3,
    alpha=0.7
)

fig.tight_layout()

#主成分を1つだけ求める(何次元に落とし込むか)
pca = PCA(n_components=1)

print("グラフの初期設定が完了しました。")

x_min, x_max = x_data.min(), x_data.max()
y_min, y_max = y_data.min(), y_data.max()

x_min_index = np.argmin(x_data)
x_max_index = np.argmax(x_data)
y_min_index = np.argmin(y_data)
y_max_index = np.argmax(y_data)

x_min_timestamp_ns = time_data[x_min_index]
x_max_timestamp_ns = time_data[x_max_index]
y_min_timestamp_ns = time_data[y_min_index]
y_max_timestamp_ns = time_data[y_max_index]

x_min_time_sec = (x_min_timestamp_ns - time_data[0]) / 1_000_000_000
x_max_time_sec = (x_max_timestamp_ns - time_data[0]) / 1_000_000_000
y_min_time_sec = (y_min_timestamp_ns - time_data[0]) / 1_000_000_000
y_max_time_sec = (y_max_timestamp_ns - time_data[0]) / 1_000_000_000


def pca_axis_endpoints(unit_vector, limit):
    half_axis_vector = unit_vector * (limit * PCA_AXIS_HALF_LENGTH_RATIO)
    return -half_axis_vector, half_axis_vector


#更新用関数
def update(frame):
    
    start_index = max(0 , (frame - (WINDOW_SIZE - 1)))
    end_index = frame + 1

    x_window = x_data[start_index:end_index]
    y_window = y_data[start_index:end_index]

    points.set_data(x_window, y_window)

    if end_index - start_index < WINDOW_SIZE:
        pca_line.set_data([], [])

        return points, pca_line

    #データ数行，2列の2次元配列
    data_window = np.column_stack((x_window, y_window))

    #引数は2次元配列
    pca.fit(data_window)

    #第一主成分の方向ベクトルの配列[PCA1_x, PCA1_y]を格納
    pca_vector = pca.components_[0]
    
    #pca_vectorを正規化
    norm_vector = pca_vector / np.linalg.norm(pca_vector)

    start, end = pca_axis_endpoints(norm_vector, detail_limit)

    pca_line.set_data(
    [start[0], end[0]],
    [start[1], end[1]]
    )

    return points, pca_line


#アニメーションの生成
#必要引数:fig, func(各フレームごとに呼ばれる更新関数), frames(フレーム数)
animation = FuncAnimation(fig, update, frames=len(df), blit=False)


#GIFとして保存
try:
    print("GIFの保存を開始します")
    animation.save(OUTPUT_GIF_PATH, writer='pillow', fps= int(round(real_time_fps)))
    
    print(f"'{OUTPUT_GIF_PATH}' として保存が完了しました。")

except Exception as e:
    print(f"GIFの保存中にエラーが発生しました: {e}")
    if "No module named 'sklearn'" in str(e):
        print("エラー: scikit-learn (sklearn) がインストールされていないようです。")
