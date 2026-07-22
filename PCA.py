#---PCAを用いて第一主成分の動きを矢印ベクトルで追うGif画像の出力---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import japanize_matplotlib

print("ライブラリのインポートが完了しました。")

# --- 1. 定数と設定 ---
WINDOW_SIZE = 40
ARROW_LENGTH = 1
FILE_PATH = r'260630香川共同研究/KL2.csv'
OUTPUT_GIF_PATH = FILE_PATH.replace('.csv','') + '-PCA.gif'

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
ax = merged_df['X_acc']
ay = merged_df['Y_acc']
az = merged_df['Z_acc']
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
Ax = (2*gwc*gwc + 2*gxc*gxc - 1)*ax + (2*gxc*gyc - 2*gzc*gwc)*ay + (2*gxc*gzc + 2*gyc*gwc)*az
Ay = (2*gxc*gyc + 2*gzc*gwc)*ax + (2*gwc*gwc + 2*gyc*gyc - 1)*ay + (2*gyc*gzc - 2*gxc*gwc)*az
Az = (2*gxc*gzc - 2*gyc*gwc)*ax + (2*gyc*gzc + 2*gxc*gwc)*ay + (2*gwc*gwc + 2*gzc*gzc - 1)*az

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

fig, (detail_ax, overview_ax) = plt.subplots(1, 2, figsize=(16, 8))

for plot_ax, plot_limit, title in (
        (detail_ax, detail_limit, '通常範囲（99.5パーセンタイル）'),
        (overview_ax, overview_limit, '全体範囲（外れ値を含む）')):
    plot_ax.set_xlim(-plot_limit, plot_limit)
    plot_ax.set_ylim(-plot_limit, plot_limit)
    plot_ax.set_aspect('equal', adjustable='box')
    plot_ax.set_xlabel('X方向加速度')
    plot_ax.set_ylabel('Y方向加速度')
    plot_ax.set_title(title)
    plot_ax.grid(True)

#アニメーション要素の初期化
detail_points, = detail_ax.plot(
    [], [], marker='o', linestyle='None', markersize=2, alpha=0.7)
overview_points, = overview_ax.plot(
    [], [], marker='o', linestyle='None', markersize=2, alpha=0.7)

#ベクトル(X,Y,U,V):X,Y(始点):U,V(成分)
detail_pca_quiver = detail_ax.quiver(
    0, 0, 0, 0, color='red', scale=1, scale_units='xy', angles='xy')
overview_pca_quiver = overview_ax.quiver(
    0, 0, 0, 0, color='red', scale=1, scale_units='xy', angles='xy')

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

print(f"xの最小値は {x_min:.6f} で，時刻は {x_min_time_sec:.6f} 秒です。")
print(f"xの最大値は {x_max:.6f} で，時刻は {x_max_time_sec:.6f} 秒です。")
print(f"yの最小値は {y_min:.6f} で，時刻は {y_min_time_sec:.6f} 秒です。")
print(f"yの最大値は {y_max:.6f} で，時刻は {y_max_time_sec:.6f} 秒です。")

#更新用関数
def update(frame):
    
    start_index = max(0 , (frame - (WINDOW_SIZE - 1)))
    end_index = frame + 1

    x_window = x_data[start_index:end_index]
    y_window = y_data[start_index:end_index]

    detail_points.set_data(x_window, y_window)
    overview_points.set_data(x_window, y_window)
    
    if end_index - start_index < WINDOW_SIZE:
        #必要引数:U(x成分),V(y成分)
        detail_pca_quiver.set_UVC(0, 0)
        overview_pca_quiver.set_UVC(0, 0)

        return (detail_points, overview_points,
                detail_pca_quiver, overview_pca_quiver)

    #データ数行，2列の2次元配列
    data_window = np.column_stack((x_window, y_window))

    #引数は2次元配列
    pca.fit(data_window)

    #第一主成分の方向ベクトルの配列[PCA1_x, PCA1_y]を格納
    pca_vector = pca.components_[0]
    
    #pca_vectorを正規化
    norm_vector = pca_vector / np.linalg.norm(pca_vector)

    #大きさを設定した値に変更
    scaled_vector = norm_vector * ARROW_LENGTH

    #pca_vectorに再度方向ベクトルを代入
    detail_pca_quiver.set_UVC(scaled_vector[0], scaled_vector[1])
    overview_pca_quiver.set_UVC(scaled_vector[0], scaled_vector[1])

    return (detail_points, overview_points,
            detail_pca_quiver, overview_pca_quiver)


#アニメーションの生成
#必要引数:fig, func(各フレームごとに呼ばれる更新関数), frames(フレーム数)
animation = FuncAnimation(fig, update, frames=len(x_data), blit=False)

#GIFとして保存
try:
    print("GIFの保存を開始します")
    animation.save(OUTPUT_GIF_PATH, writer='pillow', fps= int(round(real_time_fps)))
    
    print(f"'{OUTPUT_GIF_PATH}' として保存が完了しました。")

except Exception as e:
    print(f"GIFの保存中にエラーが発生しました: {e}")
    if "No module named 'sklearn'" in str(e):
        print("エラー: scikit-learn (sklearn) がインストールされていないようです。")
