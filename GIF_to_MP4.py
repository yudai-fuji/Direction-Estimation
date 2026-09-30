# --- PCAを用いて第一主成分軸の動きを追うMP4動画の出力 ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.decomposition import PCA
import japanize_matplotlib
from pathlib import Path
import imageio_ffmpeg

print("ライブラリのインポートが完了しました。")


# =========================================================
# 1. 定数と設定
# =========================================================

# PCAを適用する窓幅
WINDOW_SIZE = 50

# PCA軸の長さ
PCA_AXIS_HALF_LENGTH_RATIO = 0.8

# 入力CSV
FILE_PATH = 'PC2.csv'

# MP4出力先
OUTPUT_MP4_DIR = Path(
    r"C:\書類\宇都宮大学大学院\藤井研究室_研究\yudai_データ分析\Python & CSVファイル\MP4画像"
)

# フォルダが存在しなければ作成
OUTPUT_MP4_DIR.mkdir(parents=True, exist_ok=True)

# 出力ファイル名
# 例：PC2-30.mp4
OUTPUT_MP4_PATH = OUTPUT_MP4_DIR / f'{Path(FILE_PATH).stem}-{WINDOW_SIZE}.mp4'


# =========================================================
# 2. ffmpegの設定
# =========================================================

# imageio-ffmpegが持っているffmpegをMatplotlibに使用させる
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

print(f"ffmpeg: {ffmpeg_path}")


# =========================================================
# 3. CSV読み込み
# =========================================================

try:
    df = pd.read_csv(FILE_PATH)
    print(f"'{FILE_PATH}' を読み込みました。データ数: {len(df)}")
except FileNotFoundError:
    print(f"エラー: ファイル '{FILE_PATH}' が見つかりません。")
    exit()


# =========================================================
# 4. センサデータを分離
# =========================================================

# 線形加速度
acc_df = df[df['Sensor'] == 'Lacc'].copy()

# Game Rotation Vector
gamerot_df = df[df['Sensor'] == 'GameRo'].copy()

# データが存在するか確認
if len(acc_df) == 0:
    raise ValueError("Laccデータが存在しません。")

if len(gamerot_df) == 0:
    raise ValueError("GameRoデータが存在しません。")


# =========================================================
# 5. LaccとGameRoを時刻同期
# =========================================================

merged_df = pd.merge_asof(
    acc_df.sort_values('Timestamp'),
    gamerot_df.sort_values('Timestamp'),
    on='Timestamp',
    direction='backward',
    suffixes=('_acc', '_ro')
)


# =========================================================
# 6. 加速度
# =========================================================

acc_x = merged_df['X_acc']
acc_y = merged_df['Y_acc']
acc_z = merged_df['Z_acc']


# =========================================================
# 7. クォータニオン
# =========================================================

gx = merged_df['X_ro']
gy = merged_df['Y_ro']
gz = merged_df['Z_ro']
gw = merged_df['W_ro']

# 初期姿勢
gx0 = gamerot_df.iloc[0]['X']
gy0 = gamerot_df.iloc[0]['Y']
gz0 = gamerot_df.iloc[0]['Z']
gw0 = gamerot_df.iloc[0]['W']


# =========================================================
# 8. 共役四元数
# =========================================================

def kyoyaku(gx, gy, gz, gw):
    return (-gx, -gy, -gz, gw)


Gx0, Gy0, Gz0, Gw0 = kyoyaku(gx0, gy0, gz0, gw0)


# =========================================================
# 9. 相対クォータニオン
# q_relative = q * q0_conjugate
# =========================================================

gwc = gw * Gw0 - gx * Gx0 - gy * Gy0 - gz * Gz0
gxc = gw * Gx0 + gx * Gw0 - gy * Gz0 + gz * Gy0
gyc = gw * Gy0 + gx * Gz0 + gy * Gw0 - gz * Gx0
gzc = gw * Gz0 - gx * Gy0 + gy * Gx0 + gz * Gw0


# =========================================================
# 10. 回転行列による加速度の回転
# =========================================================

Ax = (
    (2 * gwc * gwc + 2 * gxc * gxc - 1) * acc_x
    + (2 * gxc * gyc - 2 * gzc * gwc) * acc_y
    + (2 * gxc * gzc + 2 * gyc * gwc) * acc_z
)

Ay = (
    (2 * gxc * gyc + 2 * gzc * gwc) * acc_x
    + (2 * gwc * gwc + 2 * gyc * gyc - 1) * acc_y
    + (2 * gyc * gzc - 2 * gxc * gwc) * acc_z
)

Az = (
    (2 * gxc * gzc - 2 * gyc * gwc) * acc_x
    + (2 * gyc * gzc + 2 * gxc * gwc) * acc_y
    + (2 * gwc * gwc + 2 * gzc * gzc - 1) * acc_z
)

merged_df['X_rotated'] = Ax
merged_df['Y_rotated'] = Ay
merged_df['Z_rotated'] = Az

# 欠損除外
df = merged_df.dropna(
    subset=['X_rotated', 'Y_rotated', 'Z_rotated']
).reset_index(drop=True)

if len(df) == 0:
    raise ValueError("回転後の有効データがありません。")


# =========================================================
# 11. 時刻情報
# =========================================================

time_data = df['Timestamp'].to_numpy()
time_start_ns = time_data[0]
time_end_ns = time_data[-1]

# 計測時間 [s]
total_duration_sec = (time_end_ns - time_start_ns) / 1_000_000_000

# 総フレーム数
total_frame = len(df)

if total_duration_sec <= 0:
    raise ValueError("計測時間が0以下です。")

# 実際のサンプリング周波数
real_time_fps = total_frame / total_duration_sec

print(f"計測時間: {total_duration_sec:.2f} 秒")
print(f"データ数: {total_frame}")
print(f"リアルタイムFPS: {real_time_fps:.2f}")


# =========================================================
# 12. 回転後加速度
# =========================================================

x_data = df['X_rotated'].to_numpy()
y_data = df['Y_rotated'].to_numpy()
z_data = df['Z_rotated'].to_numpy()


# =========================================================
# 13. グラフ表示範囲
# =========================================================

absolute_acceleration = np.concatenate((np.abs(x_data), np.abs(y_data)))

# 99.5パーセンタイルを使用
# 真値ベクトルがy=5まであるので最低5.5を確保
detail_limit = max(np.percentile(absolute_acceleration, 99.5) * 1.1, 5.5)


# =========================================================
# 14. グラフ作成
# =========================================================

fig, plot_ax = plt.subplots(figsize=(8, 8))

plot_ax.set_xlim(-detail_limit, detail_limit)
plot_ax.set_ylim(-detail_limit, detail_limit)
plot_ax.set_aspect('equal', adjustable='box')

# グリッド
plot_ax.grid(True, alpha=0.6)

# X方向加速度、Y方向加速度などの
# 軸ラベルは付けない


# =========================================================
# 15. 真値ベクトル
# 原点 → (0,5)
# =========================================================

truth_arrow = plot_ax.quiver(
    0,
    0,
    0,
    5,
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


# =========================================================
# 16. 加速度サンプル
# =========================================================

points, = plot_ax.plot(
    [],
    [],
    marker='o',
    linestyle='None',
    markersize=4,
    color='tab:blue',
    alpha=1.0,
    zorder=5
)


# =========================================================
# 17. 第一主成分軸
# =========================================================

pca_line, = plot_ax.plot(
    [],
    [],
    color='red',
    linestyle='-',
    linewidth=3,
    alpha=0.7,
    zorder=8
)

fig.tight_layout()


# =========================================================
# 18. PCA
# =========================================================

pca = PCA(n_components=1)
print("グラフの初期設定が完了しました。")


# =========================================================
# 19. 第一主成分軸の長さ
# =========================================================

def pca_axis_endpoints(unit_vector, limit):
    half_axis_vector = unit_vector * limit * PCA_AXIS_HALF_LENGTH_RATIO
    return (-half_axis_vector, half_axis_vector)


# =========================================================
# 20. 各フレーム更新処理
# =========================================================

def update(frame):
    # 現在時刻からWINDOW_SIZE分を使用
    start_index = max(0, frame - (WINDOW_SIZE - 1))
    end_index = frame + 1

    # 適用窓内データ
    x_window = x_data[start_index:end_index]
    y_window = y_data[start_index:end_index]

    # 青点を更新
    points.set_data(x_window, y_window)

    # WINDOW_SIZEに達するまでは
    # PCA軸を描画しない
    if end_index - start_index < WINDOW_SIZE:
        pca_line.set_data([], [])
        return (points, pca_line)

    # -----------------------------------------
    # PCA入力データ
    # 行：各サンプル
    # 列：X, Y
    # -----------------------------------------
    data_window = np.column_stack((x_window, y_window))

    # PCA
    pca.fit(data_window)

    # 第一主成分ベクトル
    pca_vector = pca.components_[0]

    # 単位ベクトル化
    norm = np.linalg.norm(pca_vector)
    if norm == 0:
        pca_line.set_data([], [])
        return (points, pca_line)

    norm_vector = pca_vector / norm

    # 第一主成分軸の両端
    start, end = pca_axis_endpoints(norm_vector, detail_limit)

    # 赤線更新
    pca_line.set_data([start[0], end[0]], [start[1], end[1]])

    return (points, pca_line)


# =========================================================
# 21. アニメーション生成
# =========================================================

# CSV全データを使用
animation = FuncAnimation(
    fig,
    update,
    frames=len(df),
    blit=False,
    interval=1000.0 / real_time_fps
)


# =========================================================
# 22. MP4保存
# =========================================================

try:
    print("MP4の保存を開始します。")

    output_fps = max(1, int(round(real_time_fps)))
    writer = FFMpegWriter(
        fps=output_fps,
        codec='libx264',
        bitrate=3000,
        extra_args=['-pix_fmt', 'yuv420p']
    )
    animation.save(OUTPUT_MP4_PATH, writer=writer, dpi=120)

    print("MP4の保存が完了しました。")
    print(f"保存先: {OUTPUT_MP4_PATH}")
except Exception as e:
    print(f"MP4保存中にエラーが発生しました: {e}")

plt.close(fig)
