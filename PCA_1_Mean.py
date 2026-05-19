import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation, PillowWriter
import japanize_matplotlib


# =========================================================
# 0) ユーザー設定
# =========================================================
file_L = r'260518栁澤共同研究\KL1.csv'
file_R = r'260518栁澤共同研究\KR1.csv'

delay_time = 0.892716
limit_min_time = 2
limit_max_time = 45
WINDOW_SIZE = 40

MERGE_TOLERANCE = 0.05
OUTPUT_GIF_PATH = 'acc_pca_pc1_left_right_mean.gif'

# 実時間に近い再生を目指して，データの時間刻みから自動でfpsを決める
# もしGIFが重すぎる場合は 2, 3 ... と上げてください
FRAME_STRIDE = 1

# 見た目
FIGSIZE = (8, 8)
POINT_SIZE = 10
LEFT_COLOR = 'blue'
RIGHT_COLOR = 'red'
MEAN_COLOR = 'green'

LEFT_ALPHA = 0.30
RIGHT_ALPHA = 0.30
MEAN_ALPHA = 1.00

LEFT_LINEWIDTH = 2.0
RIGHT_LINEWIDTH = 2.0
MEAN_LINEWIDTH = 4.0


# =========================================================
# 1) 加速度を世界座標系へ変換
#    （元コードの加速度PCA法の前処理をそのまま利用）
# =========================================================
def load_rotated_acc(file_path):
    df = pd.read_csv(file_path)

    acc_df = df[df['Sensor'] == 'Lacc'].copy()
    gamerot_df = df[df['Sensor'] == 'GameRo'].copy()

    if len(acc_df) == 0:
        raise ValueError(f'{file_path} に Sensor == "Lacc" のデータがありません．')
    if len(gamerot_df) == 0:
        raise ValueError(f'{file_path} に Sensor == "GameRo" のデータがありません．')

    merged_df = pd.merge_asof(
        acc_df.sort_values('Timestamp'),
        gamerot_df.sort_values('Timestamp'),
        on='Timestamp',
        direction='backward',
        suffixes=('_acc', '_ro')
    )

    # 全列 dropna はしない．
    # ここでは計算に必要な列だけを対象にする．
    required_cols = [
        'Timestamp',
        'X_acc', 'Y_acc', 'Z_acc',
        'X_ro', 'Y_ro', 'Z_ro', 'W_ro'
    ]

    merged_df = merged_df.dropna(subset=required_cols).reset_index(drop=True)

    if len(merged_df) == 0:
        raise ValueError(
            f'{file_path} で Lacc と GameRo を時刻対応付けした後，有効なデータがありません．'
            'GameRo の時刻が Lacc より後から始まっている可能性があります．'
        )

    ax = merged_df['X_acc'].to_numpy()
    ay = merged_df['Y_acc'].to_numpy()
    az = merged_df['Z_acc'].to_numpy()

    gx = merged_df['X_ro'].to_numpy()
    gy = merged_df['Y_ro'].to_numpy()
    gz = merged_df['Z_ro'].to_numpy()
    gw = merged_df['W_ro'].to_numpy()

    gx0, gy0, gz0, gw0 = gamerot_df.iloc[0][['X', 'Y', 'Z', 'W']]

    def kyoyaku(x, y, z, w):
        return (-x, -y, -z, w)

    Gx0, Gy0, Gz0, Gw0 = kyoyaku(gx0, gy0, gz0, gw0)

    gwc = gw * Gw0 - gx * Gx0 - gy * Gy0 - gz * Gz0
    gxc = gw * Gx0 + gx * Gw0 - gy * Gz0 + gz * Gy0
    gyc = gw * Gy0 + gx * Gz0 + gy * Gw0 - gz * Gx0
    gzc = gw * Gz0 - gx * Gy0 + gy * Gx0 + gz * Gw0

    norm = np.sqrt(gwc**2 + gxc**2 + gyc**2 + gzc**2)

    valid_norm = norm != 0
    gwc = gwc[valid_norm] / norm[valid_norm]
    gxc = gxc[valid_norm] / norm[valid_norm]
    gyc = gyc[valid_norm] / norm[valid_norm]
    gzc = gzc[valid_norm] / norm[valid_norm]

    ax = ax[valid_norm]
    ay = ay[valid_norm]
    az = az[valid_norm]
    timestamp = merged_df['Timestamp'].to_numpy()[valid_norm]

    Ax = (2 * gwc * gwc + 2 * gxc * gxc - 1) * ax + (2 * gxc * gyc - 2 * gzc * gwc) * ay + (2 * gxc * gzc + 2 * gyc * gwc) * az
    Ay = (2 * gxc * gyc + 2 * gzc * gwc) * ax + (2 * gwc * gwc + 2 * gyc * gyc - 1) * ay + (2 * gyc * gzc - 2 * gxc * gwc) * az
    Az = (2 * gxc * gzc - 2 * gyc * gwc) * ax + (2 * gyc * gzc + 2 * gxc * gwc) * ay + (2 * gwc * gwc + 2 * gzc * gzc - 1) * az

    rotated_df = pd.DataFrame({
        'Timestamp': timestamp,
        'X_rotated': Ax,
        'Y_rotated': Ay,
        'Z_rotated': Az
    })

    rotated_df = rotated_df.dropna(
        subset=['X_rotated', 'Y_rotated', 'Z_rotated']
    ).reset_index(drop=True)

    if len(rotated_df) == 0:
        raise ValueError(f'{file_path} で座標変換後の有効データがありません．')

    time_ns = rotated_df['Timestamp'].to_numpy()
    time_s = (time_ns - time_ns[0]) / 1_000_000_000.0

    rotated_df['time_s'] = time_s

    return rotated_df[['time_s', 'X_rotated', 'Y_rotated', 'Z_rotated']]


# =========================================================
# 2) 各時刻のPCA窓と第一主成分軸を計算
# =========================================================
def compute_acc_pca_windows(file_path, window_size=40):
    rotated_df = load_rotated_acc(file_path)

    x = rotated_df['X_rotated'].to_numpy()
    y = rotated_df['Y_rotated'].to_numpy()
    t = rotated_df['time_s'].to_numpy()

    pca = PCA(n_components=2)

    rows = []

    for frame in range(len(rotated_df)):
        if frame + 1 < window_size:
            continue

        start_idx = frame - (window_size - 1)
        end_idx = frame + 1

        x_window = x[start_idx:end_idx]
        y_window = y[start_idx:end_idx]

        data_window = np.column_stack((x_window, y_window))
        pca.fit(data_window)

        vx, vy = pca.components_[0]

        rows.append({
            'time_s': t[frame],
            'vx': vx,
            'vy': vy,
            'cx': np.mean(x_window),
            'cy': np.mean(y_window),
            'x_window': x_window.copy(),
            'y_window': y_window.copy()
        })

    return pd.DataFrame(rows)


# =========================================================
# 3) 左右のPCA結果を時刻合わせ
#    左手には delay_time を加える
# =========================================================
def align_left_right_pca(pca_R, pca_L):
    pca_R = pca_R.copy().sort_values('time_s').reset_index(drop=True)
    pca_L = pca_L.copy().sort_values('time_s').reset_index(drop=True)

    pca_L['time_s'] = pca_L['time_s'] + delay_time

    aligned = pd.merge_asof(
        pca_R,
        pca_L,
        on='time_s',
        direction='backward',
        tolerance=MERGE_TOLERANCE,
        suffixes=('_R', '_L')
    )

    required_cols = [
        'vx_R', 'vy_R', 'cx_R', 'cy_R', 'x_window_R', 'y_window_R',
        'vx_L', 'vy_L', 'cx_L', 'cy_L', 'x_window_L', 'y_window_L'
    ]

    aligned = aligned.dropna(subset=required_cols).reset_index(drop=True)

    mask = (aligned['time_s'] >= limit_min_time) & (aligned['time_s'] <= limit_max_time)
    aligned = aligned.loc[mask].reset_index(drop=True)

    if FRAME_STRIDE > 1:
        aligned = aligned.iloc[::FRAME_STRIDE].reset_index(drop=True)

    return aligned


# =========================================================
# 4) 固定表示範囲を計算
# =========================================================
def compute_fixed_limits(aligned_df):
    if len(aligned_df) == 0:
        raise ValueError('表示対象データがありません．')

    x_all = []
    y_all = []

    for i in range(len(aligned_df)):
        row = aligned_df.iloc[i]
        x_all.append(np.asarray(row['x_window_L']))
        x_all.append(np.asarray(row['x_window_R']))
        y_all.append(np.asarray(row['y_window_L']))
        y_all.append(np.asarray(row['y_window_R']))

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)

    x_min, x_max = np.min(x_all), np.max(x_all)
    y_min, y_max = np.min(y_all), np.max(y_all)

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    if max_range == 0:
        max_range = 1.0

    margin = 0.15 * max_range
    half_span = (max_range / 2.0) + margin

    xlim = (x_center - half_span, x_center + half_span)
    ylim = (y_center - half_span, y_center + half_span)

    # 軸の長さ
    axis_length = 0.8 * half_span

    return xlim, ylim, axis_length


# =========================================================
# 5) 主成分軸の線分端点を計算
#    PCAの「軸」なので，両方向へ伸ばす
# =========================================================
def axis_endpoints(cx, cy, vx, vy, axis_length):
    vec = np.array([vx, vy], dtype=float)
    norm = np.linalg.norm(vec)

    if norm == 0:
        return None

    vec = vec / norm

    x1 = cx - axis_length * vec[0]
    y1 = cy - axis_length * vec[1]
    x2 = cx + axis_length * vec[0]
    y2 = cy + axis_length * vec[1]

    return x1, y1, x2, y2


# =========================================================
# 6) 左右平均の主成分軸
#    ユーザー指定どおり，
#    左右の第一主成分ベクトルを平均して作る
#    ※180°補正は入れない
# =========================================================
def mean_axis_vector(vx_R, vy_R, vx_L, vy_L):
    v_mean = np.array([vx_R + vx_L, vy_R + vy_L], dtype=float)
    norm = np.linalg.norm(v_mean)

    if norm == 0:
        return None

    v_mean = v_mean / norm
    return v_mean[0], v_mean[1]


# =========================================================
# 7) GIFのfpsをデータから決める
# =========================================================
def estimate_fps(time_array):
    if len(time_array) < 2:
        return 1

    dt = np.diff(time_array)
    dt = dt[dt > 0]

    if len(dt) == 0:
        return 1

    median_dt = np.median(dt)
    fps = int(round(1.0 / median_dt))

    return max(fps, 1)


# =========================================================
# 8) GIF生成
# =========================================================
def create_gif():
    # 左右それぞれのPCA結果
    pca_L = compute_acc_pca_windows(file_L, window_size=WINDOW_SIZE)
    pca_R = compute_acc_pca_windows(file_R, window_size=WINDOW_SIZE)

    # 左右を時刻合わせ
    aligned = align_left_right_pca(pca_R, pca_L)

    if len(aligned) == 0:
        raise ValueError('時刻合わせ後に使用可能なデータがありません．')

    # 固定表示範囲
    xlim, ylim, axis_length = compute_fixed_limits(aligned)

    # 実時間に近いfps
    fps = estimate_fps(aligned['time_s'].to_numpy())

    fig, ax = plt.subplots(figsize=FIGSIZE)

    def update(frame_idx):
        ax.clear()

        row = aligned.iloc[frame_idx]

        # 点群
        xL = np.asarray(row['x_window_L'])
        yL = np.asarray(row['y_window_L'])
        xR = np.asarray(row['x_window_R'])
        yR = np.asarray(row['y_window_R'])

        ax.scatter(xL, yL, s=POINT_SIZE, c=LEFT_COLOR, alpha=LEFT_ALPHA, label='左手 点群')
        ax.scatter(xR, yR, s=POINT_SIZE, c=RIGHT_COLOR, alpha=RIGHT_ALPHA, label='右手 点群')

        # 左手の第一主成分軸
        ep_L = axis_endpoints(row['cx_L'], row['cy_L'], row['vx_L'], row['vy_L'], axis_length)
        if ep_L is not None:
            x1, y1, x2, y2 = ep_L
            ax.plot([x1, x2], [y1, y2],
                    color=LEFT_COLOR, linewidth=LEFT_LINEWIDTH, alpha=0.9,
                    label='左手 第一主成分軸')

        # 右手の第一主成分軸
        ep_R = axis_endpoints(row['cx_R'], row['cy_R'], row['vx_R'], row['vy_R'], axis_length)
        if ep_R is not None:
            x1, y1, x2, y2 = ep_R
            ax.plot([x1, x2], [y1, y2],
                    color=RIGHT_COLOR, linewidth=RIGHT_LINEWIDTH, alpha=0.9,
                    label='右手 第一主成分軸')

        # 左右平均の第一主成分軸
        mean_v = mean_axis_vector(row['vx_R'], row['vy_R'], row['vx_L'], row['vy_L'])
        if mean_v is not None:
            mean_cx = (row['cx_R'] + row['cx_L']) / 2.0
            mean_cy = (row['cy_R'] + row['cy_L']) / 2.0

            ep_M = axis_endpoints(mean_cx, mean_cy, mean_v[0], mean_v[1], axis_length)
            if ep_M is not None:
                x1, y1, x2, y2 = ep_M
                ax.plot([x1, x2], [y1, y2],
                        color=MEAN_COLOR, linewidth=MEAN_LINEWIDTH, alpha=MEAN_ALPHA,
                        label='左右平均 第一主成分軸')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

        ax.set_xlabel('X_rotated')
        ax.set_ylabel('Y_rotated')
        ax.set_title(f'加速度PCA法：第一主成分軸の変化\nTime = {row["time_s"]:.2f} s')

        ax.legend(loc='upper right')

    anim = FuncAnimation(
        fig,
        update,
        frames=len(aligned),
        interval=1000 / fps,
        repeat=True
    )

    writer = PillowWriter(fps=fps)
    anim.save(OUTPUT_GIF_PATH, writer=writer)

    plt.close(fig)

    print(f'GIFを保存しました: {OUTPUT_GIF_PATH}')
    print(f'フレーム数: {len(aligned)}')
    print(f'fps: {fps}')


# =========================================================
# 実行
# =========================================================
if __name__ == '__main__':
    create_gif()