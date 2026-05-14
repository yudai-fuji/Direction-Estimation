# --- 角速度PCA法（第二主成分）---
# Gyro を世界座標系に変換し，
# X_rotated, Y_rotated に対して PCA を適用．
# 第二主成分軸と x軸正方向とのなす角を進行方向として算出する．
# 左右平均のRMS誤差を算出し，時系列を表示する．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import japanize_matplotlib

file_L = '0428_1L.csv'
file_R = '0428_1R.csv'
delay_time = 2.730487      # 左右端末の開始時刻ずれ補正 [s]
cod_time = 23.87       # 方向転換時刻 [s]
limit_min_time = 10.85
limit_max_time = 28.16

WINDOW_SIZE = 40      # PCA window
is_mask = 0           # プロット表示．1: 範囲内のみ表示．1以外: 全体表示
is_mask_g = 0         # 角度表示．1: -180~180° に正規化．1以外: 指定なし


def wrap_pm180(theta_deg):
    if is_mask_g == 1:
        """角度を [-180, 180) に正規化（表示用）"""
        theta_deg = np.asarray(theta_deg, dtype=float)
        return np.mod(theta_deg + 180.0, 360.0) - 180.0

    return theta_deg


def angle_diff_pm180(pred_deg, true_deg):
    """誤差（pred-true）を最短角度差として [-180, 180) に正規化（RMSE用）"""
    pred_deg = np.asarray(pred_deg, dtype=float)
    true_deg = np.asarray(true_deg, dtype=float)
    return np.mod((pred_deg - true_deg) + 180.0, 360.0) - 180.0


# =========================================================
# 共通：RMSE計算
# heading_* は列 time_s, theta_deg を持つこと
# =========================================================
def calc_rms_from_headings(heading_R, heading_L):
    heading_L = heading_L.copy()
    heading_R = heading_R.copy()

    # 計測開始時刻のずれを補正
    heading_L['time_s'] = heading_L['time_s'] + delay_time

    # 念のため time_s でソート
    heading_R = heading_R.sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.sort_values('time_s').reset_index(drop=True)

    # heading_R と heading_L を結合
    aligned = pd.merge_asof(
        heading_R,
        heading_L,
        on='time_s',
        direction='backward',
        suffixes=('_R', '_L'),
        tolerance=0.05
    )

    # マッチできなかった行は削除
    aligned = aligned.dropna(subset=['theta_deg_R', 'theta_deg_L']).reset_index(drop=True)

    # 左右の平均角度
    theta_mean = (aligned['theta_deg_R'] + aligned['theta_deg_L']) / 2.0

    # 評価区間
    t_all = aligned['time_s'].to_numpy()
    mask = (t_all >= limit_min_time) & (t_all <= limit_max_time)

    t_eval = t_all[mask]
    theta_R_eval = aligned.loc[mask, 'theta_deg_R'].to_numpy()
    theta_L_eval = aligned.loc[mask, 'theta_deg_L'].to_numpy()
    theta_mean_eval = theta_mean.loc[mask].to_numpy()

    # 真値
    true_heading_eval = np.where(t_eval <= cod_time, 90.0, 0.0)

    # 誤差
    err_R = angle_diff_pm180(theta_R_eval, true_heading_eval)
    err_L = angle_diff_pm180(theta_L_eval, true_heading_eval)
    err_mean = angle_diff_pm180(theta_mean_eval, true_heading_eval)

    # RMS
    RMS_R_deg = np.sqrt(np.mean(err_R ** 2))
    RMS_L_deg = np.sqrt(np.mean(err_L ** 2))
    RMS_mean_deg = np.sqrt(np.mean(err_mean ** 2))

    return RMS_R_deg, RMS_L_deg, RMS_mean_deg


# =========================================================
# 共通：時系列プロット
# =========================================================
def plot_heading_timeseries(heading_R, heading_L, title_str, ylabel_str):
    heading_R = heading_R.sort_values('time_s').reset_index(drop=True).copy()
    heading_L = heading_L.sort_values('time_s').reset_index(drop=True).copy()

    # delay補正
    heading_L['time_s'] = heading_L['time_s'] + delay_time

    # Rを基準にLを時刻同期
    aligned = pd.merge_asof(
        heading_R,
        heading_L,
        on='time_s',
        direction='backward',
        suffixes=('_R', '_L'),
        tolerance=0.05
    )
    aligned = aligned.dropna(subset=['theta_deg_R', 'theta_deg_L']).reset_index(drop=True)

    t_plot = aligned['time_s'].to_numpy()
    theta_R_plot = aligned['theta_deg_R'].to_numpy()
    theta_L_plot = aligned['theta_deg_L'].to_numpy()
    theta_mean_plot = (theta_R_plot + theta_L_plot) / 2.0
    true_heading_all = np.where(t_plot <= cod_time, 90.0, 0.0)

    if is_mask == 1:
        mask = (limit_min_time <= t_plot) & (t_plot <= limit_max_time)

        t_plot = t_plot[mask]
        theta_R_plot = theta_R_plot[mask]
        theta_L_plot = theta_L_plot[mask]
        theta_mean_plot = theta_mean_plot[mask]
        true_heading_all = true_heading_all[mask]

    theta_R_plot = wrap_pm180(theta_R_plot)
    theta_L_plot = wrap_pm180(theta_L_plot)
    theta_mean_plot = wrap_pm180(theta_mean_plot)
    true_heading_all = wrap_pm180(true_heading_all)

    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, theta_L_plot, label='左手の端末', c='b', alpha=0.8)
    plt.plot(t_plot, theta_R_plot, label='右手の端末', c='r', alpha=0.8)
    plt.plot(t_plot, theta_mean_plot, label='左右平均', c='g', linewidth=2, alpha=0.8)
    plt.plot(t_plot, true_heading_all, label='真値', c='k', linestyle='--', alpha=0.8)

    plt.xlabel('時間 [s]')
    plt.ylabel(ylabel_str)
    plt.title(title_str)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 角速度データPCA法（第二主成分）
# CSV -> (time_s, theta_deg) を返す
# =========================================================
def compute_heading_from_gyro_pca(file_path, window_size=40):
    df = pd.read_csv(file_path)

    # センサ種別で分離
    gyro_df = df[df['Sensor'] == 'Gyro'].copy()
    gamerot_df = df[df['Sensor'] == 'GameRo'].copy()

    # Timestamp を基準に近い GameRo を結合
    merged_df = pd.merge_asof(
        gyro_df.sort_values('Timestamp'),
        gamerot_df.sort_values('Timestamp'),
        on='Timestamp',
        direction='backward',
        suffixes=('_gyro', '_ro')
    )

    # --- 角速度とクォータニオン ---
    gyx = merged_df['X_gyro']
    gyy = merged_df['Y_gyro']
    gyz = merged_df['Z_gyro']

    gx = merged_df['X_ro']
    gy = merged_df['Y_ro']
    gz = merged_df['Z_ro']
    gw = merged_df['W_ro']

    # 基準姿勢（GameRo の 1 行目）
    gx0, gy0, gz0, gw0 = gamerot_df.iloc[0][['X', 'Y', 'Z', 'W']]

    def kyoyaku(qx, qy, qz, qw):
        return (-qx, -qy, -qz, qw)

    Gx0, Gy0, Gz0, Gw0 = kyoyaku(gx0, gy0, gz0, gw0)

    # 相対クォータニオン
    gwc = gw * Gw0 - gx * Gx0 - gy * Gy0 - gz * Gz0
    gxc = gw * Gx0 + gx * Gw0 - gy * Gz0 + gz * Gy0
    gyc = gw * Gy0 + gx * Gz0 + gy * Gw0 - gz * Gx0
    gzc = gw * Gz0 - gx * Gy0 + gy * Gx0 + gz * Gw0

    # 正規化
    norm = np.sqrt(gwc**2 + gxc**2 + gyc**2 + gzc**2)
    gwc = gwc / norm
    gxc = gxc / norm
    gyc = gyc / norm
    gzc = gzc / norm

    # 世界座標系へ変換した角速度
    Gyx = (2 * gwc * gwc + 2 * gxc * gxc - 1) * gyx + (2 * gxc * gyc - 2 * gzc * gwc) * gyy + (2 * gxc * gzc + 2 * gyc * gwc) * gyz
    Gyy = (2 * gxc * gyc + 2 * gzc * gwc) * gyx + (2 * gwc * gwc + 2 * gyc * gyc - 1) * gyy + (2 * gyc * gzc - 2 * gxc * gwc) * gyz
    Gyz = (2 * gxc * gzc - 2 * gyc * gwc) * gyx + (2 * gyc * gzc + 2 * gxc * gwc) * gyy + (2 * gwc * gwc + 2 * gzc * gzc - 1) * gyz

    # rad/s -> deg/s
    Gyx = Gyx * 180.0 / math.pi
    Gyy = Gyy * 180.0 / math.pi
    Gyz = Gyz * 180.0 / math.pi

    merged_df['X_rotated'] = Gyx
    merged_df['Y_rotated'] = Gyy
    merged_df['Z_rotated'] = Gyz

    merged_df = merged_df.dropna(subset=['X_rotated', 'Y_rotated', 'Z_rotated']).reset_index(drop=True)

    time_data = merged_df['Timestamp'].values
    time_start_ns = time_data[0]

    x_rotated = merged_df['X_rotated'].values
    y_rotated = merged_df['Y_rotated'].values

    pca = PCA(n_components=2)

    time_list = []
    deg_list = []

    for frame in range(len(x_rotated)):
        if frame + 1 < window_size:
            continue

        start_index = frame - (window_size - 1)
        end_index = frame + 1

        data_window = np.column_stack((
            x_rotated[start_index:end_index],
            y_rotated[start_index:end_index]
        ))

        pca.fit(data_window)

        # 第二主成分ベクトル
        vx, vy = pca.components_[1]

        angle_rad = np.arctan2(vy, vx)
        angle_deg = np.degrees(angle_rad)

        t_sec = (time_data[frame] - time_start_ns) / 1_000_000_000.0

        time_list.append(t_sec)
        deg_list.append(angle_deg)

    heading_df = pd.DataFrame({
        'time_s': time_list,
        'theta_deg': deg_list
    })

    return heading_df


# =========================================================
# 1) 角速度データPCA法（第二主成分）の heading と RMS
# =========================================================
heading_L_gyro_pca = compute_heading_from_gyro_pca(file_L, window_size=WINDOW_SIZE)
heading_R_gyro_pca = compute_heading_from_gyro_pca(file_R, window_size=WINDOW_SIZE)

RMS_R, RMS_L, RMS_mean = calc_rms_from_headings(heading_R_gyro_pca, heading_L_gyro_pca)

print("=== 角速度データPCA法（第二主成分） ===")
print(f"右手のRMS誤差:      {RMS_R:.4f} [deg]")
print(f"左手のRMS誤差:      {RMS_L:.4f} [deg]")
print(f"左右平均のRMS誤差:  {RMS_mean:.4f} [deg]")


# =========================================================
# 2) 時系列プロット
# =========================================================
plot_heading_timeseries(
    heading_R_gyro_pca,
    heading_L_gyro_pca,
    title_str='時系列変化（角速度PCA法）',
    ylabel_str='推定進行方向 [°]'
)
