# --- 加速度PCA法 / 寄与率重み付き加速度PCA法 / 角速度累積法 / 角速度PCA法 を
#     True / False で切り替えて実行する統合版 ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import japanize_matplotlib


# =========================================================
# 0) ユーザー設定
# =========================================================
# =========================================================
file_L = r'260518栁澤共同研究\KL1.csv'
file_R = r'260518栁澤共同研究\KR1.csv'
delay_time = 0.892716
cod_time = 21.99
limit_min_time = 2
limit_max_time = 45

eval_min_time = 0
eval_max_time = 0

WINDOW_SIZE = 40

is_mask = 1       # 時系列プロット表示 1: 範囲内のみ表示，1以外: 全体表示
is_mask_g = 1     # 角度表示 1: [-180，180) に変換，1以外: 指定なし


# =========================================================
# 実行する手法の選択
# =========================================================
DO_ACC_PCA = True              # 加速度PCA法
DO_PROPOSED_ACC_PCA = True     # 寄与率重み付き加速度PCA法
DO_GYRO_INTEGRAL = True        # 角速度累積法
DO_GYRO_PCA = True             # 角速度PCA法


# =========================================================
# 出力する処理の選択
# =========================================================
DO_RMSE = True                 # RMSE比較表
DO_SAMPLE_COUNT = True         # 進行方向推定に使用したサンプル数
DO_ERROR_STATS = True          # 指定区間の絶対誤差統計
DO_PCA_RATIO_PLOT = False      # PCA寄与率プロット
DO_PCA_RATIO_PRINT = True      # PCA寄与率の統計表示


# =========================================================
# 時系列プロットの選択
# =========================================================
PLOT_ACC_PCA = True            # 加速度PCA法
PLOT_PROPOSED_ACC_PCA = True   # 寄与率重み付き加速度PCA法
PLOT_GYRO_INTEGRAL = True      # 角速度累積法
PLOT_GYRO_PCA = True           # 角速度PCA法


# =========================================================
# 角度を [-180，180) に正規化する処理
# =========================================================
def wrap_pm180(theta_deg):
    if is_mask_g == 1:
        theta_deg = np.asarray(theta_deg, dtype=float)
        return np.mod(theta_deg + 180.0, 360.0) - 180.0

    return theta_deg


# =========================================================
# 推定角度と真値角度の差を最短角度差として計算する処理
# =========================================================
def angle_diff_pm180(pred_deg, true_deg):
    pred_deg = np.asarray(pred_deg, dtype=float)
    true_deg = np.asarray(true_deg, dtype=float)
    return np.mod((pred_deg - true_deg) + 180.0, 360.0) - 180.0


# =========================================================
# PCA寄与率の統計表示
# =========================================================
def print_pca_summary(df, label):
    if df is None or len(df) == 0:
        print(f"\n=== {label}の寄与率の各項目 ===")
        print("表示できるデータがありません．")
        return

    print(f"\n=== {label}の寄与率の各項目 ===")
    print(f"PC1 平均: {df['pc1_ratio'].mean():.3f}")
    print(f"PC1 最大: {df['pc1_ratio'].max():.3f}")
    print(f"PC1 最小: {df['pc1_ratio'].min():.3f}")
    print(f"PC2 平均: {df['pc2_ratio'].mean():.3f}")


# =========================================================
# 通常の左右平均でRMSEを計算する処理
# 対象: 加速度PCA法，角速度累積法，角速度PCA法
# =========================================================
def calc_rms_from_headings(heading_R, heading_L):
    heading_L = heading_L.copy()
    heading_R = heading_R.copy()

    heading_L['time_s'] = heading_L['time_s'] + delay_time

    heading_R = heading_R.sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.sort_values('time_s').reset_index(drop=True)

    aligned = pd.merge_asof(
        heading_R,
        heading_L,
        on='time_s',
        direction='backward',
        suffixes=('_R', '_L'),
        tolerance=0.05
    )

    aligned = aligned.dropna(subset=['theta_deg_R', 'theta_deg_L']).reset_index(drop=True)

    if len(aligned) == 0:
        return np.nan, np.nan, np.nan

    theta_mean = (aligned['theta_deg_R'] + aligned['theta_deg_L']) / 2.0

    t_all = aligned['time_s'].to_numpy()
    mask = (t_all >= limit_min_time) & (t_all <= limit_max_time)

    if np.sum(mask) == 0:
        return np.nan, np.nan, np.nan

    t_eval = t_all[mask]
    theta_R_eval = aligned.loc[mask, 'theta_deg_R'].to_numpy()
    theta_L_eval = aligned.loc[mask, 'theta_deg_L'].to_numpy()
    theta_mean_eval = theta_mean.loc[mask].to_numpy()

    true_heading_eval = np.where(t_eval <= cod_time, 90.0, 0.0)

    err_R = angle_diff_pm180(theta_R_eval, true_heading_eval)
    err_L = angle_diff_pm180(theta_L_eval, true_heading_eval)
    err_mean = angle_diff_pm180(theta_mean_eval, true_heading_eval)

    RMS_R_deg = np.sqrt(np.mean(err_R**2))
    RMS_L_deg = np.sqrt(np.mean(err_L**2))
    RMS_mean_deg = np.sqrt(np.mean(err_mean**2))

    return RMS_R_deg, RMS_L_deg, RMS_mean_deg


# =========================================================
# 寄与率重み付き加速度PCA法用のRMSE計算
# =========================================================
def calc_rms_from_weighted_vectors(heading_R, heading_L):
    heading_L = heading_L.copy()
    heading_R = heading_R.copy()

    heading_L['time_s'] = heading_L['time_s'] + delay_time

    heading_R = heading_R.sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.sort_values('time_s').reset_index(drop=True)

    aligned = pd.merge_asof(
        heading_R,
        heading_L,
        on='time_s',
        direction='backward',
        suffixes=('_R', '_L'),
        tolerance=0.05
    )

    aligned = aligned.dropna(
        subset=['theta_deg_R', 'theta_deg_L', 'qx_R', 'qy_R', 'qx_L', 'qy_L']
    ).reset_index(drop=True)

    if len(aligned) == 0:
        return np.nan, np.nan, np.nan

    qx_mean = (aligned['qx_R'].to_numpy() + aligned['qx_L'].to_numpy()) / 2.0
    qy_mean = (aligned['qy_R'].to_numpy() + aligned['qy_L'].to_numpy()) / 2.0
    theta_mean = np.degrees(np.arctan2(qy_mean, qx_mean))

    t_all = aligned['time_s'].to_numpy()
    mask = (t_all >= limit_min_time) & (t_all <= limit_max_time)

    if np.sum(mask) == 0:
        return np.nan, np.nan, np.nan

    t_eval = t_all[mask]
    theta_R_eval = aligned.loc[mask, 'theta_deg_R'].to_numpy()
    theta_L_eval = aligned.loc[mask, 'theta_deg_L'].to_numpy()
    theta_mean_eval = theta_mean[mask]

    true_heading_eval = np.where(t_eval <= cod_time, 90.0, 0.0)

    err_R = angle_diff_pm180(theta_R_eval, true_heading_eval)
    err_L = angle_diff_pm180(theta_L_eval, true_heading_eval)
    err_mean = angle_diff_pm180(theta_mean_eval, true_heading_eval)

    RMS_R_deg = np.sqrt(np.mean(err_R**2))
    RMS_L_deg = np.sqrt(np.mean(err_L**2))
    RMS_mean_deg = np.sqrt(np.mean(err_mean**2))

    return RMS_R_deg, RMS_L_deg, RMS_mean_deg


# =========================================================
# 通常の左右平均に対する指定区間の絶対誤差統計
# =========================================================
def print_error_stats_from_headings(heading_R, heading_L, method_name,
                                    eval_min_time, eval_max_time):
    heading_R = heading_R.copy().sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.copy().sort_values('time_s').reset_index(drop=True)

    heading_L['time_s'] = heading_L['time_s'] + delay_time

    aligned = pd.merge_asof(
        heading_R,
        heading_L,
        on='time_s',
        direction='backward',
        suffixes=('_R', '_L'),
        tolerance=0.05
    )

    aligned = aligned.dropna(subset=['theta_deg_R', 'theta_deg_L']).reset_index(drop=True)

    theta_mean = (aligned['theta_deg_R'] + aligned['theta_deg_L']) / 2.0

    t_all = aligned['time_s'].to_numpy()
    mask = (t_all >= eval_min_time) & (t_all <= eval_max_time)

    t_eval = t_all[mask]
    theta_mean_eval = theta_mean.loc[mask].to_numpy()

    if len(t_eval) == 0:
        print(f"\n=== {method_name}（左右平均，{eval_min_time:.2f}s～{eval_max_time:.2f}s）===")
        print("指定区間に一致するデータがありません．")
        return

    true_heading_eval = np.where(t_eval <= cod_time, 90.0, 0.0)

    signed_err = angle_diff_pm180(theta_mean_eval, true_heading_eval)
    abs_err = np.abs(signed_err)

    err_max = np.max(abs_err)
    err_min = np.min(abs_err)
    err_mean = np.mean(abs_err)
    err_rmse = np.sqrt(np.mean(signed_err**2))

    print(f"\n=== {method_name}（左右平均，{eval_min_time:.2f}s～{eval_max_time:.2f}s）===")
    print(f"最大絶対誤差: {err_max:.4f} [deg]")
    print(f"最小絶対誤差: {err_min:.4f} [deg]")
    print(f"平均絶対誤差: {err_mean:.4f} [deg]")
    print(f"RMSE:         {err_rmse:.4f} [deg]")


# =========================================================
# 寄与率重み付き加速度PCA法に対する指定区間の絶対誤差統計
# =========================================================
def print_error_stats_from_weighted_vectors(heading_R, heading_L, method_name,
                                            eval_min_time, eval_max_time):
    heading_R = heading_R.copy().sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.copy().sort_values('time_s').reset_index(drop=True)

    heading_L['time_s'] = heading_L['time_s'] + delay_time

    aligned = pd.merge_asof(
        heading_R,
        heading_L,
        on='time_s',
        direction='backward',
        suffixes=('_R', '_L'),
        tolerance=0.05
    )

    aligned = aligned.dropna(
        subset=['theta_deg_R', 'theta_deg_L', 'qx_R', 'qy_R', 'qx_L', 'qy_L']
    ).reset_index(drop=True)

    qx_mean = (aligned['qx_R'].to_numpy() + aligned['qx_L'].to_numpy()) / 2.0
    qy_mean = (aligned['qy_R'].to_numpy() + aligned['qy_L'].to_numpy()) / 2.0
    theta_mean = np.degrees(np.arctan2(qy_mean, qx_mean))

    t_all = aligned['time_s'].to_numpy()
    mask = (t_all >= eval_min_time) & (t_all <= eval_max_time)

    t_eval = t_all[mask]
    theta_mean_eval = theta_mean[mask]

    if len(t_eval) == 0:
        print(f"\n=== {method_name}（左右平均，{eval_min_time:.2f}s～{eval_max_time:.2f}s）===")
        print("指定区間に一致するデータがありません．")
        return

    true_heading_eval = np.where(t_eval <= cod_time, 90.0, 0.0)

    signed_err = angle_diff_pm180(theta_mean_eval, true_heading_eval)
    abs_err = np.abs(signed_err)

    err_max = np.max(abs_err)
    err_min = np.min(abs_err)
    err_mean = np.mean(abs_err)
    err_rmse = np.sqrt(np.mean(signed_err**2))

    print(f"\n=== {method_name}（左右平均，{eval_min_time:.2f}s～{eval_max_time:.2f}s）===")
    print(f"最大絶対誤差: {err_max:.4f} [deg]")
    print(f"最小絶対誤差: {err_min:.4f} [deg]")
    print(f"平均絶対誤差: {err_mean:.4f} [deg]")
    print(f"RMSE:         {err_rmse:.4f} [deg]")


# =========================================================
# 通常の時系列プロット
# 対象: 加速度PCA法，角速度累積法，角速度PCA法
# =========================================================
def plot_heading_timeseries(heading_R, heading_L, title_str, ylabel_str):
    heading_R = heading_R.sort_values('time_s').reset_index(drop=True).copy()
    heading_L = heading_L.sort_values('time_s').reset_index(drop=True).copy()

    heading_L['time_s'] = heading_L['time_s'] + delay_time

    aligned = pd.merge_asof(
        heading_R,
        heading_L,
        on='time_s',
        direction='backward',
        suffixes=('_R', '_L'),
        tolerance=0.05
    )

    aligned = aligned.dropna(subset=['theta_deg_R', 'theta_deg_L']).reset_index(drop=True)

    if len(aligned) == 0:
        print(f"{title_str}: プロットできるデータがありません．")
        return

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

    if len(t_plot) == 0:
        print(f"{title_str}: 指定範囲内にプロットできるデータがありません．")
        return

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
# 寄与率重み付き加速度PCA法用の時系列プロット
# =========================================================
def plot_heading_timeseries_weighted_vectors(heading_R, heading_L, title_str, ylabel_str):
    heading_R = heading_R.sort_values('time_s').reset_index(drop=True).copy()
    heading_L = heading_L.sort_values('time_s').reset_index(drop=True).copy()

    heading_L['time_s'] = heading_L['time_s'] + delay_time

    aligned = pd.merge_asof(
        heading_R,
        heading_L,
        on='time_s',
        direction='backward',
        suffixes=('_R', '_L'),
        tolerance=0.05
    )

    aligned = aligned.dropna(
        subset=['theta_deg_R', 'theta_deg_L', 'qx_R', 'qy_R', 'qx_L', 'qy_L']
    ).reset_index(drop=True)

    if len(aligned) == 0:
        print(f"{title_str}: プロットできるデータがありません．")
        return

    t_plot = aligned['time_s'].to_numpy()
    theta_R_plot = aligned['theta_deg_R'].to_numpy()
    theta_L_plot = aligned['theta_deg_L'].to_numpy()

    qx_mean = (aligned['qx_R'].to_numpy() + aligned['qx_L'].to_numpy()) / 2.0
    qy_mean = (aligned['qy_R'].to_numpy() + aligned['qy_L'].to_numpy()) / 2.0
    theta_mean_plot = np.degrees(np.arctan2(qy_mean, qx_mean))

    true_heading_all = np.where(t_plot <= cod_time, 90.0, 0.0)

    if is_mask == 1:
        mask = (limit_min_time <= t_plot) & (t_plot <= limit_max_time)

        t_plot = t_plot[mask]
        theta_R_plot = theta_R_plot[mask]
        theta_L_plot = theta_L_plot[mask]
        theta_mean_plot = theta_mean_plot[mask]
        true_heading_all = true_heading_all[mask]

    if len(t_plot) == 0:
        print(f"{title_str}: 指定範囲内にプロットできるデータがありません．")
        return

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
# 加速度PCA法
# =========================================================
def compute_heading_from_file(file_path, window_size=40):
    df = pd.read_csv(file_path)

    acc_df = df[df['Sensor'] == 'Lacc'].copy()
    gamerot_df = df[df['Sensor'] == 'GameRo'].copy()

    merged_df = pd.merge_asof(
        acc_df.sort_values('Timestamp'),
        gamerot_df.sort_values('Timestamp'),
        on='Timestamp',
        direction='backward',
        suffixes=('_acc', '_ro')
    )

    ax = merged_df['X_acc']
    ay = merged_df['Y_acc']
    az = merged_df['Z_acc']

    gx = merged_df['X_ro']
    gy = merged_df['Y_ro']
    gz = merged_df['Z_ro']
    gw = merged_df['W_ro']

    gx0, gy0, gz0, gw0 = gamerot_df.iloc[0][['X', 'Y', 'Z', 'W']]

    def kyoyaku(gx, gy, gz, gw):
        return (-gx, -gy, -gz, gw)

    Gx0, Gy0, Gz0, Gw0 = kyoyaku(gx0, gy0, gz0, gw0)

    gwc = gw*Gw0 - gx*Gx0 - gy*Gy0 - gz*Gz0
    gxc = gw*Gx0 + gx*Gw0 - gy*Gz0 + gz*Gy0
    gyc = gw*Gy0 + gx*Gz0 + gy*Gw0 - gz*Gx0
    gzc = gw*Gz0 - gx*Gy0 + gy*Gx0 + gz*Gw0

    norm = np.sqrt(gwc**2 + gxc**2 + gyc**2 + gzc**2)
    gwc = gwc / norm
    gxc = gxc / norm
    gyc = gyc / norm
    gzc = gzc / norm

    Ax = (2*gwc*gwc + 2*gxc*gxc - 1)*ax + (2*gxc*gyc - 2*gzc*gwc)*ay + (2*gxc*gzc + 2*gyc*gwc)*az
    Ay = (2*gxc*gyc + 2*gzc*gwc)*ax + (2*gwc*gwc + 2*gyc*gyc - 1)*ay + (2*gyc*gzc - 2*gxc*gwc)*az
    Az = (2*gxc*gzc - 2*gyc*gwc)*ax + (2*gyc*gzc + 2*gxc*gwc)*ay + (2*gwc*gwc + 2*gzc*gzc - 1)*az

    merged_df['X_rotated'] = Ax
    merged_df['Y_rotated'] = Ay
    merged_df['Z_rotated'] = Az

    merged_df = merged_df.dropna(subset=['X_rotated', 'Y_rotated', 'Z_rotated']).reset_index(drop=True)

    time_data = merged_df['Timestamp'].values
    time_start_ns = time_data[0]

    x_rotated = merged_df['X_rotated'].values
    y_rotated = merged_df['Y_rotated'].values

    pca = PCA(n_components=2)

    time_list = []
    deg_list = []
    pc1_ratio_list = []
    pc2_ratio_list = []

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
        vx, vy = pca.components_[0]

        angle_rad = np.arctan2(vy, vx)
        angle_deg = np.degrees(angle_rad)

        pc1_ratio = pca.explained_variance_ratio_[0]
        pc2_ratio = pca.explained_variance_ratio_[1]

        t_sec = (time_data[frame] - time_start_ns) / 1_000_000_000

        time_list.append(t_sec)
        deg_list.append(angle_deg)
        pc1_ratio_list.append(pc1_ratio)
        pc2_ratio_list.append(pc2_ratio)

    heading_df = pd.DataFrame({
        'time_s': time_list,
        'theta_deg': deg_list,
        'pc1_ratio': pc1_ratio_list,
        'pc2_ratio': pc2_ratio_list
    })

    return heading_df


# =========================================================
# 寄与率重み付き加速度PCA法
# 添付コードの計算式をそのまま使用
# =========================================================
def compute_heading_from_file_proposed(file_path, window_size=40):
    df = pd.read_csv(file_path)

    acc_df = df[df['Sensor'] == 'Lacc'].copy()
    gamerot_df = df[df['Sensor'] == 'GameRo'].copy()

    merged_df = pd.merge_asof(
        acc_df.sort_values('Timestamp'),
        gamerot_df.sort_values('Timestamp'),
        on='Timestamp',
        direction='backward',
        suffixes=('_acc', '_ro')
    )

    ax = merged_df['X_acc']
    ay = merged_df['Y_acc']
    az = merged_df['Z_acc']

    gx = merged_df['X_ro']
    gy = merged_df['Y_ro']
    gz = merged_df['Z_ro']
    gw = merged_df['W_ro']

    gx0, gy0, gz0, gw0 = gamerot_df.iloc[0][['X', 'Y', 'Z', 'W']]

    def kyoyaku(gx, gy, gz, gw):
        return (-gx, -gy, -gz, gw)

    Gx0, Gy0, Gz0, Gw0 = kyoyaku(gx0, gy0, gz0, gw0)

    gwc = gw*Gw0 - gx*Gx0 - gy*Gy0 - gz*Gz0
    gxc = gw*Gx0 + gx*Gw0 - gy*Gz0 + gz*Gy0
    gyc = gw*Gy0 + gx*Gz0 + gy*Gw0 - gz*Gx0
    gzc = gw*Gz0 - gx*Gy0 + gy*Gx0 + gz*Gw0

    norm = np.sqrt(gwc**2 + gxc**2 + gyc**2 + gzc**2)
    gwc = gwc / norm
    gxc = gxc / norm
    gyc = gyc / norm
    gzc = gzc / norm

    Ax = (2*gwc*gwc + 2*gxc*gxc - 1)*ax + (2*gxc*gyc - 2*gzc*gwc)*ay + (2*gxc*gzc + 2*gyc*gwc)*az
    Ay = (2*gxc*gyc + 2*gzc*gwc)*ax + (2*gwc*gwc + 2*gyc*gyc - 1)*ay + (2*gyc*gzc - 2*gxc*gwc)*az
    Az = (2*gxc*gzc - 2*gyc*gwc)*ax + (2*gyc*gzc + 2*gxc*gwc)*ay + (2*gwc*gwc + 2*gzc*gzc - 1)*az

    merged_df['X_rotated'] = Ax
    merged_df['Y_rotated'] = Ay
    merged_df['Z_rotated'] = Az

    merged_df = merged_df.dropna(subset=['X_rotated', 'Y_rotated', 'Z_rotated']).reset_index(drop=True)

    time_data = merged_df['Timestamp'].values
    time_start_ns = time_data[0]

    x_rotated = merged_df['X_rotated'].values
    y_rotated = merged_df['Y_rotated'].values

    pca = PCA(n_components=2)

    time_list = []
    theta_deg_list = []
    qx_list = []
    qy_list = []
    theta1_deg_list = []
    theta2_deg_list = []
    pc1_ratio_list = []
    pc2_ratio_list = []

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
        vx, vy = pca.components_[0]

        theta1_rad = np.arctan2(vy, vx)
        theta2_rad = theta1_rad - np.pi / 2.0

        pc1_ratio = pca.explained_variance_ratio_[0]
        pc2_ratio = pca.explained_variance_ratio_[1]

        qx = pc1_ratio * np.cos(theta1_rad) + pc2_ratio * np.cos(theta2_rad)
        qy = pc1_ratio * np.sin(theta1_rad) + pc2_ratio * np.sin(theta2_rad)

        theta_deg = np.degrees(np.arctan2(qy, qx))
        theta1_deg = np.degrees(theta1_rad)
        theta2_deg = np.degrees(theta2_rad)

        t_sec = (time_data[frame] - time_start_ns) / 1_000_000_000

        time_list.append(t_sec)
        theta_deg_list.append(theta_deg)
        qx_list.append(qx)
        qy_list.append(qy)
        theta1_deg_list.append(theta1_deg)
        theta2_deg_list.append(theta2_deg)
        pc1_ratio_list.append(pc1_ratio)
        pc2_ratio_list.append(pc2_ratio)

    heading_df = pd.DataFrame({
        'time_s': time_list,
        'theta_deg': theta_deg_list,
        'qx': qx_list,
        'qy': qy_list,
        'theta1_deg': theta1_deg_list,
        'theta2_deg': theta2_deg_list,
        'pc1_ratio': pc1_ratio_list,
        'pc2_ratio': pc2_ratio_list
    })

    return heading_df


# =========================================================
# 角速度累積法
# =========================================================
def compute_heading(file_path, initial_heading_deg):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f'{file_path} が見つかりませんでした．')
        raise

    gyro_df = df[df['Sensor'] == 'Gyro'].copy()
    gamerot_df = df[df['Sensor'] == 'GameRo'].copy()

    merged_df = pd.merge_asof(
        gyro_df.sort_values('Timestamp'),
        gamerot_df.sort_values('Timestamp'),
        on='Timestamp',
        direction='backward',
        suffixes=('_gyro', '_ro')
    )

    gyx = merged_df['X_gyro']
    gyy = merged_df['Y_gyro']
    gyz = merged_df['Z_gyro']

    gx = merged_df['X_ro']
    gy = merged_df['Y_ro']
    gz = merged_df['Z_ro']
    gw = merged_df['W_ro']

    gx0, gy0, gz0, gw0 = gamerot_df.iloc[0][['X', 'Y', 'Z', 'W']]

    def kyoyaku(gx, gy, gz, gw):
        return (-gx, -gy, -gz, gw)

    Gx0, Gy0, Gz0, Gw0 = kyoyaku(gx0, gy0, gz0, gw0)

    gwc = gw*Gw0 - gx*Gx0 - gy*Gy0 - gz*Gz0
    gxc = gw*Gx0 + gx*Gw0 - gy*Gz0 + gz*Gy0
    gyc = gw*Gy0 + gx*Gz0 + gy*Gw0 - gz*Gx0
    gzc = gw*Gz0 - gx*Gy0 + gy*Gx0 + gz*Gw0

    norm = np.sqrt(gwc**2 + gxc**2 + gyc**2 + gzc**2)
    gwc = gwc / norm
    gxc = gxc / norm
    gyc = gyc / norm
    gzc = gzc / norm

    Gyx = (2*gwc*gwc + 2*gxc*gxc - 1)*gyx + (2*gxc*gyc - 2*gzc*gwc)*gyy + (2*gxc*gzc + 2*gyc*gwc)*gyz
    Gyy = (2*gxc*gyc + 2*gzc*gwc)*gyx + (2*gwc*gwc + 2*gyc*gyc - 1)*gyy + (2*gyc*gzc - 2*gxc*gwc)*gyz
    Gyz = (2*gxc*gzc - 2*gyc*gwc)*gyx + (2*gyc*gzc + 2*gxc*gwc)*gyy + (2*gwc*gwc + 2*gzc*gzc - 1)*gyz

    Gyx = Gyx * 180.0 / math.pi
    Gyy = Gyy * 180.0 / math.pi
    Gyz = Gyz * 180.0 / math.pi

    merged_df['X_rotated'] = Gyx
    merged_df['Y_rotated'] = Gyy
    merged_df['Z_rotated'] = Gyz

    merged_df = merged_df.dropna(subset=['X_rotated', 'Y_rotated', 'Z_rotated']).reset_index(drop=True)

    s_timestamp = merged_df['Timestamp'] / 1_000_000_000.0
    t = s_timestamp - s_timestamp.iloc[0]

    t_np = t.to_numpy()
    dt = np.diff(t_np, prepend=t_np[0])

    omega_z = merged_df['Z_rotated'].to_numpy()
    dtheta = omega_z * dt

    theta = initial_heading_deg + np.cumsum(dtheta)

    result = pd.DataFrame({
        'time_s': t,
        'theta_deg': theta
    })

    return result


# =========================================================
# 角速度PCA法
# Gyroを世界座標系に変換し，X_rotated，Y_rotatedにPCAを適用する
# 第二主成分を用いる
# =========================================================
def compute_heading_from_gyro_pca(file_path, window_size=40):
    df = pd.read_csv(file_path)

    gyro_df = df[df['Sensor'] == 'Gyro'].copy()
    gamerot_df = df[df['Sensor'] == 'GameRo'].copy()

    merged_df = pd.merge_asof(
        gyro_df.sort_values('Timestamp'),
        gamerot_df.sort_values('Timestamp'),
        on='Timestamp',
        direction='backward',
        suffixes=('_gyro', '_ro')
    )

    gyx = merged_df['X_gyro']
    gyy = merged_df['Y_gyro']
    gyz = merged_df['Z_gyro']

    gx = merged_df['X_ro']
    gy = merged_df['Y_ro']
    gz = merged_df['Z_ro']
    gw = merged_df['W_ro']

    gx0, gy0, gz0, gw0 = gamerot_df.iloc[0][['X', 'Y', 'Z', 'W']]

    def kyoyaku(gx, gy, gz, gw):
        return (-gx, -gy, -gz, gw)

    Gx0, Gy0, Gz0, Gw0 = kyoyaku(gx0, gy0, gz0, gw0)

    gwc = gw*Gw0 - gx*Gx0 - gy*Gy0 - gz*Gz0
    gxc = gw*Gx0 + gx*Gw0 - gy*Gz0 + gz*Gy0
    gyc = gw*Gy0 + gx*Gz0 + gy*Gw0 - gz*Gx0
    gzc = gw*Gz0 - gx*Gy0 + gy*Gx0 + gz*Gw0

    norm = np.sqrt(gwc**2 + gxc**2 + gyc**2 + gzc**2)
    gwc = gwc / norm
    gxc = gxc / norm
    gyc = gyc / norm
    gzc = gzc / norm

    Gyx = (2*gwc*gwc + 2*gxc*gxc - 1)*gyx + (2*gxc*gyc - 2*gzc*gwc)*gyy + (2*gxc*gzc + 2*gyc*gwc)*gyz
    Gyy = (2*gxc*gyc + 2*gzc*gwc)*gyx + (2*gwc*gwc + 2*gyc*gyc - 1)*gyy + (2*gyc*gzc - 2*gxc*gwc)*gyz
    Gyz = (2*gxc*gzc - 2*gyc*gwc)*gyx + (2*gyc*gzc + 2*gxc*gwc)*gyy + (2*gwc*gwc + 2*gzc*gzc - 1)*gyz

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
# PCA寄与率プロット用の前処理
# =========================================================
def prepare_pca_ratio_df_for_plot(ratio_df, add_delay=False):
    ratio_df = ratio_df.copy().sort_values('time_s').reset_index(drop=True)

    if add_delay:
        ratio_df['time_s'] = ratio_df['time_s'] + delay_time

    mask = (ratio_df['time_s'] >= limit_min_time) & (ratio_df['time_s'] <= limit_max_time)
    ratio_df = ratio_df.loc[mask].reset_index(drop=True)

    return ratio_df


# =========================================================
# PCA寄与率プロット
# =========================================================
def plot_pca_contribution_separate(ratio_df, hand_label, add_delay=False):
    ratio_plot = prepare_pca_ratio_df_for_plot(ratio_df, add_delay=add_delay)

    if len(ratio_plot) == 0:
        print(f"PCA寄与率（{hand_label}）: プロットできるデータがありません．")
        return

    t = ratio_plot['time_s']
    pc1 = ratio_plot['pc1_ratio']
    pc2 = ratio_plot['pc2_ratio']

    if '左' in hand_label:
        color_main = 'blue'
    else:
        color_main = 'red'

    plt.figure(figsize=(10, 6))

    plt.plot(
        t,
        pc1,
        label='第一主成分',
        color=color_main,
        linewidth=2,
        alpha=0.9
    )

    plt.plot(
        t,
        pc2,
        label='第二主成分',
        color=color_main,
        linewidth=2,
        alpha=0.4
    )

    plt.xlabel('時間 [s]')
    plt.ylabel('寄与率')
    plt.title(f'PCA寄与率（{hand_label}）')
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# RMSE比較表の表示
# =========================================================
def print_rmse_table(rmse_rows):
    if len(rmse_rows) == 0:
        print("\n=== RMSE比較 ===")
        print("表示するRMSEがありません．")
        return

    rmse_df = pd.DataFrame(
        rmse_rows,
        columns=['手法', '右手RMSE [deg]', '左手RMSE [deg]', '左右平均RMSE [deg]']
    )

    print("\n=== RMSE比較 ===")
    print(rmse_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# =========================================================
# 進行方向推定に使用したサンプル数の表示
# =========================================================
def count_eval_samples(heading_R, heading_L, method_name, required_cols):
    heading_R = heading_R.copy().sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.copy().sort_values('time_s').reset_index(drop=True)

    heading_L['time_s'] = heading_L['time_s'] + delay_time

    aligned_before_dropna = pd.merge_asof(
        heading_R,
        heading_L,
        on='time_s',
        direction='backward',
        suffixes=('_R', '_L'),
        tolerance=0.05
    )

    aligned_after_dropna = aligned_before_dropna.dropna(
        subset=required_cols
    ).reset_index(drop=True)

    if len(aligned_after_dropna) == 0:
        n_eval = 0
    else:
        t_all = aligned_after_dropna['time_s'].to_numpy()
        mask = (t_all >= limit_min_time) & (t_all <= limit_max_time)
        n_eval = int(np.sum(mask))

    return [
        method_name,
        len(heading_R),
        len(heading_L),
        len(aligned_before_dropna),
        len(aligned_after_dropna),
        n_eval
    ]


def print_sample_count_table(sample_rows):
    if len(sample_rows) == 0:
        print("\n=== 進行方向推定に使用したサンプル数 ===")
        print("表示するサンプル数がありません．")
        return

    sample_df = pd.DataFrame(
        sample_rows,
        columns=[
            '手法',
            '右手推定点数',
            '左手推定点数',
            'merge後点数',
            'dropna後点数',
            '評価区間内点数'
        ]
    )

    print("\n=== 進行方向推定に使用したサンプル数 ===")
    print(sample_df.to_string(index=False))


# =========================================================
# 1) 各手法の計算
# =========================================================
heading_L_pca = None
heading_R_pca = None

heading_L_prop = None
heading_R_prop = None

heading_L_gyro = None
heading_R_gyro = None

heading_L_gyro_pca = None
heading_R_gyro_pca = None


if DO_ACC_PCA:
    heading_L_pca = compute_heading_from_file(file_L, window_size=WINDOW_SIZE)
    heading_R_pca = compute_heading_from_file(file_R, window_size=WINDOW_SIZE)


if DO_PROPOSED_ACC_PCA:
    heading_L_prop = compute_heading_from_file_proposed(file_L, window_size=WINDOW_SIZE)
    heading_R_prop = compute_heading_from_file_proposed(file_R, window_size=WINDOW_SIZE)


if DO_GYRO_INTEGRAL:
    heading_L_gyro = compute_heading(file_L, initial_heading_deg=90.0)
    heading_R_gyro = compute_heading(file_R, initial_heading_deg=90.0)


if DO_GYRO_PCA:
    heading_L_gyro_pca = compute_heading_from_gyro_pca(file_L, window_size=WINDOW_SIZE)
    heading_R_gyro_pca = compute_heading_from_gyro_pca(file_R, window_size=WINDOW_SIZE)


# =========================================================
# 2) RMSE比較
# =========================================================
if DO_RMSE:
    rmse_rows = []

    if DO_ACC_PCA and heading_R_pca is not None and heading_L_pca is not None:
        RMS_R, RMS_L, RMS_mean = calc_rms_from_headings(heading_R_pca, heading_L_pca)
        rmse_rows.append(['加速度PCA法', RMS_R, RMS_L, RMS_mean])

    if DO_PROPOSED_ACC_PCA and heading_R_prop is not None and heading_L_prop is not None:
        RMS_R, RMS_L, RMS_mean = calc_rms_from_weighted_vectors(heading_R_prop, heading_L_prop)
        rmse_rows.append(['寄与率重み付き加速度PCA法', RMS_R, RMS_L, RMS_mean])

    if DO_GYRO_INTEGRAL and heading_R_gyro is not None and heading_L_gyro is not None:
        RMS_R, RMS_L, RMS_mean = calc_rms_from_headings(heading_R_gyro, heading_L_gyro)
        rmse_rows.append(['角速度累積法', RMS_R, RMS_L, RMS_mean])

    if DO_GYRO_PCA and heading_R_gyro_pca is not None and heading_L_gyro_pca is not None:
        RMS_R, RMS_L, RMS_mean = calc_rms_from_headings(heading_R_gyro_pca, heading_L_gyro_pca)
        rmse_rows.append(['角速度PCA法', RMS_R, RMS_L, RMS_mean])

    print_rmse_table(rmse_rows)


# =========================================================
# 2.5) 進行方向推定に使用したサンプル数
# =========================================================
if DO_SAMPLE_COUNT:
    sample_rows = []

    if DO_ACC_PCA and heading_R_pca is not None and heading_L_pca is not None:
        sample_rows.append(
            count_eval_samples(
                heading_R_pca,
                heading_L_pca,
                method_name='加速度PCA法',
                required_cols=['theta_deg_R', 'theta_deg_L']
            )
        )

    if DO_PROPOSED_ACC_PCA and heading_R_prop is not None and heading_L_prop is not None:
        sample_rows.append(
            count_eval_samples(
                heading_R_prop,
                heading_L_prop,
                method_name='寄与率重み付き加速度PCA法',
                required_cols=['theta_deg_R', 'theta_deg_L', 'qx_R', 'qy_R', 'qx_L', 'qy_L']
            )
        )

    if DO_GYRO_INTEGRAL and heading_R_gyro is not None and heading_L_gyro is not None:
        sample_rows.append(
            count_eval_samples(
                heading_R_gyro,
                heading_L_gyro,
                method_name='角速度累積法',
                required_cols=['theta_deg_R', 'theta_deg_L']
            )
        )

    if DO_GYRO_PCA and heading_R_gyro_pca is not None and heading_L_gyro_pca is not None:
        sample_rows.append(
            count_eval_samples(
                heading_R_gyro_pca,
                heading_L_gyro_pca,
                method_name='角速度PCA法',
                required_cols=['theta_deg_R', 'theta_deg_L']
            )
        )

    print_sample_count_table(sample_rows)


# =========================================================
# 3) 時系列プロット
# =========================================================
if PLOT_ACC_PCA:
    if DO_ACC_PCA and heading_R_pca is not None and heading_L_pca is not None:
        plot_heading_timeseries(
            heading_R_pca,
            heading_L_pca,
            title_str='時系列変化（加速度PCA法）',
            ylabel_str='推定進行方向 [°]'
        )
    else:
        print("加速度PCA法の時系列プロットは，手法がOFFのためスキップしました．")


if PLOT_PROPOSED_ACC_PCA:
    if DO_PROPOSED_ACC_PCA and heading_R_prop is not None and heading_L_prop is not None:
        plot_heading_timeseries_weighted_vectors(
            heading_R_prop,
            heading_L_prop,
            title_str='時系列変化（寄与率重み付き加速度PCA法）',
            ylabel_str='推定進行方向 [°]'
        )
    else:
        print("寄与率重み付き加速度PCA法の時系列プロットは，手法がOFFのためスキップしました．")


if PLOT_GYRO_INTEGRAL:
    if DO_GYRO_INTEGRAL and heading_R_gyro is not None and heading_L_gyro is not None:
        plot_heading_timeseries(
            heading_R_gyro,
            heading_L_gyro,
            title_str='時系列変化（角速度累積法）',
            ylabel_str='推定進行方向 [°]'
        )
    else:
        print("角速度累積法の時系列プロットは，手法がOFFのためスキップしました．")


if PLOT_GYRO_PCA:
    if DO_GYRO_PCA and heading_R_gyro_pca is not None and heading_L_gyro_pca is not None:
        plot_heading_timeseries(
            heading_R_gyro_pca,
            heading_L_gyro_pca,
            title_str='時系列変化（角速度PCA法）',
            ylabel_str='推定進行方向 [°]'
        )
    else:
        print("角速度PCA法の時系列プロットは，手法がOFFのためスキップしました．")


# =========================================================
# 4) 指定区間における左右平均絶対誤差の統計表示
# =========================================================
if DO_ERROR_STATS:
    if eval_min_time == eval_max_time:
        print("\n=== 指定区間の絶対誤差統計 ===")
        print("eval_min_time と eval_max_time が同じため，誤差統計をスキップしました．")

    elif eval_min_time > eval_max_time:
        print("\n=== 指定区間の絶対誤差統計 ===")
        print("eval_min_time が eval_max_time より大きいため，誤差統計をスキップしました．")

    else:
        if DO_ACC_PCA and heading_R_pca is not None and heading_L_pca is not None:
            print_error_stats_from_headings(
                heading_R_pca,
                heading_L_pca,
                method_name='加速度PCA法',
                eval_min_time=eval_min_time,
                eval_max_time=eval_max_time
            )

        if DO_PROPOSED_ACC_PCA and heading_R_prop is not None and heading_L_prop is not None:
            print_error_stats_from_weighted_vectors(
                heading_R_prop,
                heading_L_prop,
                method_name='寄与率重み付き加速度PCA法',
                eval_min_time=eval_min_time,
                eval_max_time=eval_max_time
            )

        if DO_GYRO_INTEGRAL and heading_R_gyro is not None and heading_L_gyro is not None:
            print_error_stats_from_headings(
                heading_R_gyro,
                heading_L_gyro,
                method_name='角速度累積法',
                eval_min_time=eval_min_time,
                eval_max_time=eval_max_time
            )

        if DO_GYRO_PCA and heading_R_gyro_pca is not None and heading_L_gyro_pca is not None:
            print_error_stats_from_headings(
                heading_R_gyro_pca,
                heading_L_gyro_pca,
                method_name='角速度PCA法',
                eval_min_time=eval_min_time,
                eval_max_time=eval_max_time
            )


# =========================================================
# 5) PCA寄与率プロット
# 加速度PCA法と寄与率重み付き加速度PCA法の寄与率は同じ
# 両方ONなら提案手法側を優先して表示
# =========================================================
ratio_L_source = None
ratio_R_source = None

if DO_PROPOSED_ACC_PCA and heading_L_prop is not None and heading_R_prop is not None:
    ratio_L_source = heading_L_prop
    ratio_R_source = heading_R_prop

elif DO_ACC_PCA and heading_L_pca is not None and heading_R_pca is not None:
    ratio_L_source = heading_L_pca
    ratio_R_source = heading_R_pca


if DO_PCA_RATIO_PLOT:
    if ratio_L_source is not None and ratio_R_source is not None:
        plot_pca_contribution_separate(
            ratio_L_source,
            hand_label='左手の端末',
            add_delay=True
        )

        plot_pca_contribution_separate(
            ratio_R_source,
            hand_label='右手の端末',
            add_delay=False
        )
    else:
        print("\nPCA寄与率プロットは，加速度PCA系の手法がOFFのためスキップしました．")


# =========================================================
# 6) PCA寄与率の統計表示
# =========================================================
if DO_PCA_RATIO_PRINT:
    if ratio_L_source is not None and ratio_R_source is not None:
        ratio_L = prepare_pca_ratio_df_for_plot(ratio_L_source, add_delay=True)
        ratio_R = prepare_pca_ratio_df_for_plot(ratio_R_source, add_delay=False)

        print_pca_summary(ratio_L, "左手")
        print_pca_summary(ratio_R, "右手")
    else:
        print("\nPCA寄与率の統計表示は，加速度PCA系の手法がOFFのためスキップしました．")