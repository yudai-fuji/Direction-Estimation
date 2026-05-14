# --- PCA法 / 提案手法 / 角速度累積法 / 2手法平均 を
#     RMS比較＋時系列比較＋PCA寄与率プロットする版 ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import japanize_matplotlib

file_L = '0429_2L.csv'
file_R = '0429_2R.csv'
delay_time = 1.259762      # 左右端末の開始時刻ずれ補正 [s]
cod_time = 23      # 方向転換時刻 [s]
limit_min_time = 12.66
limit_max_time = 33.55

# 指定した範囲のRMSE
eval_min_time = 0
eval_max_time = 0

WINDOW_SIZE = 40       # PCA window
is_mask = 1            # プロット表示 1:範囲内のみ表示 1以外:全体表示
is_mask_g = 1          # 角速度累積法プロット表示 1:-180~180° 1以外:指定なし


# 補足=================================================================================
# ・is_mask_g が 1 のとき，角度を [-180, 180) に正規化する関数
# ・それ以外のときは入力値をそのまま返す
# ・何を返すか：正規化後の角度配列，または入力値そのものを返す
# ・引数：正規化したい角度
def wrap_pm180(theta_deg):
    if is_mask_g == 1:
        """角度を [-180, 180) に正規化（表示用）"""
        theta_deg = np.asarray(theta_deg, dtype=float)
        return np.mod(theta_deg + 180.0, 360.0) - 180.0

    return theta_deg


# 補足=================================================================================
# ・推定角度と真値角度の差を計算し，最短角度差として [-180, 180) に正規化する関数
# ・何を返すか：正規化後の角度差を返す
# ・引数は何か：pred_deg は推定角度，true_deg は真値角度
def angle_diff_pm180(pred_deg, true_deg):
    """誤差（pred-true）を最短角度差として [-180, 180) に正規化（RMSE用）"""
    pred_deg = np.asarray(pred_deg, dtype=float)
    true_deg = np.asarray(true_deg, dtype=float)
    return np.mod((pred_deg - true_deg) + 180.0, 360.0) - 180.0

# 補足=================================================================================
# ・何をする処理か：PCA の寄与率が入った DataFrame から，PC1 と PC2 の統計量を表示する関数
# ・何を返すか：printによって結果を表示
# ・引数は何か：df は　pc1_ratio と pc2_ratio 列を持つ DataFrame，label は表示用ラベル
def print_pca_summary(df, label):
    print(f"\n=== {label}の寄与率の各項目 ===")
    print(f"PC1 平均: {df['pc1_ratio'].mean():.3f}")
    print(f"PC1 最大: {df['pc1_ratio'].max():.3f}")
    print(f"PC1 最小: {df['pc1_ratio'].min():.3f}")
    print(f"PC2 平均: {df['pc2_ratio'].mean():.3f}")

# 補足=================================================================================
# ・何をする処理か：左右端末の推定角度系列を時刻合わせし，評価区間内の右手，左手，左右平均の RMSE を計算
# ・何を返すか：右手 RMSE，左手 RMSE，左右平均 RMSE の 3 つを返す．
# ・引数は何か：heading_R と heading_L は time_s 列と theta_deg 列を持つ DataFrame
def calc_rms_from_headings(heading_R, heading_L):
    heading_L = heading_L.copy()
    heading_R = heading_R.copy()

    heading_L['time_s'] = heading_L['time_s'] + delay_time

    heading_R = heading_R.sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.sort_values('time_s').reset_index(drop=True)

    aligned = pd.merge_asof(
        heading_R, heading_L,
        on='time_s',
        direction='backward',
        suffixes=('_R', '_L'),
        tolerance=0.05
    )

    aligned = aligned.dropna(subset=['theta_deg_R', 'theta_deg_L']).reset_index(drop=True)

    # 既存法では角度平均
    theta_mean = (aligned['theta_deg_R'] + aligned['theta_deg_L']) / 2.0

    t_all = aligned['time_s'].to_numpy()
    mask = (t_all >= limit_min_time) & (t_all <= limit_max_time)

    t_eval          = t_all[mask]
    theta_R_eval    = aligned.loc[mask, 'theta_deg_R'].to_numpy()
    theta_L_eval    = aligned.loc[mask, 'theta_deg_L'].to_numpy()
    theta_mean_eval = theta_mean.loc[mask].to_numpy()

    true_heading_eval = np.where(t_eval <= cod_time, 90.0, 0.0)

    err_R    = angle_diff_pm180(theta_R_eval, true_heading_eval)
    err_L    = angle_diff_pm180(theta_L_eval, true_heading_eval)
    err_mean = angle_diff_pm180(theta_mean_eval, true_heading_eval)

    RMS_R_deg    = np.sqrt(np.mean(err_R**2))
    RMS_L_deg    = np.sqrt(np.mean(err_L**2))
    RMS_mean_deg = np.sqrt(np.mean(err_mean**2))

    return RMS_R_deg, RMS_L_deg, RMS_mean_deg

# 補足=================================================================================
# ・何をする処理か：左右端末の qx，qy を時刻合わせし，左右ベクトル平均から求めた角度を使って評価区間内の RMSE を計算
# ・何を返すか：右手 RMSE，左手 RMSE，左右平均 RMSE の 3 つを返す
# ・引数は何か：heading_R と heading_L は time_s，theta_deg，qx，qy 列を持つ DataFrame
def calc_rms_from_weighted_vectors(heading_R, heading_L):
    heading_L = heading_L.copy()
    heading_R = heading_R.copy()

    heading_L['time_s'] = heading_L['time_s'] + delay_time

    heading_R = heading_R.sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.sort_values('time_s').reset_index(drop=True)

    aligned = pd.merge_asof(
        heading_R, heading_L,
        on='time_s',
        direction='backward',
        suffixes=('_R', '_L'),
        tolerance=0.05
    )

    aligned = aligned.dropna(
        subset=['theta_deg_R', 'theta_deg_L', 'qx_R', 'qy_R', 'qx_L', 'qy_L']
    ).reset_index(drop=True)

    # 左右ベクトル平均
    qx_mean = (aligned['qx_R'].to_numpy() + aligned['qx_L'].to_numpy()) / 2.0
    qy_mean = (aligned['qy_R'].to_numpy() + aligned['qy_L'].to_numpy()) / 2.0
    theta_mean = np.degrees(np.arctan2(qy_mean, qx_mean))

    t_all = aligned['time_s'].to_numpy()
    mask = (t_all >= limit_min_time) & (t_all <= limit_max_time)

    t_eval          = t_all[mask]
    theta_R_eval    = aligned.loc[mask, 'theta_deg_R'].to_numpy()
    theta_L_eval    = aligned.loc[mask, 'theta_deg_L'].to_numpy()
    theta_mean_eval = theta_mean[mask]

    true_heading_eval = np.where(t_eval <= cod_time, 90.0, 0.0)

    err_R    = angle_diff_pm180(theta_R_eval, true_heading_eval)
    err_L    = angle_diff_pm180(theta_L_eval, true_heading_eval)
    err_mean = angle_diff_pm180(theta_mean_eval, true_heading_eval)

    RMS_R_deg    = np.sqrt(np.mean(err_R**2))
    RMS_L_deg    = np.sqrt(np.mean(err_L**2))
    RMS_mean_deg = np.sqrt(np.mean(err_mean**2))

    return RMS_R_deg, RMS_L_deg, RMS_mean_deg

# 補足=================================================================================
# ・何をする処理か：指定した時間区間に対して，左右平均角度の最大絶対誤差，最小絶対誤差，平均絶対誤差，RMSE を表示
# ・何を返すか：値は返さず，print によって結果を表示する
# ・引数は何か：heading_R と heading_L は time_s 列と theta_deg 列を持つ DataFrame，method_name は表示名，
#   eval_min_time と eval_max_time は評価区間の開始時刻と終了時刻
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

# 補足=================================================================================
# ・何をする処理か：指定した時間区間に対して，左右ベクトル平均から得た角度の最大絶対誤差，最小絶対誤差，平均絶対誤差，RMSE を表示します．
# ・何を返すか：値は返さず，print によって結果を表示
# ・引数は何か：heading_R と heading_L は time_s，theta_deg，qx，qy 列を持つ DataFrame，method_name は表示名，
#   eval_min_time と eval_max_time は評価区間の開始時刻と終了時刻です．
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

# 補足=================================================================================
# ・何をする処理か：左右端末と左右平均，真値の時系列を同じグラフにプロット
# ・何を返すか：値は返さず，matplotlib でグラフを表示する
# ・引数は何か：heading_R と heading_L は time_s 列と theta_deg 列を持つ DataFrame，title_str はタイトル，ylabel_str は縦軸ラベルです．
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

    theta_R_plot     = wrap_pm180(theta_R_plot)
    theta_L_plot     = wrap_pm180(theta_L_plot)
    theta_mean_plot  = wrap_pm180(theta_mean_plot)
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
    plt.tight_layout()
    plt.show()

# 補足=================================================================================
# ・何をする処理か：左右端末と，左右ベクトル平均から得た推定角度，真値の時系列を同じグラフにプロット
# ・何を返すか：値は返さず，matplotlib でグラフを表示
# ・引数は何か：heading_R と heading_L は time_s，theta_deg，qx，qy 列を持つ DataFrame，title_str はタイトル，ylabel_str は縦軸ラベルです．
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

    theta_R_plot     = wrap_pm180(theta_R_plot)
    theta_L_plot     = wrap_pm180(theta_L_plot)
    theta_mean_plot  = wrap_pm180(theta_mean_plot)
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
    plt.tight_layout()
    plt.show()

# 補足=================================================================================
# ・何をする処理か：加速度PCA法．CSV を読み込み，加速度を世界座標系へ変換した後，スライディングウィンドウごとに PCA を行って進行方向角と寄与率を求めます．
# ・何を返すか：time_s，theta_deg，pc1_ratio，pc2_ratio を持つ DataFrame を返します．
# ・引数は何か：file_path は入力 CSV のパス，window_size は PCA に使う窓長です．
def compute_heading_from_file(file_path, window_size=40):
    df = pd.read_csv(file_path)

    acc_df     = df[df['Sensor'] == 'Lacc'].copy()
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

    # 補足
    # ・何をする処理か：四元数の共役を計算する補助関数です．
    # ・何を返すか：(-gx, -gy, -gz, gw) の 4 つ組を返します．
    # ・引数は何か：gx，gy，gz，gw は四元数の各成分です．
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
        end_index   = frame + 1

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

# 補足=================================================================================
# ・何をする処理か：CSV を読み込み，加速度を世界座標系へ変換した後，PCA の第 1・第 2 主成分と寄与率から各端末内の重み付きベクトルを作り，その偏角を求める
# ・何を返すか：time_s，theta_deg，qx，qy，theta1_deg，theta2_deg，pc1_ratio，pc2_ratio を持つ DataFrame を返します．
# ・引数は何か：file_path は入力 CSV のパス，window_size は PCA に使う窓長です．
def compute_heading_from_file_proposed(file_path, window_size=40):
    df = pd.read_csv(file_path)

    acc_df     = df[df['Sensor'] == 'Lacc'].copy()
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

    # 補足
    # ・何をする処理か：四元数の共役を計算する補助関数です．
    # ・何を返すか：(-gx, -gy, -gz, gw) の 4 つ組を返します．
    # ・引数は何か：gx，gy，gz，gw は四元数の各成分です．
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
        end_index   = frame + 1

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

        # 各端末内の寄与率重み付きベクトル
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


# ・何をする処理か：角速度累積法．CSV を読み込み，角速度を世界座標系へ変換し，Z 軸角速度を時間積分して推定角度を求めます．
# ・何を返すか：time_s と theta_deg を持つ DataFrame を返します．
# ・引数は何か：file_path は入力 CSV のパス，initial_heading_deg は積分開始時の初期方位角です．
def compute_heading(file_path, initial_heading_deg):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f'{file_path} が見つかりませんでした')
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

    # 補足
    # ・何をする処理か：四元数の共役を計算する補助関数です．
    # ・何を返すか：(-gx, -gy, -gz, gw) の 4 つ組を返します．
    # ・引数は何か：gx，gy，gz，gw は四元数の各成分です．
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

# 補足=================================================================================
# ・何をする処理か：PCA 法と角速度累積法の推定角度を時刻合わせし，両者の角度平均から融合結果を作ります．
# ・何を返すか：time_s と theta_deg を持つ融合後の DataFrame を返します．
# ・引数は何か：heading_pca と heading_gyro は time_s 列と theta_deg 列を持つ DataFrame です．
def fuse_two_methods_heading(heading_pca, heading_gyro):
    heading_pca  = heading_pca.sort_values('time_s').reset_index(drop=True)
    heading_gyro = heading_gyro.sort_values('time_s').reset_index(drop=True)

    fused = pd.merge_asof(
        heading_pca,
        heading_gyro,
        on='time_s',
        direction='backward',
        suffixes=('_pca', '_gyro'),
        tolerance=0.05
    )
    fused = fused.dropna(subset=['theta_deg_pca', 'theta_deg_gyro']).reset_index(drop=True)

    fused_theta = (fused['theta_deg_pca'] + fused['theta_deg_gyro']) / 2.0

    heading_fused = pd.DataFrame({
        'time_s': fused['time_s'],
        'theta_deg': fused_theta
    })

    return heading_fused


# 補足=================================================================================
# ・何をする処理か：寄与率プロット用に DataFrame を時刻順へ並べ替え，必要なら delay_time を加え，指定範囲だけに絞ります．
# ・何を返すか：プロット用に整形された DataFrame を返します．
# ・引数は何か：ratio_df は time_s 列を持つ DataFrame，add_delay は左手側へ delay_time を加えるかどうかの真偽値です．
def prepare_pca_ratio_df_for_plot(ratio_df, add_delay=False):
    ratio_df = ratio_df.copy().sort_values('time_s').reset_index(drop=True)

    if add_delay:
        ratio_df['time_s'] = ratio_df['time_s'] + delay_time

    mask = (ratio_df['time_s'] >= limit_min_time) & (ratio_df['time_s'] <= limit_max_time)
    ratio_df = ratio_df.loc[mask].reset_index(drop=True)

    return ratio_df

# 補足=================================================================================
# ・何をする処理か：片手分の PCA 寄与率を前処理し，第一主成分と第二主成分を別々の線として描画します．
# ・何を返すか：値は返さず，matplotlib でグラフを表示します．
# ・引数は何か：ratio_df は pc1_ratio と pc2_ratio を含む DataFrame，hand_label は表示用ラベル，add_delay は delay_time を加えるかどうかです．
def plot_pca_contribution_separate(ratio_df, hand_label, add_delay=False):
    ratio_plot = prepare_pca_ratio_df_for_plot(ratio_df, add_delay=add_delay)

    t = ratio_plot['time_s']
    pc1 = ratio_plot['pc1_ratio']
    pc2 = ratio_plot['pc2_ratio']

    if '左' in hand_label:
        color_main = 'blue'
    else:
        color_main = 'red'

    plt.figure(figsize=(10, 6))

    plt.plot(t, pc1,
             label='第一主成分',
             color=color_main,
             linewidth=2,
             alpha=0.9)

    plt.plot(t, pc2,
             label='第二主成分',
             color=color_main,
             linewidth=2,
             alpha=0.4)

    plt.xlabel('時間 [s]')
    plt.ylabel('寄与率')
    plt.title(f'PCA寄与率（{hand_label}）')
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 1) PCA法の heading と RMS
# =========================================================
heading_L_pca = compute_heading_from_file(file_L, window_size=WINDOW_SIZE)
heading_R_pca = compute_heading_from_file(file_R, window_size=WINDOW_SIZE)

RMS_R_pca, RMS_L_pca, RMS_mean_pca = calc_rms_from_headings(heading_R_pca, heading_L_pca)

print("=== 加速度PCA法 ===")
print(f"右手のRMS誤差:      {RMS_R_pca:.4f} [deg]")
print(f"左手のRMS誤差:      {RMS_L_pca:.4f} [deg]")
print(f"左右平均のRMS誤差:  {RMS_mean_pca:.4f} [deg]")


# =========================================================
# 2) 提案手法の heading と RMS
# =========================================================
heading_L_prop = compute_heading_from_file_proposed(file_L, window_size=WINDOW_SIZE)
heading_R_prop = compute_heading_from_file_proposed(file_R, window_size=WINDOW_SIZE)

RMS_R_prop, RMS_L_prop, RMS_mean_prop = calc_rms_from_weighted_vectors(heading_R_prop, heading_L_prop)

print("\n=== 加速度PCA法（寄与率重み付き） ===")
print(f"右手のRMS誤差:      {RMS_R_prop:.4f} [deg]")
print(f"左手のRMS誤差:      {RMS_L_prop:.4f} [deg]")
print(f"左右平均のRMS誤差:  {RMS_mean_prop:.4f} [deg]")

"""
# =========================================================
# 3) 角速度累積法の heading と RMS
# =========================================================
heading_R_gyro = compute_heading(file_R, initial_heading_deg=90.0)
heading_L_gyro = compute_heading(file_L, initial_heading_deg=90.0)

RMS_R_gyro, RMS_L_gyro, RMS_mean_gyro = calc_rms_from_headings(heading_R_gyro, heading_L_gyro)

print("\n=== 角速度累積法 ===")
print(f"右手のRMS誤差:      {RMS_R_gyro:.4f} [deg]")
print(f"左手のRMS誤差:      {RMS_L_gyro:.4f} [deg]")
print(f"左右平均のRMS誤差:  {RMS_mean_gyro:.4f} [deg]")
"""
""""
# =========================================================
# 4) 2手法平均（従来PCA法と角速度累積法の角度平均）
# =========================================================
heading_R_avg = fuse_two_methods_heading(heading_R_pca, heading_R_gyro)
heading_L_avg = fuse_two_methods_heading(heading_L_pca, heading_L_gyro)

RMS_R_avg, RMS_L_avg, RMS_mean_avg = calc_rms_from_headings(heading_R_avg, heading_L_avg)

print("\n=== 2手法平均（推定角度を時刻ごとに平均） ===")
print(f"右手のRMS誤差:      {RMS_R_avg:.4f} [deg]")
print(f"左手のRMS誤差:      {RMS_L_avg:.4f} [deg]")
print(f"左右平均のRMS誤差:  {RMS_mean_avg:.4f} [deg]")
"""

# =========================================================
# 5) 時系列プロット
# =========================================================
plot_heading_timeseries(
    heading_R_pca, heading_L_pca,
    title_str='時系列変化 (加速度PCA法)',
    ylabel_str='推定進行方向 [°]'
)

plot_heading_timeseries_weighted_vectors(
    heading_R_prop, heading_L_prop,
    title_str='時系列変化 (寄与率重み付き加速度PCA法)',
    ylabel_str='推定進行方向 [°]'
)
"""
plot_heading_timeseries(
    heading_R_gyro, heading_L_gyro,
    title_str='時系列変化 (角速度累積法)',
    ylabel_str='推定進行方向 [°]'
)
"""
"""
# =========================================================
# 6) 指定区間における左右平均絶対誤差の統計表示
# =========================================================
print_error_stats_from_headings(
    heading_R_pca, heading_L_pca,
    method_name='加速度PCA法',
    eval_min_time=eval_min_time,
    eval_max_time=eval_max_time
)


print_error_stats_from_weighted_vectors(
    heading_R_prop, heading_L_prop,
    method_name='寄与率重み付き加速度PCA法',
    eval_min_time=eval_min_time,
    eval_max_time=eval_max_time
)


print_error_stats_from_headings(
    heading_R_gyro, heading_L_gyro,
    method_name='角速度累積法',
    eval_min_time=eval_min_time,
    eval_max_time=eval_max_time
)
"""


# =========================================================
# 7) PCA寄与率プロット
# 既存PCA法と提案手法で寄与率は同じなので，提案手法側を表示
# =========================================================
plot_pca_contribution_separate(
    heading_L_prop,
    hand_label='左手の端末',
    add_delay=True
)

plot_pca_contribution_separate(
    heading_R_prop,
    hand_label='右手の端末',
    add_delay=False
)


# =========================================================
# 8) 寄与率をPrint表示(詳細に)
# =========================================================
ratio_L = prepare_pca_ratio_df_for_plot(heading_L_prop, add_delay=True)
ratio_R = prepare_pca_ratio_df_for_plot(heading_R_prop, add_delay=False)

print_pca_summary(ratio_L, "左手")
print_pca_summary(ratio_R, "右手")