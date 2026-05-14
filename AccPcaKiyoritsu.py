# --- PCA法 / 角速度累積法 / 2手法平均（推定角度を時刻ごとに平均）を
#     RMS比較（棒グラフ）＋時系列比較（3枚）＋
#     PCA寄与率プロット（左右重ね描き＋左右別描き）する完成版 ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import japanize_matplotlib

file_L = '0428_1L.csv'
file_R = '0428_1R.csv'
delay_time = 2.737099       # 左右端末の開始時刻ずれ補正[s]
cod_time = 23.42       # 方向転換時刻[s]
limit_min_time = 12.91
limit_max_time = 35.49

#指定した範囲のRMSE
eval_min_time = 16.4
eval_max_time = 17.6

WINDOW_SIZE = 40       # PCA window
is_mask = 0            # プロット表示 1:範囲内のみ表示 1以外:全体表示
is_mask_g = 1          # 角速度累積法プロット表示 1:-180~180° 1以外:指定なし

#------------------------------------------------------
# is_mask_g = 1のとき，
# 角度を表示用に[-180°~180°)に変換する処理
# 引数:変換したい角度
#------------------------------------------------------
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

#------------------------------------------------------
# PCAの寄与率を4種類表示する処理
# 第一主成分の寄与率の平均，最大，最小，第二主成分の寄与率の平均
# 引数:寄与率が含まれたdfと名前
#------------------------------------------------------
def print_pca_summary(df, label):
    print(f"=== {label} ===")
    print(f"PC1 平均: {df['pc1_ratio'].mean():.3f}")
    print(f"PC1 最大: {df['pc1_ratio'].max():.3f}")
    print(f"PC1 最小: {df['pc1_ratio'].min():.3f}")
    print(f"PC2 平均: {df['pc2_ratio'].mean():.3f}")
    print("\n")

# =========================================================
# RMSEを計算して返す処理
# 右手単体，左手単体，左右平均の3つ
# 引数:右手と左手側の推定進行方向と時間が格納されたdf
# =========================================================
def calc_rms_from_headings(heading_R, heading_L):
    heading_L = heading_L.copy()
    heading_R = heading_R.copy()

    # 計測開始時刻のずれを補正
    heading_L['time_s'] = heading_L['time_s'] + delay_time

    # time_s でソート
    heading_R = heading_R.sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.sort_values('time_s').reset_index(drop=True)

    # heading_Rと_Lを結合
    aligned = pd.merge_asof(
        heading_R, heading_L,
        on='time_s',
        direction='backward',
        suffixes=('_R', '_L'),
        tolerance=0.05
    )

    # マッチできなかった行（NaN）は削除
    aligned = aligned.dropna(subset=['theta_deg_R', 'theta_deg_L']).reset_index(drop=True)

    # 左右の平均角度
    theta_mean = (aligned['theta_deg_R'] + aligned['theta_deg_L']) / 2.0

    # 評価区間
    t_all = aligned['time_s'].to_numpy()
    mask = (t_all >= limit_min_time) & (t_all <= limit_max_time)

    t_eval          = t_all[mask]
    theta_R_eval    = aligned.loc[mask, 'theta_deg_R'].to_numpy()
    theta_L_eval    = aligned.loc[mask, 'theta_deg_L'].to_numpy()
    theta_mean_eval = theta_mean.loc[mask].to_numpy()

    # 真値
    true_heading_eval = np.where(t_eval <= cod_time, 90.0, 0.0)

    # 誤差（最短角度差で評価）
    err_R    = angle_diff_pm180(theta_R_eval, true_heading_eval)
    err_L    = angle_diff_pm180(theta_L_eval, true_heading_eval)
    err_mean = angle_diff_pm180(theta_mean_eval, true_heading_eval)

    # RMS
    RMS_R_deg    = np.sqrt(np.mean(err_R**2))
    RMS_L_deg    = np.sqrt(np.mean(err_L**2))
    RMS_mean_deg = np.sqrt(np.mean(err_mean**2))

    return RMS_R_deg, RMS_L_deg, RMS_mean_deg

# =========================================================
# 指定した時間のRMSEを計算する処理
# =========================================================
def print_error_stats_from_headings(heading_R, heading_L, method_name,
                                    eval_min_time, eval_max_time):
    heading_R = heading_R.copy().sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.copy().sort_values('time_s').reset_index(drop=True)

    # 左手の時間ずれ補正
    heading_L['time_s'] = heading_L['time_s'] + delay_time

    # 右手基準で左手を時刻同期
    aligned = pd.merge_asof(
        heading_R,
        heading_L,
        on='time_s',
        direction='backward',
        suffixes=('_R', '_L'),
        tolerance=0.05
    )

    aligned = aligned.dropna(subset=['theta_deg_R', 'theta_deg_L']).reset_index(drop=True)

    # 左右平均角度
    theta_mean = (aligned['theta_deg_R'] + aligned['theta_deg_L']) / 2.0

    # 指定区間で抽出
    t_all = aligned['time_s'].to_numpy()
    mask = (t_all >= eval_min_time) & (t_all <= eval_max_time)

    t_eval = t_all[mask]
    theta_mean_eval = theta_mean.loc[mask].to_numpy()

    if len(t_eval) == 0:
        print(f"\n=== {method_name}（左右平均，{eval_min_time:.2f}s～{eval_max_time:.2f}s）===")
        print("指定区間に一致するデータがありません．")
        return

    # 真値
    true_heading_eval = np.where(t_eval <= cod_time, 90.0, 0.0)

    # 符号付き誤差 → 絶対誤差
    signed_err = angle_diff_pm180(theta_mean_eval, true_heading_eval)
    abs_err = np.abs(signed_err)

    # 統計量
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
# 推定進行方向の時系列変化をプロットする処理
# 右手，左手，左右平均，真値をプロットする
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


# =========================================================
# PCA進行方向推定法を行う処理
# 時間，進行方向[°]，寄与率(第一，第二主成分)の4つを返す
# =========================================================
def compute_heading_from_file(file_path, window_size=40):
    df = pd.read_csv(file_path)

    # センサ種別で分離
    acc_df     = df[df['Sensor'] == 'Lacc'].copy()
    gamerot_df = df[df['Sensor'] == 'GameRo'].copy()

    # Timestamp を基準に近い GameRo を結合
    merged_df = pd.merge_asof(
        acc_df.sort_values('Timestamp'),
        gamerot_df.sort_values('Timestamp'),
        on='Timestamp',
        direction='backward',
        suffixes=('_acc', '_ro')
    )

    # 加速度とクォータニオン
    ax = merged_df['X_acc']
    ay = merged_df['Y_acc']
    az = merged_df['Z_acc']

    gx = merged_df['X_ro']
    gy = merged_df['Y_ro']
    gz = merged_df['Z_ro']
    gw = merged_df['W_ro']

    # 基準姿勢（GameRoの1行目）
    gx0, gy0, gz0, gw0 = gamerot_df.iloc[0][['X', 'Y', 'Z', 'W']]

    def kyoyaku(gx, gy, gz, gw):
        return (-gx, -gy, -gz, gw)

    Gx0, Gy0, Gz0, Gw0 = kyoyaku(gx0, gy0, gz0, gw0)

    # 相対クォータニオン
    gwc = gw*Gw0 - gx*Gx0 - gy*Gy0 - gz*Gz0
    gxc = gw*Gx0 + gx*Gw0 - gy*Gz0 + gz*Gy0
    gyc = gw*Gy0 + gx*Gz0 + gy*Gw0 - gz*Gx0
    gzc = gw*Gz0 - gx*Gy0 + gy*Gx0 + gz*Gw0

    # 正規化
    norm = np.sqrt(gwc**2 + gxc**2 + gyc**2 + gzc**2)
    gwc = gwc / norm
    gxc = gxc / norm
    gyc = gyc / norm
    gzc = gzc / norm

    # 回転後加速度
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

    # 2次元PCAなので主成分は第1，第2まで
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
        vx, vy = pca.components_[0]   # 第一主成分ベクトル [vx, vy]

        angle_rad = np.arctan2(vy, vx)
        angle_deg = np.degrees(angle_rad)

        #寄与率を保持
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
# 角速度累積法を行う処理
# 進行方向[°]，時間を返す
# =========================================================
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
# 提案2手法を時間で結合し，平均する
# 時間と進行方向[°]を返す
# =========================================================
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


# =========================================================
# PCA寄与率の時系列変化をプロットするための準備を行う処理
# 左手に delay_time を加えて時間をそろえる
# 時間範囲 limit_min_time ～ limit_max_time に適用
# =========================================================
def prepare_pca_ratio_df_for_plot(ratio_df, add_delay=False):
    ratio_df = ratio_df.copy().sort_values('time_s').reset_index(drop=True)

    if add_delay:
        ratio_df['time_s'] = ratio_df['time_s'] + delay_time

    mask = (ratio_df['time_s'] >= limit_min_time) & (ratio_df['time_s'] <= limit_max_time)
    ratio_df = ratio_df.loc[mask].reset_index(drop=True)

    return ratio_df

# =========================================================
# PCA寄与率の時系列変化をプロットする処理 
# 右手，左手の結果をそれぞれ表示
# =========================================================
def plot_pca_contribution_separate(ratio_df, hand_label, add_delay=False):
    ratio_plot = prepare_pca_ratio_df_for_plot(ratio_df, add_delay=add_delay)

    t = ratio_plot['time_s']
    pc1 = ratio_plot['pc1_ratio']
    pc2 = ratio_plot['pc2_ratio']

    # 色設定（左＝青，右＝赤）
    if '左' in hand_label:
        color_main = 'blue'
    else:
        color_main = 'red'

    plt.figure(figsize=(10, 6))

    # 第一主成分（濃く・太く）
    plt.plot(t, pc1,
             label='第一主成分',
             color=color_main,
             linewidth=2,
             alpha=0.9)

    # 第二主成分（薄く・細く・破線）
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

print("=== PCA法 ===")
print(f"右手のRMS誤差:      {RMS_R_pca:.4f} [deg]")
print(f"左手のRMS誤差:      {RMS_L_pca:.4f} [deg]")
print(f"左右平均のRMS誤差:  {RMS_mean_pca:.4f} [deg]")


# =========================================================
# 2) 角速度累積法の heading と RMS
# =========================================================
heading_R_gyro = compute_heading(file_R, initial_heading_deg=90.0)
heading_L_gyro = compute_heading(file_L, initial_heading_deg=90.0)

RMS_R_gyro, RMS_L_gyro, RMS_mean_gyro = calc_rms_from_headings(heading_R_gyro, heading_L_gyro)

print("\n=== 角速度累積法 ===")
print(f"右手のRMS誤差:      {RMS_R_gyro:.4f} [deg]")
print(f"左手のRMS誤差:      {RMS_L_gyro:.4f} [deg]")
print(f"左右平均のRMS誤差:  {RMS_mean_gyro:.4f} [deg]")


# =========================================================
# 3) 2手法平均（推定角度を時刻ごとに平均してからRMS）
# =========================================================
heading_R_avg = fuse_two_methods_heading(heading_R_pca, heading_R_gyro)
heading_L_avg = fuse_two_methods_heading(heading_L_pca, heading_L_gyro)

RMS_R_avg, RMS_L_avg, RMS_mean_avg = calc_rms_from_headings(heading_R_avg, heading_L_avg)

print("\n=== 2手法平均（推定角度を時刻ごとに平均） ===")
print(f"右手のRMS誤差:      {RMS_R_avg:.4f} [deg]")
print(f"左手のRMS誤差:      {RMS_L_avg:.4f} [deg]")
print(f"左右平均のRMS誤差:  {RMS_mean_avg:.4f} [deg]")


# =========================================================
# 4) 時系列プロット（PCA / 角速度累積 / 2手法平均）
# =========================================================
plot_heading_timeseries(
    heading_R_pca, heading_L_pca,
    title_str='時系列変化 (PCA進行方向推定法)',
    ylabel_str='第一主成分軸とx軸とのなす角 [°]'
)

plot_heading_timeseries(
    heading_R_gyro, heading_L_gyro,
    title_str='時系列変化 (角速度累積法)',
    ylabel_str='推定進行方向 [°]'
)

# =========================================================
# 5) 指定区間における左右平均絶対誤差の統計表示
# =========================================================
print_error_stats_from_headings(
    heading_R_pca, heading_L_pca,
    method_name='加速度PCA法',
    eval_min_time=eval_min_time,
    eval_max_time=eval_max_time
)

print_error_stats_from_headings(
    heading_R_gyro, heading_L_gyro,
    method_name='角速度累積法',
    eval_min_time=eval_min_time,
    eval_max_time=eval_max_time
)

# =========================================================
# 6) PCA寄与率プロット
# 左右を別々に描く
# =========================================================
plot_pca_contribution_separate(
    heading_L_pca,
    hand_label='左手の端末',
    add_delay=True
)

plot_pca_contribution_separate(
    heading_R_pca,
    hand_label='右手の端末',
    add_delay=False
)

# =========================================================
# 7) 寄与率をPrint表示(詳細に)
# =========================================================
ratio_L = prepare_pca_ratio_df_for_plot(heading_L_pca, add_delay=True)
ratio_R = prepare_pca_ratio_df_for_plot(heading_R_pca, add_delay=False)

print_pca_summary(ratio_L, "左手")
print_pca_summary(ratio_R, "右手")