# --- 加速度PCA法 / 寄与率重み付き加速度PCA法 / 角速度累積法 / 角速度PCA法 を
#     True / False で切り替えて実行する統合版
#     センサデータ整合仕様:
#       B案: 左端末の time_s に delay_time を加えたうえで，
#            左右の重複区間に SAMPLE_INTERVAL ごとの共通時刻グリッドを作成する．
#            各共通時刻に最も近いセンサ値を採用する．
#            NEAREST_TOLERANCE より遠いセンサ値は採用しない．
#     注意:
#       ベクトル平均への変更は行わない．
#       通常手法の左右平均は従来通り角度の単純平均を用いる．
#       寄与率重み付き加速度PCA法は，元コード通り qx，qy の平均を用いる．

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import japanize_matplotlib


# =========================================================
# 0) ユーザー設定
# =========================================================
file_L = r'260518栁澤共同研究/YwL4.csv'
file_R = r'260518栁澤共同研究/YwR4.csv'
delay_time = 2.986508
cod_times = [18.56, 23.31, 32.74]
limit_min_time = 9.98
limit_max_time = 36.85

# 各区間の真値進行方向 [deg]
# 現在のコードでは，1回曲がった後を 90° -> 0° としていたため，
# その座標系に合わせて左折1回あたり -90° としている
# 区間: 開始～1回目，1回目～2回目，2回目～3回目，3回目以降
true_headings = [90.0, -180.0, -90.0, .0]

# 指定した範囲の誤差統計
# eval_min_time == eval_max_time の場合は実行しない
eval_min_time = 0
eval_max_time = 0

WINDOW_SIZE = 40

is_mask = 1       # 時系列プロット表示 1: 範囲内のみ表示，1以外: 全体表示
is_mask_g = 1     # 角度表示 1: [-180，180) に変換，1以外: 指定なし


# =========================================================
# センサデータ整合設定
# =========================================================
SAMPLE_INTERVAL = 0.02
NEAREST_TOLERANCE = SAMPLE_INTERVAL / 1.0

# 各ファイル内で0秒基準に使うセンサ
TIME_ZERO_SENSORS = ['Lacc', 'Gyro', 'GameRo']

# 同期状況を表示するか
PRINT_SYNC_DIAGNOSTICS = True


# =========================================================
# 実行する手法の選択
# =========================================================
DO_ACC_PCA = True              # 加速度PCA法
DO_PROPOSED_ACC_PCA = True     # 寄与率重み付き加速度PCA法
DO_GYRO_INTEGRAL = True        # 角速度累積法
DO_GYRO_PCA = False             # 角速度PCA法


# =========================================================
# 出力する処理の選択
# =========================================================
DO_RMSE = True                 # RMSE比較表
DO_ERROR_STATS = True          # 指定区間の絶対誤差統計
DO_PCA_RATIO_PLOT = False       # PCA寄与率プロット
DO_PCA_RATIO_PRINT = True      # PCA寄与率の統計表示


# =========================================================
# 時系列プロットの選択
# =========================================================
PLOT_ACC_PCA = True            # 加速度PCA法
PLOT_PROPOSED_ACC_PCA = True   # 寄与率重み付き加速度PCA法
PLOT_GYRO_INTEGRAL = True      # 角速度累積法
PLOT_GYRO_PCA = True           # 角速度PCA法


# =========================================================
# 使用するセンサを手法設定から決める処理
# =========================================================
def get_required_sensors():
    required = set()

    if DO_ACC_PCA or DO_PROPOSED_ACC_PCA:
        required.add('Lacc')
        required.add('GameRo')

    if DO_GYRO_INTEGRAL or DO_GYRO_PCA:
        required.add('Gyro')
        required.add('GameRo')

    sensor_order = ['Lacc', 'Gyro', 'GameRo']
    required_sensors = [s for s in sensor_order if s in required]

    if len(required_sensors) == 0:
        raise ValueError('実行する手法がすべて False です．少なくとも1つの手法を True にしてください．')

    return required_sensors


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
# 方向変換時刻と真値角度設定を確認する処理
# =========================================================
def validate_true_heading_settings():
    if len(cod_times) == 0:
        raise ValueError('cod_times が空です．方向変換時刻を1つ以上設定してください．')

    if len(true_headings) != len(cod_times) + 1:
        raise ValueError(
            'true_headings の数は cod_times の数 + 1 にしてください．'
            f' 現在: cod_times={len(cod_times)}個，true_headings={len(true_headings)}個'
        )

    cod_times_array = np.asarray(cod_times, dtype=float)

    if np.any(np.diff(cod_times_array) <= 0):
        raise ValueError('cod_times は小さい時刻から大きい時刻の順に設定してください．')


# =========================================================
# 任意の時刻配列に対応する真値進行方向を作る処理
# =========================================================
def make_true_heading(time_s):
    validate_true_heading_settings()

    t = np.asarray(time_s, dtype=float)
    cod_times_array = np.asarray(cod_times, dtype=float)
    true_headings_array = np.asarray(true_headings, dtype=float)

    # side='left' により，t == cod_times の点は曲がる前の区間に含める
    # これは元コードの t <= cod_time と同じ扱い
    segment_index = np.searchsorted(cod_times_array, t, side='left')

    return true_headings_array[segment_index]


# =========================================================
# グラフに方向変換時刻の縦線を追加する処理
# =========================================================
def add_cod_time_lines_to_plot():
    for i, ct in enumerate(cod_times):
        label = '方向変換時刻' if i == 0 else None
        plt.axvline(ct, c='gray', linestyle=':', alpha=0.7, label=label)


# =========================================================
# 四元数の共役
# =========================================================
def kyoyaku(gx, gy, gz, gw):
    return (-gx, -gy, -gz, gw)


# =========================================================
# 初期姿勢を基準にした相対四元数を計算する処理
# =========================================================
def calc_relative_quaternion(gx, gy, gz, gw, initial_quat):
    gx0, gy0, gz0, gw0 = initial_quat

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

    return gwc, gxc, gyc, gzc


# =========================================================
# ベクトルを初期姿勢基準の座標系へ回転する処理
# =========================================================
def rotate_xyz_by_gamero(x, y, z, gx, gy, gz, gw, initial_quat):
    gwc, gxc, gyc, gzc = calc_relative_quaternion(
        gx, gy, gz, gw,
        initial_quat
    )

    Xr = (2*gwc*gwc + 2*gxc*gxc - 1)*x + (2*gxc*gyc - 2*gzc*gwc)*y + (2*gxc*gzc + 2*gyc*gwc)*z
    Yr = (2*gxc*gyc + 2*gzc*gwc)*x + (2*gwc*gwc + 2*gyc*gyc - 1)*y + (2*gyc*gzc - 2*gxc*gwc)*z
    Zr = (2*gxc*gzc - 2*gyc*gwc)*x + (2*gyc*gzc + 2*gxc*gwc)*y + (2*gwc*gwc + 2*gzc*gzc - 1)*z

    return Xr, Yr, Zr


# =========================================================
# CSVを読み込み，ファイル内相対時刻 time_s を作る処理
# 左端末の場合は time_s に delay_time を加える
# =========================================================
def load_sensor_file_with_time(file_path, time_offset_s=0.0):
    df = pd.read_csv(file_path)

    zero_df = df[df['Sensor'].isin(TIME_ZERO_SENSORS)].copy()
    if len(zero_df) == 0:
        raise ValueError(f'{file_path} に {TIME_ZERO_SENSORS} のいずれも存在しません．')

    time0_ns = zero_df['Timestamp'].min()
    df['time_s'] = (df['Timestamp'] - time0_ns) / 1_000_000_000.0 + time_offset_s

    gamerot_df = df[df['Sensor'] == 'GameRo'].sort_values('Timestamp').reset_index(drop=True)
    if len(gamerot_df) == 0:
        raise ValueError(f'{file_path} に GameRo が存在しません．')

    initial_quat = gamerot_df.iloc[0][['X', 'Y', 'Z', 'W']].astype(float).to_numpy()

    return df, initial_quat


# =========================================================
# 指定センサの時刻範囲を取得する処理
# =========================================================
def get_sensor_time_range(df, sensor_name):
    sensor_df = df[df['Sensor'] == sensor_name].copy()

    if len(sensor_df) == 0:
        raise ValueError(f'{sensor_name} がファイル内に存在しません．')

    return sensor_df['time_s'].min(), sensor_df['time_s'].max()


# =========================================================
# 左右両方に必要センサが存在する重複区間から共通時刻グリッドを作る処理
# =========================================================
def make_common_grid(df_L, df_R, required_sensors):
    starts = []
    ends = []

    for df in [df_L, df_R]:
        for sensor_name in required_sensors:
            sensor_start, sensor_end = get_sensor_time_range(df, sensor_name)
            starts.append(sensor_start)
            ends.append(sensor_end)

    overlap_start = max(starts)
    overlap_end = min(ends)

    if overlap_start >= overlap_end:
        raise ValueError('左右の重複区間がありません．delay_time やファイルを確認してください．')

    n_grid = int(np.floor((overlap_end - overlap_start) / SAMPLE_INTERVAL)) + 1
    grid_time = overlap_start + np.arange(n_grid) * SAMPLE_INTERVAL

    grid_df = pd.DataFrame({
        'time_s': grid_time
    })

    return grid_df, overlap_start, overlap_end


# =========================================================
# 1つのセンサを共通時刻グリッドへ最近傍割当する処理
# =========================================================
def nearest_sensor_to_grid(df, sensor_name, grid_df):
    sensor_df = df[df['Sensor'] == sensor_name].copy().sort_values('time_s').reset_index(drop=True)

    if len(sensor_df) == 0:
        raise ValueError(f'{sensor_name} が存在しません．')

    value_cols = [c for c in ['X', 'Y', 'Z', 'W'] if c in sensor_df.columns]

    sensor_df = sensor_df[['time_s'] + value_cols].copy()
    sensor_df[f'{sensor_name}_source_time_s'] = sensor_df['time_s']

    matched = pd.merge_asof(
        grid_df.sort_values('time_s'),
        sensor_df.sort_values('time_s'),
        on='time_s',
        direction='backward',
        tolerance=NEAREST_TOLERANCE
    )

    rename_dict = {c: f'{sensor_name}_{c}' for c in value_cols}
    matched = matched.rename(columns=rename_dict)

    matched[f'{sensor_name}_dt_s'] = np.abs(
        matched['time_s'] - matched[f'{sensor_name}_source_time_s']
    )

    use_cols = ['time_s'] + list(rename_dict.values()) + [
        f'{sensor_name}_source_time_s',
        f'{sensor_name}_dt_s'
    ]

    return matched[use_cols]


# =========================================================
# 片側端末の全必要センサを共通時刻グリッドへ割り当てる処理
# =========================================================
def build_synced_side(df, grid_df, required_sensors):
    synced = grid_df.copy()

    for sensor_name in required_sensors:
        matched = nearest_sensor_to_grid(df, sensor_name, grid_df)
        synced = pd.merge(synced, matched, on='time_s', how='left')

    return synced


# =========================================================
# 同期状況を表示する処理
# =========================================================
def print_sync_diagnostics(sync_df, side_label, required_sensors):
    print(f"\n=== {side_label}端末の共通グリッド同期状況 ===")
    print(f"共通グリッド点数: {len(sync_df)}")

    for sensor_name in required_sensors:
        dt_col = f'{sensor_name}_dt_s'

        if dt_col not in sync_df.columns:
            print(f"{sensor_name}: dt列がありません．")
            continue

        valid_count = sync_df[dt_col].notna().sum()
        missing_count = len(sync_df) - valid_count
        max_dt = sync_df[dt_col].max()
        mean_dt = sync_df[dt_col].mean()

        print(f"{sensor_name}: 採用 {valid_count} 点，欠損 {missing_count} 点，平均ずれ {mean_dt:.6f} s，最大ずれ {max_dt:.6f} s")


# =========================================================
# 左右CSVを共通時刻グリッドに同期する処理
# =========================================================
def prepare_synchronized_sensor_data():
    required_sensors = get_required_sensors()

    df_L_raw, initial_quat_L = load_sensor_file_with_time(
        file_L,
        time_offset_s=delay_time
    )

    df_R_raw, initial_quat_R = load_sensor_file_with_time(
        file_R,
        time_offset_s=0.0
    )

    grid_df, overlap_start, overlap_end = make_common_grid(
        df_L_raw,
        df_R_raw,
        required_sensors
    )

    sync_L = build_synced_side(df_L_raw, grid_df, required_sensors)
    sync_R = build_synced_side(df_R_raw, grid_df, required_sensors)

    if PRINT_SYNC_DIAGNOSTICS:
        print_sync_diagnostics(sync_L, '左', required_sensors)
        print_sync_diagnostics(sync_R, '右', required_sensors)

    return {
        'L': sync_L,
        'R': sync_R,
        'initial_quat_L': initial_quat_L,
        'initial_quat_R': initial_quat_R,
        'required_sensors': required_sensors
    }


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
    heading_R = heading_R.copy().sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.copy().sort_values('time_s').reset_index(drop=True)

    aligned = pd.merge(
        heading_R,
        heading_L,
        on='time_s',
        how='inner',
        suffixes=('_R', '_L')
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

    true_heading_eval = make_true_heading(t_eval)

    err_R = angle_diff_pm180(theta_R_eval, true_heading_eval)
    err_L = angle_diff_pm180(theta_L_eval, true_heading_eval)
    err_mean = angle_diff_pm180(theta_mean_eval, true_heading_eval)

    RMS_R_deg = np.sqrt(np.mean(err_R**2))
    RMS_L_deg = np.sqrt(np.mean(err_L**2))
    RMS_mean_deg = np.sqrt(np.mean(err_mean**2))

    return RMS_R_deg, RMS_L_deg, RMS_mean_deg


# =========================================================
# 寄与率重み付き加速度PCA法用のRMSE計算
# 元コード通り，qx，qy の平均から左右平均角度を求める
# =========================================================
def calc_rms_from_weighted_vectors(heading_R, heading_L):
    heading_R = heading_R.copy().sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.copy().sort_values('time_s').reset_index(drop=True)

    aligned = pd.merge(
        heading_R,
        heading_L,
        on='time_s',
        how='inner',
        suffixes=('_R', '_L')
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

    true_heading_eval = make_true_heading(t_eval)

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

    aligned = pd.merge(
        heading_R,
        heading_L,
        on='time_s',
        how='inner',
        suffixes=('_R', '_L')
    )

    aligned = aligned.dropna(subset=['theta_deg_R', 'theta_deg_L']).reset_index(drop=True)

    if len(aligned) == 0:
        print(f"\n=== {method_name}（左右平均，{eval_min_time:.2f}s～{eval_max_time:.2f}s）===")
        print("時刻同期できるデータがありません．")
        return

    theta_mean = (aligned['theta_deg_R'] + aligned['theta_deg_L']) / 2.0

    t_all = aligned['time_s'].to_numpy()
    mask = (t_all >= eval_min_time) & (t_all <= eval_max_time)

    t_eval = t_all[mask]
    theta_mean_eval = theta_mean.loc[mask].to_numpy()

    if len(t_eval) == 0:
        print(f"\n=== {method_name}（左右平均，{eval_min_time:.2f}s～{eval_max_time:.2f}s）===")
        print("指定区間に一致するデータがありません．")
        return

    true_heading_eval = make_true_heading(t_eval)

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
# 元コード通り，qx，qy の平均から左右平均角度を求める
# =========================================================
def print_error_stats_from_weighted_vectors(heading_R, heading_L, method_name,
                                            eval_min_time, eval_max_time):
    heading_R = heading_R.copy().sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.copy().sort_values('time_s').reset_index(drop=True)

    aligned = pd.merge(
        heading_R,
        heading_L,
        on='time_s',
        how='inner',
        suffixes=('_R', '_L')
    )

    aligned = aligned.dropna(
        subset=['theta_deg_R', 'theta_deg_L', 'qx_R', 'qy_R', 'qx_L', 'qy_L']
    ).reset_index(drop=True)

    if len(aligned) == 0:
        print(f"\n=== {method_name}（左右平均，{eval_min_time:.2f}s～{eval_max_time:.2f}s）===")
        print("時刻同期できるデータがありません．")
        return

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

    true_heading_eval = make_true_heading(t_eval)

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

    aligned = pd.merge(
        heading_R,
        heading_L,
        on='time_s',
        how='inner',
        suffixes=('_R', '_L')
    )

    aligned = aligned.dropna(subset=['theta_deg_R', 'theta_deg_L']).reset_index(drop=True)

    if len(aligned) == 0:
        print(f"{title_str}: プロットできるデータがありません．")
        return

    t_plot = aligned['time_s'].to_numpy()
    theta_R_plot = aligned['theta_deg_R'].to_numpy()
    theta_L_plot = aligned['theta_deg_L'].to_numpy()
    theta_mean_plot = (theta_R_plot + theta_L_plot) / 2.0
    true_heading_all = make_true_heading(t_plot)

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
    #plt.scatter(t_plot, theta_L_plot, label='左手の端末', c='b', s=12, alpha=0.8)
    #plt.scatter(t_plot, theta_R_plot, label='右手の端末', c='r', s=12, alpha=0.8)
    plt.scatter(t_plot, theta_mean_plot, label='左右平均', c='g', s=8, alpha=0.8)
    plt.plot(t_plot, true_heading_all, label='真値', c='k', linestyle='--', alpha=0.8)
    add_cod_time_lines_to_plot()

    plt.xlabel('時間 [s]')
    plt.ylabel(ylabel_str)
    plt.title(title_str)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 寄与率重み付き加速度PCA法用の時系列プロット
# 元コード通り，qx，qy の平均から左右平均角度を求める
# =========================================================
def plot_heading_timeseries_weighted_vectors(heading_R, heading_L, title_str, ylabel_str):
    heading_R = heading_R.sort_values('time_s').reset_index(drop=True).copy()
    heading_L = heading_L.sort_values('time_s').reset_index(drop=True).copy()

    aligned = pd.merge(
        heading_R,
        heading_L,
        on='time_s',
        how='inner',
        suffixes=('_R', '_L')
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

    true_heading_all = make_true_heading(t_plot)

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
    #plt.scatter(t_plot, theta_L_plot, label='左手の端末', c='b', s=12, alpha=0.8)
    #plt.scatter(t_plot, theta_R_plot, label='右手の端末', c='r', s=12, alpha=0.8)
    plt.scatter(t_plot, theta_mean_plot, label='左右平均', c='g', s=8, alpha=0.8)
    plt.plot(t_plot, true_heading_all, label='真値', c='k', linestyle='--', alpha=0.8)
    add_cod_time_lines_to_plot()

    plt.xlabel('時間 [s]')
    plt.ylabel(ylabel_str)
    plt.title(title_str)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 加速度PCA法の共通処理
# use_ratio_weight = False: 通常の加速度PCA法
# use_ratio_weight = True : 寄与率重み付き加速度PCA法
# =========================================================
def compute_acc_pca_core(sync_df, initial_quat, window_size=40, use_ratio_weight=False):
    required_cols = [
        'time_s',
        'Lacc_X', 'Lacc_Y', 'Lacc_Z',
        'GameRo_X', 'GameRo_Y', 'GameRo_Z', 'GameRo_W'
    ]
    
    data = sync_df.dropna(subset=required_cols).copy().reset_index(drop=True)
   
    if len(data) < window_size:
        if use_ratio_weight:
            return pd.DataFrame(columns=[
                'time_s', 'theta_deg', 'qx', 'qy',
                'theta1_deg', 'theta2_deg', 'pc1_ratio', 'pc2_ratio'
            ])
        else:
            return pd.DataFrame(columns=[
                'time_s', 'theta_deg', 'pc1_ratio', 'pc2_ratio'
            ])

    ax = data['Lacc_X'].to_numpy()
    ay = data['Lacc_Y'].to_numpy()
    az = data['Lacc_Z'].to_numpy()

    gx = data['GameRo_X'].to_numpy()
    gy = data['GameRo_Y'].to_numpy()
    gz = data['GameRo_Z'].to_numpy()
    gw = data['GameRo_W'].to_numpy()

    Ax, Ay, Az = rotate_xyz_by_gamero(
        ax, ay, az,
        gx, gy, gz, gw,
        initial_quat
    )

    time_data = data['time_s'].to_numpy()

    x_rotated = Ax
    y_rotated = Ay

    pca = PCA(n_components=2)

    time_list = []
    theta_deg_list = []
    pc1_ratio_list = []
    pc2_ratio_list = []

    qx_list = []
    qy_list = []
    theta1_deg_list = []
    theta2_deg_list = []

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
        theta1_deg = np.degrees(theta1_rad)

        pc1_ratio = pca.explained_variance_ratio_[0]
        pc2_ratio = pca.explained_variance_ratio_[1]

        if use_ratio_weight:
            theta2_rad = theta1_rad - np.pi / 2.0

            qx = pc1_ratio * np.cos(theta1_rad) + pc2_ratio * np.cos(theta2_rad)
            qy = pc1_ratio * np.sin(theta1_rad) + pc2_ratio * np.sin(theta2_rad)

            theta_deg = np.degrees(np.arctan2(qy, qx))
            theta2_deg = np.degrees(theta2_rad)

            qx_list.append(qx)
            qy_list.append(qy)
            theta1_deg_list.append(theta1_deg)
            theta2_deg_list.append(theta2_deg)

        else:
            theta_deg = theta1_deg

        time_list.append(time_data[frame])
        theta_deg_list.append(theta_deg)
        pc1_ratio_list.append(pc1_ratio)
        pc2_ratio_list.append(pc2_ratio)

    if use_ratio_weight:
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
    else:
        heading_df = pd.DataFrame({
            'time_s': time_list,
            'theta_deg': theta_deg_list,
            'pc1_ratio': pc1_ratio_list,
            'pc2_ratio': pc2_ratio_list
        })

    return heading_df


# =========================================================
# 加速度PCA法
# =========================================================
def compute_heading_acc_pca_from_synced(sync_df, initial_quat, window_size=40):
    return compute_acc_pca_core(
        sync_df,
        initial_quat,
        window_size=window_size,
        use_ratio_weight=False
    )


# =========================================================
# 寄与率重み付き加速度PCA法
# 添付コードの計算式をそのまま使用
# =========================================================
def compute_heading_acc_pca_proposed_from_synced(sync_df, initial_quat, window_size=40):
    return compute_acc_pca_core(
        sync_df,
        initial_quat,
        window_size=window_size,
        use_ratio_weight=True
    )


# =========================================================
# 角速度累積法
# =========================================================
def compute_heading_gyro_integral_from_synced(sync_df, initial_quat, initial_heading_deg):
    required_cols = [
        'time_s',
        'Gyro_X', 'Gyro_Y', 'Gyro_Z',
        'GameRo_X', 'GameRo_Y', 'GameRo_Z', 'GameRo_W'
    ]

    data = sync_df.dropna(subset=required_cols).copy().reset_index(drop=True)

    if len(data) == 0:
        return pd.DataFrame(columns=['time_s', 'theta_deg'])

    gyx = data['Gyro_X'].to_numpy()
    gyy = data['Gyro_Y'].to_numpy()
    gyz = data['Gyro_Z'].to_numpy()

    gx = data['GameRo_X'].to_numpy()
    gy = data['GameRo_Y'].to_numpy()
    gz = data['GameRo_Z'].to_numpy()
    gw = data['GameRo_W'].to_numpy()

    Gyx, Gyy, Gyz = rotate_xyz_by_gamero(
        gyx, gyy, gyz,
        gx, gy, gz, gw,
        initial_quat
    )

    Gyx = Gyx * 180.0 / math.pi
    Gyy = Gyy * 180.0 / math.pi
    Gyz = Gyz * 180.0 / math.pi

    t = data['time_s'].to_numpy()
    dt = np.diff(t, prepend=t[0])

    omega_z = Gyz
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
def compute_heading_gyro_pca_from_synced(sync_df, initial_quat, window_size=40):
    required_cols = [
        'time_s',
        'Gyro_X', 'Gyro_Y', 'Gyro_Z',
        'GameRo_X', 'GameRo_Y', 'GameRo_Z', 'GameRo_W'
    ]

    data = sync_df.dropna(subset=required_cols).copy().reset_index(drop=True)

    if len(data) < window_size:
        return pd.DataFrame(columns=['time_s', 'theta_deg'])

    gyx = data['Gyro_X'].to_numpy()
    gyy = data['Gyro_Y'].to_numpy()
    gyz = data['Gyro_Z'].to_numpy()

    gx = data['GameRo_X'].to_numpy()
    gy = data['GameRo_Y'].to_numpy()
    gz = data['GameRo_Z'].to_numpy()
    gw = data['GameRo_W'].to_numpy()

    Gyx, Gyy, Gyz = rotate_xyz_by_gamero(
        gyx, gyy, gyz,
        gx, gy, gz, gw,
        initial_quat
    )

    Gyx = Gyx * 180.0 / math.pi
    Gyy = Gyy * 180.0 / math.pi
    Gyz = Gyz * 180.0 / math.pi

    time_data = data['time_s'].to_numpy()

    x_rotated = Gyx
    y_rotated = Gyy

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

        time_list.append(time_data[frame])
        deg_list.append(angle_deg)

    heading_df = pd.DataFrame({
        'time_s': time_list,
        'theta_deg': deg_list
    })

    return heading_df


# =========================================================
# PCA寄与率プロット用の前処理
# すでに共通時間軸なので delay_time は加えない
# =========================================================
def prepare_pca_ratio_df_for_plot(ratio_df):
    ratio_df = ratio_df.copy().sort_values('time_s').reset_index(drop=True)

    mask = (ratio_df['time_s'] >= limit_min_time) & (ratio_df['time_s'] <= limit_max_time)
    ratio_df = ratio_df.loc[mask].reset_index(drop=True)

    return ratio_df


# =========================================================
# PCA寄与率プロット
# =========================================================
def plot_pca_contribution_separate(ratio_df, hand_label):
    ratio_plot = prepare_pca_ratio_df_for_plot(ratio_df)

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
# 1) 左右センサ値を共通時刻グリッドへ同期
# =========================================================
validate_true_heading_settings()

sync_data = prepare_synchronized_sensor_data()

sync_L = sync_data['L']
sync_R = sync_data['R']

initial_quat_L = sync_data['initial_quat_L']
initial_quat_R = sync_data['initial_quat_R']


# =========================================================
# 2) 各手法の計算
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
    heading_L_pca = compute_heading_acc_pca_from_synced(
        sync_L,
        initial_quat_L,
        window_size=WINDOW_SIZE
    )

    heading_R_pca = compute_heading_acc_pca_from_synced(
        sync_R,
        initial_quat_R,
        window_size=WINDOW_SIZE
    )


if DO_PROPOSED_ACC_PCA:
    heading_L_prop = compute_heading_acc_pca_proposed_from_synced(
        sync_L,
        initial_quat_L,
        window_size=WINDOW_SIZE
    )

    heading_R_prop = compute_heading_acc_pca_proposed_from_synced(
        sync_R,
        initial_quat_R,
        window_size=WINDOW_SIZE
    )


if DO_GYRO_INTEGRAL:
    heading_L_gyro = compute_heading_gyro_integral_from_synced(
        sync_L,
        initial_quat_L,
        initial_heading_deg=90.0
    )

    heading_R_gyro = compute_heading_gyro_integral_from_synced(
        sync_R,
        initial_quat_R,
        initial_heading_deg=90.0
    )


if DO_GYRO_PCA:
    heading_L_gyro_pca = compute_heading_gyro_pca_from_synced(
        sync_L,
        initial_quat_L,
        window_size=WINDOW_SIZE
    )

    heading_R_gyro_pca = compute_heading_gyro_pca_from_synced(
        sync_R,
        initial_quat_R,
        window_size=WINDOW_SIZE
    )


# =========================================================
# 3) RMSE比較
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
# 4) 時系列プロット
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
# 5) 指定区間における左右平均絶対誤差の統計表示
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
# 6) PCA寄与率プロット
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
            hand_label='左手の端末'
        )

        plot_pca_contribution_separate(
            ratio_R_source,
            hand_label='右手の端末'
        )
    else:
        print("\nPCA寄与率プロットは，加速度PCA系の手法がOFFのためスキップしました．")


# =========================================================
# 7) PCA寄与率の統計表示
# =========================================================
if DO_PCA_RATIO_PRINT:
    if ratio_L_source is not None and ratio_R_source is not None:
        ratio_L = prepare_pca_ratio_df_for_plot(ratio_L_source)
        ratio_R = prepare_pca_ratio_df_for_plot(ratio_R_source)

        print_pca_summary(ratio_L, "左手")
        print_pca_summary(ratio_R, "右手")
    else:
        print("\nPCA寄与率の統計表示は，加速度PCA系の手法がOFFのためスキップしました．")
