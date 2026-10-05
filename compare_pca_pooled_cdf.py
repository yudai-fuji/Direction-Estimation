# 研究会用
# 4試行の全誤差を3手法に分けて累積分布関数

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


# =========================================================
# 3手法で共通の計算設定（参照元と同じ値）
# =========================================================
WINDOW_SIZE = 40
PCA_WINDOW_MAX = 50
PCA_WINDOW_MIN = 30
PCA_PC1_LOW_THRESHOLD = 0.75
PCA_PC1_HIGH_THRESHOLD = 0.875
WINDOW_RECOVERY_STEP = 1
WINDOW_RECOVERY_INTERVAL = 5

# False: 左右共通型、True: 左右独立型。
USE_INDEPENDENT_PREVIOUS_PC1_WINDOW = False

SAMPLE_INTERVAL = 0.02
NEAREST_TOLERANCE = SAMPLE_INTERVAL
TIME_ZERO_SENSORS = ['Lacc', 'Gyro', 'GameRo']
is_mask_g = 1
PRINT_SYNC_DIAGNOSTICS = False
APPLY_PCA_STANDARDIZATION = False
APPLY_ACC_LOWPASS = False
ACC_LOWPASS_ALPHA = 0.2

COLOR_FIXED = '#0072B2'
COLOR_THRESHOLD = '#ff7f00'
COLOR_PREVIOUS = '#4daf4a'


# =========================================================
# 4試行の設定
# =========================================================
# limit_min_time / limit_max_time はセンサ時刻 [s]。
# 下記では「動画の評価時刻 + video_to_sensor_offset」で記載する。
# オフセットを変更する場合は、この2つの式の加算値も更新する。
# turn_start_times_video / turn_end_times_video は動画時刻 [s]。
# 方向転換のセンサ時刻は、試行処理時にオフセット加算・小数3桁丸めで作る。
# turn_internal_headings は参照元の初期値。各試行で個別に編集できる。
TRIALS = [
    {
        'name': 'K1',
        'file_L': r'260630香川共同研究/KL1.csv',
        'file_R': r'260630香川共同研究/KR1.csv',
        'delay_time': 2.055392,
        'video_to_sensor_offset': 10.704,
        'limit_min_time': 3.51 + 10.704,
        'limit_max_time': 50.60 + 10.704,
        'true_headings': [90.0, -180.0, 90.0, 0.0, -90.0, -180.0, -90.0, 0.0],
        'turn_start_times_video': [12.63, 17.07, 22.21, 26.79, 31.86, 36.44, 46.00],
        'turn_end_times_video': [13.73, 18.27, 23.32, 27.86, 33.05, 37.57, 47.26],
        'turn_internal_headings': [
            (90.0, 180.0),
            (-180.0, -270.0),
            (90.0, 0.0),
            (0.0, -90.0),
            (-90.0, -180.0),
            (-180.0, -90.0),
            (-90.0, 0.0)
        ]
    },
    {
        'name': 'K2',
        'file_L': r'260630香川共同研究/KL2.csv',
        'file_R': r'260630香川共同研究/KR2.csv',
        'delay_time': 2.243022,
        'video_to_sensor_offset': 10.451,
        'limit_min_time': 3.00 + 10.451,
        'limit_max_time': 49.52 + 10.451,
        'true_headings': [90.0, -180.0, 90.0, 0.0, -90.0, -180.0, -90.0, 0.0],
        'turn_start_times_video': [11.40, 15.88, 20.81, 25.32, 30.38, 34.87, 44.43],
        'turn_end_times_video': [12.51, 16.96, 21.96, 26.45, 31.54, 36.00, 45.61],
        'turn_internal_headings': [
            (90.0, 180.0),
            (-180.0, -270.0),
            (90.0, 0.0),
            (0.0, -90.0),
            (-90.0, -180.0),
            (-180.0, -90.0),
            (-90.0, 0.0)
        ]
    },
    {
        'name': 'M1',
        'file_L': r'260630香川共同研究/ML1.csv',
        'file_R': r'260630香川共同研究/MR1.csv',
        'delay_time': 3.387520,
        'video_to_sensor_offset': 13.527,
        'limit_min_time': 3.68 + 13.527,
        'limit_max_time': 52.08 + 13.527,
        'true_headings': [90.0, -180.0, 90.0, 0.0, -90.0, -180.0, -90.0, 0.0],
        'turn_start_times_video': [12.58, 17.44, 22.78, 28.04, 32.77, 37.49, 46.81],
        'turn_end_times_video': [13.71, 18.58, 23.89, 29.17, 33.82, 38.58, 47.97],
        'turn_internal_headings': [
            (90.0, 180.0),
            (-180.0, -270.0),
            (90.0, 0.0),
            (0.0, -90.0),
            (-90.0, -180.0),
            (-180.0, -90.0),
            (-90.0, 0.0)
        ]
    },
    {
        'name': 'M2',
        'file_L': r'260630香川共同研究/ML2.csv',
        'file_R': r'260630香川共同研究/MR2.csv',
        'delay_time': 3.513334,
        'video_to_sensor_offset': 12.458,
        'limit_min_time': 2.88 + 12.458,
        'limit_max_time': 48.30 + 12.458,
        'true_headings': [90.0, -180.0, 90.0, 0.0, -90.0, -180.0, -90.0, 0.0],
        'turn_start_times_video': [10.79, 15.20, 20.59, 24.87, 29.66, 34.01, 43.20],
        'turn_end_times_video': [11.96, 16.31, 21.72, 25.98, 30.82, 35.16, 44.32],
        'turn_internal_headings': [
            (90.0, 180.0),
            (-180.0, -270.0),
            (90.0, 0.0),
            (0.0, -90.0),
            (-90.0, -180.0),
            (-180.0, -90.0),
            (-90.0, 0.0)
        ]
    }
]


# =========================================================
# 角度・標準化・ローパスフィルタ
# =========================================================
def wrap_pm180(theta_deg):
    if is_mask_g == 1:
        theta_deg = np.asarray(theta_deg, dtype=float)
        return np.mod(theta_deg + 180.0, 360.0) - 180.0

    return theta_deg


def angle_diff_pm180(pred_deg, true_deg):
    pred_deg = np.asarray(pred_deg, dtype=float)
    true_deg = np.asarray(true_deg, dtype=float)
    return np.mod((pred_deg - true_deg) + 180.0, 360.0) - 180.0


def standardize_pca_window(data_window):
    data_window = np.asarray(data_window, dtype=float)

    mean = np.mean(data_window, axis=0)
    std = np.std(data_window, axis=0, ddof=0)

    std_safe = np.where(std > 1e-12, std, 1.0)

    return (data_window - mean) / std_safe


def mean_two_headings_by_vector(theta_R_deg, theta_L_deg):
    theta_R_deg = np.asarray(theta_R_deg, dtype=float)
    theta_L_deg = np.asarray(theta_L_deg, dtype=float)

    theta_R_rad = np.radians(theta_R_deg)
    theta_L_rad = np.radians(theta_L_deg)

    x_mean = (np.cos(theta_R_rad) + np.cos(theta_L_rad)) / 2.0
    y_mean = (np.sin(theta_R_rad) + np.sin(theta_L_rad)) / 2.0

    return np.degrees(np.arctan2(y_mean, x_mean))


def exponential_lowpass(values, alpha):
    if not (0.0 < alpha <= 1.0):
        raise ValueError('ACC_LOWPASS_ALPHA は 0 より大きく 1 以下にしてください。')

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    filtered = np.empty_like(values)
    filtered[0] = values[0]

    for i in range(1, len(values)):
        filtered[i] = filtered[i - 1] + alpha * (values[i] - filtered[i - 1])

    return filtered


# =========================================================
# 試行ごとの真値生成
# =========================================================
def validate_true_heading_settings(trial):
    true_headings = trial['true_headings']
    turn_internal_headings = trial['turn_internal_headings']
    turn_start_times = trial['turn_start_times']
    turn_end_times = trial['turn_end_times']
    limit_min_time = trial['limit_min_time']
    limit_max_time = trial['limit_max_time']

    if len(true_headings) != 8:
        raise ValueError(
            'true_headings には8回分の直進歩行方向を設定してください。'
        )

    if len(turn_internal_headings) != 7:
        raise ValueError(
            'turn_internal_headings には7回分の方向転換前後の角度ペアを設定してください。'
        )

    if len(turn_start_times) != 7 or len(turn_end_times) != 7:
        raise ValueError(
            'turn_start_times と turn_end_times には7個ずつ時刻を設定してください。'
        )

    start = np.asarray(turn_start_times, dtype=float)
    end = np.asarray(turn_end_times, dtype=float)

    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError('turn_start_times と turn_end_times には有限の数値を設定してください。')

    if limit_min_time >= limit_max_time:
        raise ValueError('limit_min_time は limit_max_time より小さくしてください。')

    if np.any(start >= end):
        raise ValueError('各方向転換では，開始時刻を終了時刻より小さくしてください。')

    if limit_min_time >= start[0]:
        raise ValueError('limit_min_time は1回目の方向転換開始時刻より小さくしてください。')

    if np.any(end[:-1] >= start[1:]):
        raise ValueError('方向転換区間は時刻順に並べ，重ならないようにしてください。')

    if end[-1] >= limit_max_time:
        raise ValueError('7回目の方向転換終了時刻は limit_max_time より小さくしてください。')


def make_true_heading(time_s, trial):
    validate_true_heading_settings(trial)
    true_headings = trial['true_headings']
    turn_internal_headings = trial['turn_internal_headings']
    turn_start_times = trial['turn_start_times']
    turn_end_times = trial['turn_end_times']

    scalar_input = np.isscalar(time_s)
    t = np.atleast_1d(np.asarray(time_s, dtype=float))
    result = np.full(t.shape, true_headings[-1], dtype=float)

    straight_starts = [-np.inf] + list(turn_end_times)
    straight_ends = list(turn_start_times) + [np.inf]

    for i, heading in enumerate(true_headings):
        if i == len(true_headings) - 1:
            mask = (t >= straight_starts[i]) & (t <= straight_ends[i])
        else:
            mask = (t >= straight_starts[i]) & (t < straight_ends[i])
        result[mask] = heading

    for i, (turn_before, turn_after) in enumerate(turn_internal_headings):
        start = float(turn_start_times[i])
        end = float(turn_end_times[i])
        mask = (t >= start) & (t < end)
        ratio = (t[mask] - start) / (end - start)
        result[mask] = turn_before + ratio * (turn_after - turn_before)

    if scalar_input:
        return result[0]

    return result


# =========================================================
# クォータニオン・姿勢補償
# =========================================================
def kyoyaku(gx, gy, gz, gw):
    return (-gx, -gy, -gz, gw)


def calc_relative_quaternion(gx, gy, gz, gw, initial_quat):
    gx0, gy0, gz0, gw0 = initial_quat

    Gx0, Gy0, Gz0, Gw0 = kyoyaku(gx0, gy0, gz0, gw0)

    gwc = gw * Gw0 - gx * Gx0 - gy * Gy0 - gz * Gz0
    gxc = gw * Gx0 + gx * Gw0 - gy * Gz0 + gz * Gy0
    gyc = gw * Gy0 + gx * Gz0 + gy * Gw0 - gz * Gx0
    gzc = gw * Gz0 - gx * Gy0 + gy * Gx0 + gz * Gw0

    norm = np.sqrt(gwc**2 + gxc**2 + gyc**2 + gzc**2)

    gwc = gwc / norm
    gxc = gxc / norm
    gyc = gyc / norm
    gzc = gzc / norm

    return gwc, gxc, gyc, gzc


def rotate_xyz_by_gamero(x, y, z, gx, gy, gz, gw, initial_quat):
    gwc, gxc, gyc, gzc = calc_relative_quaternion(
        gx, gy, gz, gw,
        initial_quat
    )

    Xr = (
        (2 * gwc * gwc + 2 * gxc * gxc - 1) * x
        + (2 * gxc * gyc - 2 * gzc * gwc) * y
        + (2 * gxc * gzc + 2 * gyc * gwc) * z
    )

    Yr = (
        (2 * gxc * gyc + 2 * gzc * gwc) * x
        + (2 * gwc * gwc + 2 * gyc * gyc - 1) * y
        + (2 * gyc * gzc - 2 * gxc * gwc) * z
    )

    Zr = (
        (2 * gxc * gzc - 2 * gyc * gwc) * x
        + (2 * gyc * gzc + 2 * gxc * gwc) * y
        + (2 * gwc * gwc + 2 * gzc * gzc - 1) * z
    )

    return Xr, Yr, Zr


# =========================================================
# CSV読み込み・時刻同期
# =========================================================
def load_sensor_file_with_time(file_path, time_offset_s=0.0):
    df = pd.read_csv(file_path)

    zero_df = df[df['Sensor'].isin(TIME_ZERO_SENSORS)]
    if len(zero_df) == 0:
        raise ValueError(f'{file_path} に {TIME_ZERO_SENSORS} のいずれも存在しません．')

    time0_ns = zero_df['Timestamp'].min()
    df['time_s'] = (
        (df['Timestamp'] - time0_ns) / 1_000_000_000.0
        + time_offset_s
    )

    gamerot_df = (
        df[df['Sensor'] == 'GameRo']
        .sort_values('Timestamp')
        .reset_index(drop=True)
    )

    if len(gamerot_df) == 0:
        raise ValueError(f'{file_path} に GameRo が存在しません．')

    initial_quat = (
        gamerot_df
        .iloc[0][['X', 'Y', 'Z', 'W']]
        .astype(float)
        .to_numpy()
    )

    return df, initial_quat


def get_sensor_time_range(df, sensor_name):
    sensor_df = df[df['Sensor'] == sensor_name]

    if len(sensor_df) == 0:
        raise ValueError(f'{sensor_name} がファイル内に存在しません．')

    return sensor_df['time_s'].min(), sensor_df['time_s'].max()


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
        raise ValueError(
            '左右の必要センサの重複区間がありません．'
            'delay_time やCSVを確認してください．'
        )

    n_grid = int(np.floor((overlap_end - overlap_start) / SAMPLE_INTERVAL)) + 1
    grid_time = overlap_start + np.arange(n_grid) * SAMPLE_INTERVAL

    grid_df = pd.DataFrame({
        'time_s': grid_time
    })

    return grid_df, overlap_start, overlap_end


def nearest_sensor_to_grid(df, sensor_name, grid_df):
    sensor_df = (
        df[df['Sensor'] == sensor_name]
        .sort_values('time_s')
        .reset_index(drop=True)
    )

    if len(sensor_df) == 0:
        raise ValueError(f'{sensor_name} が存在しません．')

    value_cols = [c for c in ['X', 'Y', 'Z', 'W'] if c in sensor_df.columns]

    sensor_df = sensor_df[['time_s'] + value_cols].copy()
    sensor_df[f'{sensor_name}_source_time_s'] = sensor_df['time_s']

    matched = pd.merge_asof(
        grid_df.sort_values('time_s'),
        sensor_df,
        on='time_s',
        direction='backward',
        tolerance=NEAREST_TOLERANCE
    )

    rename_dict = {
        c: f'{sensor_name}_{c}'
        for c in value_cols
    }

    matched = matched.rename(columns=rename_dict)

    matched[f'{sensor_name}_dt_s'] = np.abs(
        matched['time_s'] - matched[f'{sensor_name}_source_time_s']
    )

    use_cols = (
        ['time_s']
        + list(rename_dict.values())
        + [
            f'{sensor_name}_source_time_s',
            f'{sensor_name}_dt_s'
        ]
    )

    return matched[use_cols]


def build_synced_side(df, grid_df, required_sensors):
    synced = grid_df.copy()

    for sensor_name in required_sensors:
        matched = nearest_sensor_to_grid(df, sensor_name, grid_df)
        synced = pd.merge(
            synced,
            matched,
            on='time_s',
            how='left'
        )

    return synced


def print_sync_diagnostics(sync_df, side_label, required_sensors):
    print(f'\n=== {side_label}端末の共通グリッド同期状況 ===')
    print(f'共通グリッド点数: {len(sync_df)}')

    for sensor_name in required_sensors:
        dt_col = f'{sensor_name}_dt_s'

        if dt_col not in sync_df.columns:
            print(f'{sensor_name}: dt列がありません．')
            continue

        valid_count = sync_df[dt_col].notna().sum()
        missing_count = len(sync_df) - valid_count
        mean_dt = sync_df[dt_col].mean()
        max_dt = sync_df[dt_col].max()

        print(
            f'{sensor_name}: '
            f'採用 {valid_count} 点，'
            f'欠損 {missing_count} 点，'
            f'平均ずれ {mean_dt:.6f} s，'
            f'最大ずれ {max_dt:.6f} s'
        )


def prepare_synchronized_sensor_data(trial):
    file_L = trial['file_L']
    file_R = trial['file_R']
    delay_time = trial['delay_time']

    required_sensors = ['Lacc', 'Gyro', 'GameRo']

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

    print('\n=== 共通時刻グリッド ===')
    print(f'重複開始時刻: {overlap_start:.6f} s')
    print(f'重複終了時刻: {overlap_end:.6f} s')
    print(f'グリッド点数: {len(grid_df)}')
    print(f'サンプリング間隔: {SAMPLE_INTERVAL:.6f} s')

    sync_L = build_synced_side(df_L_raw, grid_df, required_sensors)
    sync_R = build_synced_side(df_R_raw, grid_df, required_sensors)

    if PRINT_SYNC_DIAGNOSTICS:
        print_sync_diagnostics(sync_L, '左', required_sensors)
        print_sync_diagnostics(sync_R, '右', required_sensors)

    return {
        'L': sync_L,
        'R': sync_R,
        'initial_quat_L': initial_quat_L,
        'initial_quat_R': initial_quat_R
    }


# =========================================================
# 角速度累積法の基準方位
# =========================================================
def compute_heading_gyro_integral_from_synced(
    sync_df,
    initial_quat,
    initial_heading_deg
):
    required_cols = [
        'time_s',
        'Gyro_X', 'Gyro_Y', 'Gyro_Z',
        'GameRo_X', 'GameRo_Y', 'GameRo_Z', 'GameRo_W'
    ]

    data = (
        sync_df
        .dropna(subset=required_cols)
        .copy()
        .reset_index(drop=True)
    )

    if len(data) == 0:
        return pd.DataFrame(columns=['time_s', 'theta_deg'])

    gyx = data['Gyro_X'].to_numpy()
    gyy = data['Gyro_Y'].to_numpy()
    gyz = data['Gyro_Z'].to_numpy()

    gx = data['GameRo_X'].to_numpy()
    gy = data['GameRo_Y'].to_numpy()
    gz = data['GameRo_Z'].to_numpy()
    gw = data['GameRo_W'].to_numpy()

    _, _, Gyz = rotate_xyz_by_gamero(
        gyx, gyy, gyz,
        gx, gy, gz, gw,
        initial_quat
    )

    Gyz = Gyz * 180.0 / math.pi

    t = data['time_s'].to_numpy()
    dt = np.diff(t, prepend=t[0])

    dtheta = Gyz * dt
    theta = initial_heading_deg + np.cumsum(dtheta)

    result = pd.DataFrame({
        'time_s': t,
        'theta_deg': theta
    })

    return result


def make_gyro_mean_heading(gyro_heading_R, gyro_heading_L):
    if gyro_heading_R is None or gyro_heading_L is None:
        return pd.DataFrame(columns=['time_s', 'theta_deg'])

    aligned = pd.merge(
        gyro_heading_R[['time_s', 'theta_deg']],
        gyro_heading_L[['time_s', 'theta_deg']],
        on='time_s',
        how='inner',
        suffixes=('_R', '_L')
    )

    aligned = aligned.dropna(
        subset=['theta_deg_R', 'theta_deg_L']
    ).sort_values('time_s').reset_index(drop=True)

    if len(aligned) == 0:
        return pd.DataFrame(columns=['time_s', 'theta_deg'])

    aligned['theta_deg'] = (
        aligned['theta_deg_R'].to_numpy()
        + aligned['theta_deg_L'].to_numpy()
    ) / 2.0

    return aligned[['time_s', 'theta_deg']]


# =========================================================
# 水平加速度・PCA・窓幅制御・180度補正
# =========================================================
def prepare_acc_pca_data(sync_df, initial_quat):
    required_cols = [
        'time_s',
        'Lacc_X', 'Lacc_Y', 'Lacc_Z',
        'GameRo_X', 'GameRo_Y', 'GameRo_Z', 'GameRo_W'
    ]

    data = (
        sync_df
        .dropna(subset=required_cols)
        .copy()
        .reset_index(drop=True)
    )

    if len(data) == 0:
        return pd.DataFrame(columns=[
            'time_s',
            'x_rotated',
            'y_rotated'
        ])

    ax = data['Lacc_X'].to_numpy()
    ay = data['Lacc_Y'].to_numpy()
    az = data['Lacc_Z'].to_numpy()

    gx = data['GameRo_X'].to_numpy()
    gy = data['GameRo_Y'].to_numpy()
    gz = data['GameRo_Z'].to_numpy()
    gw = data['GameRo_W'].to_numpy()

    Ax, Ay, _ = rotate_xyz_by_gamero(
        ax, ay, az,
        gx, gy, gz, gw,
        initial_quat
    )

    if APPLY_ACC_LOWPASS:
        x_rotated = exponential_lowpass(Ax, ACC_LOWPASS_ALPHA)
        y_rotated = exponential_lowpass(Ay, ACC_LOWPASS_ALPHA)
    else:
        x_rotated = Ax
        y_rotated = Ay

    return pd.DataFrame({
        'time_s': data['time_s'].to_numpy(),
        'x_rotated': x_rotated,
        'y_rotated': y_rotated
    })


def empty_pc1_window_schedule_df():
    return pd.DataFrame(columns=[
        'time_s',
        'window_size',
        'pc1_ratio_L',
        'pc1_ratio_R'
    ])


def compute_pc1_ratio_for_window(x_rotated, y_rotated, frame, window_size, pca):
    if frame + 1 < window_size:
        return np.nan

    start_index = frame - (window_size - 1)
    end_index = frame + 1

    data_window = np.column_stack((
        x_rotated[start_index:end_index],
        y_rotated[start_index:end_index]
    ))

    if APPLY_PCA_STANDARDIZATION:
        data_window = standardize_pca_window(data_window)

    pca.fit(data_window)
    return float(pca.explained_variance_ratio_[0])


def update_window_size_from_pc1(current_window, recovery_count, pc1_ratio_L, pc1_ratio_R):
    finite_L = np.isfinite(pc1_ratio_L)
    finite_R = np.isfinite(pc1_ratio_R)

    low_detected = (
        (finite_L and pc1_ratio_L < PCA_PC1_LOW_THRESHOLD)
        or (finite_R and pc1_ratio_R < PCA_PC1_LOW_THRESHOLD)
    )
    high_both = (
        finite_L
        and finite_R
        and pc1_ratio_L > PCA_PC1_HIGH_THRESHOLD
        and pc1_ratio_R > PCA_PC1_HIGH_THRESHOLD
    )

    if low_detected:
        return PCA_WINDOW_MIN, 0

    if current_window >= PCA_WINDOW_MAX:
        return current_window, 0

    if high_both:
        next_window = min(
            PCA_WINDOW_MAX,
            PCA_WINDOW_MIN
            + (recovery_count // WINDOW_RECOVERY_INTERVAL)
            * WINDOW_RECOVERY_STEP
        )
        return next_window, recovery_count + 1

    return current_window, 0


def make_common_window_schedule_from_pc1(
    sync_L,
    initial_quat_L,
    sync_R,
    initial_quat_R
):
    left = prepare_acc_pca_data(sync_L, initial_quat_L)
    right = prepare_acc_pca_data(sync_R, initial_quat_R)

    if len(left) == 0 or len(right) == 0:
        return empty_pc1_window_schedule_df()

    left = (
        left
        .rename(columns={
            'x_rotated': 'x_rotated_L',
            'y_rotated': 'y_rotated_L'
        })
        .sort_values('time_s')
        .reset_index(drop=True)
    )
    right = (
        right
        .rename(columns={
            'x_rotated': 'x_rotated_R',
            'y_rotated': 'y_rotated_R'
        })
        .sort_values('time_s')
        .reset_index(drop=True)
    )

    common_data = pd.merge(
        left,
        right,
        on='time_s',
        how='inner'
    )

    if len(common_data) == 0:
        return empty_pc1_window_schedule_df()

    time_data = common_data['time_s'].to_numpy()
    x_L = common_data['x_rotated_L'].to_numpy()
    y_L = common_data['y_rotated_L'].to_numpy()
    x_R = common_data['x_rotated_R'].to_numpy()
    y_R = common_data['y_rotated_R'].to_numpy()

    pca_L = PCA(n_components=2)
    pca_R = PCA(n_components=2)

    current_window = PCA_WINDOW_MAX
    recovery_count = 0

    window_size_list = []
    pc1_ratio_L_list = []
    pc1_ratio_R_list = []

    for frame in range(len(common_data)):
        used_window = int(current_window)
        pc1_ratio_L = compute_pc1_ratio_for_window(
            x_L,
            y_L,
            frame,
            used_window,
            pca_L
        )
        pc1_ratio_R = compute_pc1_ratio_for_window(
            x_R,
            y_R,
            frame,
            used_window,
            pca_R
        )

        window_size_list.append(used_window)
        pc1_ratio_L_list.append(pc1_ratio_L)
        pc1_ratio_R_list.append(pc1_ratio_R)

        current_window, recovery_count = update_window_size_from_pc1(
            current_window,
            recovery_count,
            pc1_ratio_L,
            pc1_ratio_R
        )

    return pd.DataFrame({
        'time_s': time_data,
        'window_size': window_size_list,
        'pc1_ratio_L': pc1_ratio_L_list,
        'pc1_ratio_R': pc1_ratio_R_list
    })


def empty_previous_pc1_window_schedule_df():
    return pd.DataFrame(columns=[
        'time_s',
        'window_size',
        'pc1_ratio_L',
        'pc1_ratio_R',
        'removed_count',
        'forced_min_window'
    ])


def make_common_window_schedule_by_previous_pc1(
    sync_L,
    initial_quat_L,
    sync_R,
    initial_quat_R
):
    left = prepare_acc_pca_data(sync_L, initial_quat_L)
    right = prepare_acc_pca_data(sync_R, initial_quat_R)

    if len(left) == 0 or len(right) == 0:
        return empty_previous_pc1_window_schedule_df()

    left = (
        left
        .rename(columns={
            'x_rotated': 'x_rotated_L',
            'y_rotated': 'y_rotated_L'
        })
        .sort_values('time_s')
        .reset_index(drop=True)
    )
    right = (
        right
        .rename(columns={
            'x_rotated': 'x_rotated_R',
            'y_rotated': 'y_rotated_R'
        })
        .sort_values('time_s')
        .reset_index(drop=True)
    )

    common_data = pd.merge(
        left,
        right,
        on='time_s',
        how='inner'
    )

    if len(common_data) == 0:
        return empty_previous_pc1_window_schedule_df()

    time_data = common_data['time_s'].to_numpy()
    x_L = common_data['x_rotated_L'].to_numpy()
    y_L = common_data['y_rotated_L'].to_numpy()
    x_R = common_data['x_rotated_R'].to_numpy()
    y_R = common_data['y_rotated_R'].to_numpy()

    pca_L = PCA(n_components=2)
    pca_R = PCA(n_components=2)

    previous_window_size = None
    previous_ratio_L = np.nan
    previous_ratio_R = np.nan

    window_size_list = []
    pc1_ratio_L_list = []
    pc1_ratio_R_list = []
    removed_count_list = []
    forced_min_window_list = []

    for frame in range(len(common_data)):
        if frame + 1 < PCA_WINDOW_MIN:
            window_size_list.append(PCA_WINDOW_MIN)
            pc1_ratio_L_list.append(np.nan)
            pc1_ratio_R_list.append(np.nan)
            removed_count_list.append(0)
            forced_min_window_list.append(False)
            continue

        if previous_window_size is None:
            initial_candidate_window_size = PCA_WINDOW_MIN
        else:
            initial_candidate_window_size = min(
                previous_window_size + 1,
                PCA_WINDOW_MAX,
                frame + 1
            )

        candidate_window_size = int(initial_candidate_window_size)
        candidate_ratio_L = compute_pc1_ratio_for_window(
            x_L,
            y_L,
            frame,
            candidate_window_size,
            pca_L
        )
        candidate_ratio_R = compute_pc1_ratio_for_window(
            x_R,
            y_R,
            frame,
            candidate_window_size,
            pca_R
        )

        forced_min_window = False

        if previous_window_size is not None:
            while not (
                candidate_ratio_L >= previous_ratio_L
                and candidate_ratio_R >= previous_ratio_R
            ):
                if candidate_window_size <= PCA_WINDOW_MIN:
                    forced_min_window = True
                    break

                candidate_window_size -= 1
                candidate_ratio_L = compute_pc1_ratio_for_window(
                    x_L,
                    y_L,
                    frame,
                    candidate_window_size,
                    pca_L
                )
                candidate_ratio_R = compute_pc1_ratio_for_window(
                    x_R,
                    y_R,
                    frame,
                    candidate_window_size,
                    pca_R
                )

        removed_count = (
            initial_candidate_window_size - candidate_window_size
        )

        window_size_list.append(candidate_window_size)
        pc1_ratio_L_list.append(candidate_ratio_L)
        pc1_ratio_R_list.append(candidate_ratio_R)
        removed_count_list.append(int(removed_count))
        forced_min_window_list.append(forced_min_window)

        previous_window_size = candidate_window_size
        previous_ratio_L = candidate_ratio_L
        previous_ratio_R = candidate_ratio_R

    return pd.DataFrame({
        'time_s': time_data,
        'window_size': window_size_list,
        'pc1_ratio_L': pc1_ratio_L_list,
        'pc1_ratio_R': pc1_ratio_R_list,
        'removed_count': removed_count_list,
        'forced_min_window': forced_min_window_list
    })


def empty_independent_previous_pc1_window_schedule_df():
    return pd.DataFrame(columns=[
        'time_s',
        'window_size',
        'pc1_ratio',
        'removed_count',
        'forced_min_window'
    ])


def make_independent_window_schedule_by_previous_pc1(
    sync_df,
    initial_quat
):
    data = prepare_acc_pca_data(sync_df, initial_quat)

    if len(data) == 0:
        return empty_independent_previous_pc1_window_schedule_df()

    data = data.sort_values('time_s').reset_index(drop=True)

    time_data = data['time_s'].to_numpy()
    x_rotated = data['x_rotated'].to_numpy()
    y_rotated = data['y_rotated'].to_numpy()

    pca = PCA(n_components=2)

    previous_window_size = None
    previous_ratio = np.nan

    window_size_list = []
    pc1_ratio_list = []
    removed_count_list = []
    forced_min_window_list = []

    for frame in range(len(data)):
        if frame + 1 < PCA_WINDOW_MIN:
            window_size_list.append(PCA_WINDOW_MIN)
            pc1_ratio_list.append(np.nan)
            removed_count_list.append(0)
            forced_min_window_list.append(False)
            continue

        if previous_window_size is None:
            initial_candidate_window_size = PCA_WINDOW_MIN
        else:
            initial_candidate_window_size = min(
                previous_window_size + 1,
                PCA_WINDOW_MAX,
                frame + 1
            )

        candidate_window_size = int(initial_candidate_window_size)
        candidate_ratio = compute_pc1_ratio_for_window(
            x_rotated,
            y_rotated,
            frame,
            candidate_window_size,
            pca
        )

        forced_min_window = False

        if previous_window_size is not None:
            while not (
                candidate_ratio >= previous_ratio
            ):
                if candidate_window_size <= PCA_WINDOW_MIN:
                    forced_min_window = True
                    break

                candidate_window_size -= 1
                candidate_ratio = compute_pc1_ratio_for_window(
                    x_rotated,
                    y_rotated,
                    frame,
                    candidate_window_size,
                    pca
                )

        removed_count = (
            initial_candidate_window_size - candidate_window_size
        )

        window_size_list.append(candidate_window_size)
        pc1_ratio_list.append(candidate_ratio)
        removed_count_list.append(int(removed_count))
        forced_min_window_list.append(forced_min_window)

        previous_window_size = candidate_window_size
        previous_ratio = candidate_ratio

    return pd.DataFrame({
        'time_s': time_data,
        'window_size': window_size_list,
        'pc1_ratio': pc1_ratio_list,
        'removed_count': removed_count_list,
        'forced_min_window': forced_min_window_list
    })


def empty_acc_pca_df(use_ratio_weight):
    if use_ratio_weight:
        return pd.DataFrame(columns=[
            'time_s', 'theta_deg', 'qx', 'qy',
            'theta1_deg', 'theta2_deg',
            'pc1_ratio', 'pc2_ratio', 'window_size'
        ])

    return pd.DataFrame(columns=[
        'time_s', 'theta_deg',
        'pc1_ratio', 'pc2_ratio', 'window_size'
    ])


def attach_window_schedule(data, window_schedule_df, default_window_size):
    if window_schedule_df is None:
        data = data.copy()
        data['window_size'] = int(default_window_size)
        return data

    schedule = window_schedule_df[['time_s', 'window_size']].copy()

    data = pd.merge(
        data,
        schedule,
        on='time_s',
        how='left'
    )

    data['window_size'] = (
        data['window_size']
        .fillna(default_window_size)
        .astype(int)
        .clip(lower=PCA_WINDOW_MIN, upper=PCA_WINDOW_MAX)
    )

    return data


def compute_acc_pca_core(
    sync_df,
    initial_quat,
    window_size=WINDOW_SIZE,
    window_schedule_df=None,
    use_ratio_weight=False
):
    data = prepare_acc_pca_data(sync_df, initial_quat)

    if len(data) == 0:
        return empty_acc_pca_df(use_ratio_weight)

    data = attach_window_schedule(
        data,
        window_schedule_df,
        default_window_size=window_size
    )

    if len(data) < data['window_size'].min():
        return empty_acc_pca_df(use_ratio_weight)

    time_data = data['time_s'].to_numpy()
    window_data = data['window_size'].to_numpy(dtype=int)
    x_rotated = data['x_rotated'].to_numpy()
    y_rotated = data['y_rotated'].to_numpy()

    pca = PCA(n_components=2)

    time_list = []
    theta_deg_list = []
    pc1_ratio_list = []
    pc2_ratio_list = []
    window_size_list = []

    qx_list = []
    qy_list = []
    theta1_deg_list = []
    theta2_deg_list = []

    for frame in range(len(x_rotated)):
        current_window = int(window_data[frame])

        if frame + 1 < current_window:
            continue

        start_index = frame - (current_window - 1)
        end_index = frame + 1

        data_window = np.column_stack((
            x_rotated[start_index:end_index],
            y_rotated[start_index:end_index]
        ))

        if APPLY_PCA_STANDARDIZATION:
            data_window = standardize_pca_window(data_window)

        pca.fit(data_window)
        vx, vy = pca.components_[0]

        theta1_rad = np.arctan2(vy, vx)
        theta1_deg = np.degrees(theta1_rad)

        pc1_ratio = pca.explained_variance_ratio_[0]
        pc2_ratio = pca.explained_variance_ratio_[1]

        if use_ratio_weight:
            theta2_rad = theta1_rad - np.pi / 2.0

            qx = (
                pc1_ratio * np.cos(theta1_rad)
                + pc2_ratio * np.cos(theta2_rad)
            )
            qy = (
                pc1_ratio * np.sin(theta1_rad)
                + pc2_ratio * np.sin(theta2_rad)
            )

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
        window_size_list.append(current_window)

    if use_ratio_weight:
        heading_df = pd.DataFrame({
            'time_s': time_list,
            'theta_deg': theta_deg_list,
            'qx': qx_list,
            'qy': qy_list,
            'theta1_deg': theta1_deg_list,
            'theta2_deg': theta2_deg_list,
            'pc1_ratio': pc1_ratio_list,
            'pc2_ratio': pc2_ratio_list,
            'window_size': window_size_list
        })
    else:
        heading_df = pd.DataFrame({
            'time_s': time_list,
            'theta_deg': theta_deg_list,
            'pc1_ratio': pc1_ratio_list,
            'pc2_ratio': pc2_ratio_list,
            'window_size': window_size_list
        })

    return heading_df


def compute_heading_acc_pca_from_synced(sync_df, initial_quat, window_size=WINDOW_SIZE):
    return compute_acc_pca_core(
        sync_df,
        initial_quat,
        window_size=window_size,
        window_schedule_df=None,
        use_ratio_weight=False
    )


def compute_heading_acc_pca_variable_from_synced(
    sync_df,
    initial_quat,
    window_schedule_df,
    use_ratio_weight=False
):
    return compute_acc_pca_core(
        sync_df,
        initial_quat,
        window_size=PCA_WINDOW_MAX,
        window_schedule_df=window_schedule_df,
        use_ratio_weight=use_ratio_weight
    )


def resolve_pca_180_by_gyro(heading_df, gyro_heading_df, use_ratio_weight=False):
    if heading_df is None or len(heading_df) == 0:
        return heading_df

    if gyro_heading_df is None or len(gyro_heading_df) == 0:
        return heading_df.copy()

    heading = (
        heading_df
        .copy()
        .sort_values('time_s')
        .reset_index(drop=True)
    )
    gyro = (
        gyro_heading_df[['time_s', 'theta_deg']]
        .copy()
        .sort_values('time_s')
        .rename(columns={'theta_deg': 'gyro_theta_deg'})
        .reset_index(drop=True)
    )

    corrected = pd.merge_asof(
        heading,
        gyro,
        on='time_s',
        direction='backward',
        tolerance=NEAREST_TOLERANCE
    )

    valid_mask = corrected['gyro_theta_deg'].notna()

    theta_base = corrected.loc[valid_mask, 'theta_deg'].to_numpy()
    theta_flip = theta_base + 180.0
    theta_gyro = corrected.loc[valid_mask, 'gyro_theta_deg'].to_numpy()

    err_base = np.abs(angle_diff_pm180(theta_base, theta_gyro))
    err_flip = np.abs(angle_diff_pm180(theta_flip, theta_gyro))
    use_flip = err_flip < err_base

    valid_indices = corrected.index[valid_mask].to_numpy()
    flip_indices = valid_indices[use_flip]

    corrected.loc[flip_indices, 'theta_deg'] = (
        corrected.loc[flip_indices, 'theta_deg'] + 180.0
    )

    if use_ratio_weight:
        for col in ['qx', 'qy']:
            if col in corrected.columns:
                corrected.loc[flip_indices, col] = -corrected.loc[flip_indices, col]

        for col in ['theta1_deg', 'theta2_deg']:
            if col in corrected.columns:
                corrected.loc[flip_indices, col] = (
                    corrected.loc[flip_indices, col] + 180.0
                )
                corrected[col] = wrap_pm180(corrected[col].to_numpy())

    corrected['theta_deg'] = wrap_pm180(corrected['theta_deg'].to_numpy())
    corrected['gyro_reference_deg'] = corrected['gyro_theta_deg']
    corrected['used_180_flip'] = False
    corrected.loc[flip_indices, 'used_180_flip'] = True

    corrected = corrected.drop(columns=['gyro_theta_deg'])

    return corrected


# =========================================================
# 左右方位の時刻整合とベクトル平均
# =========================================================
def align_headings_for_section_rmse(
    heading_R,
    heading_L,
    use_weighted_mean=False,
    use_simple_mean=False
):
    heading_R = heading_R.sort_values('time_s').reset_index(drop=True)
    heading_L = heading_L.sort_values('time_s').reset_index(drop=True)

    aligned = pd.merge(
        heading_R,
        heading_L,
        on='time_s',
        how='inner',
        suffixes=('_R', '_L')
    )

    required_cols = ['theta_deg_R', 'theta_deg_L']
    if use_weighted_mean:
        required_cols.extend(['qx_R', 'qy_R', 'qx_L', 'qy_L'])

    aligned = aligned.dropna(subset=required_cols).reset_index(drop=True)

    if len(aligned) == 0:
        return aligned

    theta_R = aligned['theta_deg_R'].to_numpy()
    theta_L = aligned['theta_deg_L'].to_numpy()

    if use_weighted_mean:
        qx_mean = (
            aligned['qx_R'].to_numpy()
            + aligned['qx_L'].to_numpy()
        ) / 2.0
        qy_mean = (
            aligned['qy_R'].to_numpy()
            + aligned['qy_L'].to_numpy()
        ) / 2.0
        theta_mean = np.degrees(np.arctan2(qy_mean, qx_mean))
    elif use_simple_mean:
        theta_mean = (theta_R + theta_L) / 2.0
    else:
        theta_mean = mean_two_headings_by_vector(theta_R, theta_L)

    aligned['theta_mean_deg'] = theta_mean

    return aligned


# =========================================================
# 通常PCAの3手法
# =========================================================
def calculate_method_headings(sync_data, gyro_mean_heading):
    """3手法の (表示名, 補正済み右方位, 補正済み左方位, 色) を返す。"""
    sync_L = sync_data['L']
    sync_R = sync_data['R']
    initial_quat_L = sync_data['initial_quat_L']
    initial_quat_R = sync_data['initial_quat_R']

    # 固定窓PCA: 既存の窓幅をそのまま使用する。
    heading_L_fixed = compute_heading_acc_pca_from_synced(
        sync_L, initial_quat_L, window_size=WINDOW_SIZE
    )
    heading_R_fixed = compute_heading_acc_pca_from_synced(
        sync_R, initial_quat_R, window_size=WINDOW_SIZE
    )

    # 閾値型可変窓PCA: 左右共通の窓幅・閾値・回復処理を再利用する。
    threshold_schedule = make_common_window_schedule_from_pc1(
        sync_L, initial_quat_L, sync_R, initial_quat_R
    )
    heading_L_threshold = compute_heading_acc_pca_variable_from_synced(
        sync_L, initial_quat_L, threshold_schedule, use_ratio_weight=False
    )
    heading_R_threshold = compute_heading_acc_pca_variable_from_synced(
        sync_R, initial_quat_R, threshold_schedule, use_ratio_weight=False
    )

    # 前時刻比較型可変窓PCA: このファイルの設定だけで切り替える。
    if USE_INDEPENDENT_PREVIOUS_PC1_WINDOW:
        previous_name = '左右独立・前時刻比較型可変窓PCA'
        previous_schedule_L = (
            make_independent_window_schedule_by_previous_pc1(
                sync_L, initial_quat_L
            )
        )
        previous_schedule_R = (
            make_independent_window_schedule_by_previous_pc1(
                sync_R, initial_quat_R
            )
        )
    else:
        previous_name = '前時刻比較型可変窓PCA'
        previous_schedule = make_common_window_schedule_by_previous_pc1(
            sync_L, initial_quat_L, sync_R, initial_quat_R
        )
        previous_schedule_L = previous_schedule
        previous_schedule_R = previous_schedule

    heading_L_previous = compute_heading_acc_pca_variable_from_synced(
        sync_L, initial_quat_L, previous_schedule_L, use_ratio_weight=False
    )
    heading_R_previous = compute_heading_acc_pca_variable_from_synced(
        sync_R, initial_quat_R, previous_schedule_R, use_ratio_weight=False
    )

    method_specs = [
        ('固定窓PCA', heading_R_fixed, heading_L_fixed, COLOR_FIXED),
        (
            '閾値型可変窓PCA',
            heading_R_threshold,
            heading_L_threshold,
            COLOR_THRESHOLD
        ),
        (previous_name, heading_R_previous, heading_L_previous, COLOR_PREVIOUS)
    ]

    # 左右とも、同じ角速度累積法の平均方位を基準に180度補正する。
    corrected_methods = []
    for method_name, heading_R, heading_L, color in method_specs:
        corrected_R = resolve_pca_180_by_gyro(
            heading_R, gyro_mean_heading, use_ratio_weight=False
        )
        corrected_L = resolve_pca_180_by_gyro(
            heading_L, gyro_mean_heading, use_ratio_weight=False
        )
        corrected_methods.append((method_name, corrected_R, corrected_L, color))

    return corrected_methods


# =========================================================
# CDFの軸設定
# =========================================================
def configure_cdf_axis(ax):
    """統合CDFの表示範囲と軸設定を適用する。"""
    ax.set_xlabel('左右平均の絶対誤差 [deg]')
    ax.set_ylabel('累積確率')
    ax.set_xlim(0.0, 50.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.grid(True, which='both', linestyle='--', alpha=0.5)


# =========================================================
# 試行設定・絶対誤差・4試行の統合
# =========================================================
def prepare_trial_config(trial):
    """元の辞書を変更せず、試行条件の検査と方向転換時刻の生成を行う。"""
    if not isinstance(trial, dict):
        raise ValueError('各試行の設定は辞書で指定してください。')

    config = deepcopy(trial)
    name = config.get('name') or '名前未設定の試行'
    required_keys = (
        'name', 'file_L', 'file_R', 'delay_time', 'video_to_sensor_offset',
        'limit_min_time', 'limit_max_time', 'true_headings',
        'turn_start_times_video', 'turn_end_times_video', 'turn_internal_headings'
    )
    missing = [
        key for key in required_keys
        if config.get(key) is None
        or (isinstance(config.get(key), str) and not config[key].strip())
    ]
    if missing:
        raise ValueError(f"{name}: 未設定の項目: {', '.join(missing)}")

    for key in (
        'delay_time', 'video_to_sensor_offset', 'limit_min_time', 'limit_max_time'
    ):
        try:
            value = float(config[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f'{name}: {key} は有限の数値で指定してください。') from error
        if not np.isfinite(value):
            raise ValueError(f'{name}: {key} は有限の数値で指定してください。')
        config[key] = value

    array_shapes = {
        'true_headings': (8,),
        'turn_start_times_video': (7,),
        'turn_end_times_video': (7,),
        'turn_internal_headings': (7, 2)
    }
    for key, shape in array_shapes.items():
        try:
            values = np.asarray(config[key], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{name}: {key} の数値配列を確認してください。') from error
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(
                f'{name}: {key} は形状 {shape} の有限の数値配列で指定してください。'
            )
        config[key] = values.tolist()

    config['turn_start_times'] = [
        round(t + config['video_to_sensor_offset'], 3)
        for t in config['turn_start_times_video']
    ]
    config['turn_end_times'] = [
        round(t + config['video_to_sensor_offset'], 3)
        for t in config['turn_end_times_video']
    ]
    try:
        validate_true_heading_settings(config)
    except ValueError as error:
        raise ValueError(f'{name}: {error}') from error

    for key in ('file_L', 'file_R'):
        try:
            exists = Path(config[key]).is_file()
        except (TypeError, ValueError) as error:
            raise ValueError(f'{name}: {key} のCSVパスを確認してください。') from error
        if not exists:
            raise FileNotFoundError(f"{name}: {key} のCSVがありません: {config[key]}")

    return config


def get_method_styles():
    """計算と出力で使用する3手法の順序・表示名・色を返す。"""
    previous_name = (
        '左右独立・前時刻比較型可変窓PCA'
        if USE_INDEPENDENT_PREVIOUS_PC1_WINDOW
        else '前時刻比較型可変窓PCA'
    )
    return [
        ('固定窓PCA', COLOR_FIXED),
        ('閾値型可変窓PCA', COLOR_THRESHOLD),
        (previous_name, COLOR_PREVIOUS)
    ]


def make_absolute_error_data(heading_R, heading_L, trial):
    """評価時間内の左右ベクトル平均と真値との有限な絶対角度誤差を返す。"""
    # infを含む角度の演算で生じるNaNは、最後の有限値マスクで除外する。
    with np.errstate(invalid='ignore'):
        aligned = align_headings_for_section_rmse(
            heading_R,
            heading_L,
            use_weighted_mean=False,
            use_simple_mean=False
        )
    if aligned.empty:
        return np.array([], dtype=float)

    mask = (
        (aligned['time_s'] >= trial['limit_min_time'])
        & (aligned['time_s'] <= trial['limit_max_time'])
    )
    evaluation = aligned.loc[mask, ['time_s', 'theta_mean_deg']]
    if evaluation.empty:
        return np.array([], dtype=float)

    true_heading = make_true_heading(evaluation['time_s'].to_numpy(), trial)
    with np.errstate(invalid='ignore'):
        error_deg = angle_diff_pm180(
            evaluation['theta_mean_deg'].to_numpy(), true_heading
        )
    absolute_error_deg = np.abs(error_deg)
    return absolute_error_deg[np.isfinite(absolute_error_deg)]


def process_trial(trial):
    """1試行を計算し、3手法それぞれの絶対角度誤差配列を返す。"""
    config = prepare_trial_config(trial)
    # 同期処理の診断printのみ抑制し、計算条件・時刻整合は変更しない。
    with redirect_stdout(StringIO()):
        sync_data = prepare_synchronized_sensor_data(config)

    gyro_heading_L = compute_heading_gyro_integral_from_synced(
        sync_data['L'], sync_data['initial_quat_L'], initial_heading_deg=90.0
    )
    gyro_heading_R = compute_heading_gyro_integral_from_synced(
        sync_data['R'], sync_data['initial_quat_R'], initial_heading_deg=90.0
    )
    gyro_mean_heading = make_gyro_mean_heading(gyro_heading_R, gyro_heading_L)
    method_specs = calculate_method_headings(sync_data, gyro_mean_heading)

    errors_by_method = {}
    for method_name, heading_R, heading_L, _ in method_specs:
        errors = make_absolute_error_data(heading_R, heading_L, config)
        if len(errors) == 0:
            raise ValueError(
                f"{config['name']}: {method_name} の評価区間に有効な絶対誤差がありません。"
            )
        errors_by_method[method_name] = errors
    return errors_by_method


def pool_trial_errors(trials):
    """4試行の全絶対誤差を手法別に結合する。試行別CDFの平均は行わない。"""
    if len(trials) != 4:
        raise ValueError('TRIALSには4試行分の設定を指定してください。')

    # 全試行の設定とCSVを先に確認し、不足時に一部の試行だけで集計しない。
    prepared_trials = [prepare_trial_config(trial) for trial in trials]
    error_chunks = {name: [] for name, _ in get_method_styles()}
    for trial in prepared_trials:
        trial_errors = process_trial(trial)
        for method_name in error_chunks:
            error_chunks[method_name].append(trial_errors[method_name])

    # 各試行のサンプル数にかかわらず、すべてのサンプルを同じ重みで扱う。
    return {
        method_name: np.concatenate(chunks)
        for method_name, chunks in error_chunks.items()
    }


# =========================================================
# 統合CDF・80%点・90%点・結果表示
# =========================================================
def make_pooled_cdf(errors):
    """全試行の有限な誤差から、1/N, 2/N, ..., N/Nの経験CDFを作る。"""
    errors = np.asarray(errors, dtype=float)
    finite_errors = errors[np.isfinite(errors)]
    if len(finite_errors) == 0:
        raise ValueError('CDFを計算できる有限な絶対誤差がありません。')

    absolute_error_sorted = np.sort(finite_errors)
    cumulative_probability = (
        np.arange(1, len(absolute_error_sorted) + 1, dtype=float)
        / len(absolute_error_sorted)
    )
    return pd.DataFrame({
        'absolute_error_deg': absolute_error_sorted,
        'cdf': cumulative_probability
    })


def get_cdf_error_at_probability(cdf_data, probability):
    """経験CDFが指定確率に初めて到達する実測誤差を、補間せず返す。"""
    if not 0.0 < probability <= 1.0:
        raise ValueError('累積確率は0より大きく1以下で指定してください。')
    if cdf_data.empty:
        raise ValueError('分位点を計算できるCDFデータがありません。')

    index = np.searchsorted(
        cdf_data['cdf'].to_numpy(), probability, side='left'
    )
    if index >= len(cdf_data):
        raise ValueError('指定した累積確率に到達するCDFデータがありません。')
    return float(cdf_data['absolute_error_deg'].iloc[index])


def print_pooled_summary(cdf_by_method):
    """80%・90%点を小数2桁で表示し、別途、手法別の総サンプル数を表示する。"""
    for method_name, _ in get_method_styles():
        cdf_data = cdf_by_method[method_name]
        error_80 = get_cdf_error_at_probability(cdf_data, 0.80)
        error_90 = get_cdf_error_at_probability(cdf_data, 0.90)
        print(f'\n=== {method_name} ===')
        print(f'80%点の絶対誤差: {error_80:.2f} deg')
        print(f'90%点の絶対誤差: {error_90:.2f} deg')

    print()
    for method_name, _ in get_method_styles():
        print(f'{method_name} 総評価サンプル数: {len(cdf_by_method[method_name])}')


def plot_pooled_cdf(cdf_by_method):
    """3手法の統合CDFを、凡例付きの1つのAxesに重ねる。"""
    fig, ax = plt.subplots(figsize=(11, 7))
    configure_cdf_axis(ax)
    for method_name, color in get_method_styles():
        cdf_data = cdf_by_method[method_name]
        ax.step(
            cdf_data['absolute_error_deg'],
            cdf_data['cdf'],
            where='post',
            label=method_name,
            color=color,
            linewidth=3.0,
            alpha=0.9
        )
    # ax.legend()
    fig.tight_layout()
    return fig, ax


def main():
    errors_by_method = pool_trial_errors(TRIALS)
    cdf_by_method = {
        name: make_pooled_cdf(errors)
        for name, errors in errors_by_method.items()
    }
    print_pooled_summary(cdf_by_method)
    plot_pooled_cdf(cdf_by_method)
    plt.show()


if __name__ == '__main__':
    main()
