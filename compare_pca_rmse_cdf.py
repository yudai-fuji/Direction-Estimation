# 研究会用に作成
# 加速度PCAにおける，固定窓・可変窓2手法のCDF，推定進行方向の時系列変化，RMSEを出力

from contextlib import redirect_stdout
from io import StringIO

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


# =========================================================
# User settings
# =========================================================
file_L = r'260630香川共同研究/KL2.csv'
file_R = r'260630香川共同研究/KR2.csv'
delay_time = 2.243022
video_to_sensor_offset = 10.451
limit_min_time = 3.00 + video_to_sensor_offset
limit_max_time = 49.52 + video_to_sensor_offset
#真値
true_headings = [90.0, -180.0, 90.0, 0.0, -90.0, -180.0, -90.0, 0.0]
turn_start_times_video = [11.40, 15.88, 20.81, 25.32, 30.38, 34.87, 44.43]
turn_end_times_video = [12.51, 16.96, 21.96, 26.45, 31.54, 36.00, 45.61]
turn_start_times = [round(t + video_to_sensor_offset, 3) for t in turn_start_times_video]
turn_end_times = [round(t + video_to_sensor_offset, 3) for t in turn_end_times_video]

# 方向転換中の線形補間だけで使う内部角度。
# 表示・誤差計算の直前までは，あえて [-180, 180) に丸めない。
turn_internal_headings = [
    (90.0, 180.0),
    (-180.0, -270.0),
    (90.0, 0.0),
    (0.0, -90.0),
    (-90.0, -180.0),
    (-180.0, -90.0),
    (-90.0, 0.0)
]

#固定窓のPCA適用本数
WINDOW_SIZE = 40

# 可変窓のPCA適用本数の最大値と最小値
PCA_WINDOW_MAX = 50
PCA_WINDOW_MIN = 30

# 第一主成分の寄与率がこの値を下回ったら窓幅を最小にする。
PCA_PC1_LOW_THRESHOLD = 0.75

# 左右両方の第一主成分寄与率がこの値を上回ったら窓幅回復の条件を満たす。
PCA_PC1_HIGH_THRESHOLD = 0.875

# 窓幅回復開始からこの本数ごとに窓幅を回復させる。
WINDOW_RECOVERY_STEP = 1

# 窓幅回復開始からこの本数ごとに窓幅を回復させる条件を満たすとみなす。
WINDOW_RECOVERY_INTERVAL = 5

# Falseでは左右共通のAND条件，Trueでは左右独立の条件で窓幅を決定する。
USE_INDEPENDENT_PREVIOUS_PC1_WINDOW = False

# センサ値を共通グリッドへ最近傍で割り当てる際の許容時間差の最大値
SAMPLE_INTERVAL = 0.02

NEAREST_TOLERANCE = SAMPLE_INTERVAL

TIME_ZERO_SENSORS = ['Lacc', 'Gyro', 'GameRo']

is_mask = 1
is_mask_g = 1

PRINT_SYNC_DIAGNOSTICS = False

# 左右単体の結果も表示させるか
SHOW_INDIVIDUAL_HEADINGS = True

# 推定結果を点で表示するか
PLOT_HEADING_AS_POINTS = True

# 標準化を行うか
APPLY_PCA_STANDARDIZATION = False

# 回転後の水平加速度にローパスフィルタをかけてからPCAに渡すか
APPLY_ACC_LOWPASS = False
ACC_LOWPASS_ALPHA = 0.2

# CDFでは手法別の色、時系列では左=青・右=赤・左右平均=緑を使用する。
COLOR_FIXED = '#0072B2'
COLOR_THRESHOLD = '#E69F00'
COLOR_PREVIOUS = '#009E73'


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
# 真値と評価区間
# =========================================================
def validate_true_heading_settings():
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


def make_evaluation_sections():
    validate_true_heading_settings()

    sections = []
    for i in range(7):
        straight_start = limit_min_time if i == 0 else turn_end_times[i - 1]
        straight_end = turn_start_times[i]
        sections.append({
            'section_name': f'{i + 1}回目の直進歩行区間',
            'section_kind': 'straight',
            'section_kind_label': '直進歩行',
            'start_time_s': float(straight_start),
            'end_time_s': float(straight_end),
            'true_start_deg': float(true_headings[i]),
            'true_end_deg': float(true_headings[i])
        })

        turn_before, turn_after = turn_internal_headings[i]
        sections.append({
            'section_name': f'{i + 1}回目の方向転換区間',
            'section_kind': 'turn',
            'section_kind_label': '方向転換',
            'start_time_s': float(turn_start_times[i]),
            'end_time_s': float(turn_end_times[i]),
            'true_start_deg': float(turn_before),
            'true_end_deg': float(turn_after)
        })

    sections.append({
        'section_name': '8回目の直進歩行区間',
        'section_kind': 'straight',
        'section_kind_label': '直進歩行',
        'start_time_s': float(turn_end_times[-1]),
        'end_time_s': float(limit_max_time),
        'true_start_deg': float(true_headings[-1]),
        'true_end_deg': float(true_headings[-1])
    })

    return sections


def make_true_heading(time_s):
    validate_true_heading_settings()

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
# クォータニオンと姿勢補償
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


def prepare_synchronized_sensor_data():
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
# 180度補正に用いる角速度累積方位
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
# 区間RMSEと絶対角度誤差CDF
# =========================================================
def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def rms_or_nan(errors_deg):
    errors_deg = np.asarray(errors_deg, dtype=float)

    if len(errors_deg) == 0:
        return np.nan

    return float(np.sqrt(np.mean(errors_deg**2)))


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


def empty_absolute_error_cdf_df():
    return pd.DataFrame(columns=['absolute_error_deg', 'cdf'])


def make_absolute_error_cdf_data(
    heading_R,
    heading_L,
    use_weighted_mean=False,
    use_simple_mean=False
):
    aligned = align_headings_for_section_rmse(
        heading_R,
        heading_L,
        use_weighted_mean=use_weighted_mean,
        use_simple_mean=use_simple_mean
    )

    if aligned.empty:
        return empty_absolute_error_cdf_df()

    mask = (
        (aligned['time_s'] >= limit_min_time)
        & (aligned['time_s'] <= limit_max_time)
    )

    evaluation_df = (
        aligned.loc[mask, ['time_s', 'theta_mean_deg']]
        .reset_index(drop=True)
    )

    if evaluation_df.empty:
        return empty_absolute_error_cdf_df()

    true_heading = make_true_heading(evaluation_df['time_s'].to_numpy())
    error_deg = angle_diff_pm180(
        evaluation_df['theta_mean_deg'].to_numpy(),
        true_heading
    )

    absolute_error_deg = np.abs(error_deg)
    absolute_error_sorted = np.sort(absolute_error_deg)
    cumulative_probability = (
        np.arange(1, len(absolute_error_sorted) + 1, dtype=float)
        / len(absolute_error_sorted)
    )

    return pd.DataFrame({
        'absolute_error_deg': absolute_error_sorted,
        'cdf': cumulative_probability
    })


def calc_section_rmse_rows(
    method_name,
    heading_R,
    heading_L,
    use_weighted_mean=False,
    use_simple_mean=False
):
    sections = make_evaluation_sections()
    aligned = align_headings_for_section_rmse(
        heading_R,
        heading_L,
        use_weighted_mean=use_weighted_mean,
        use_simple_mean=use_simple_mean
    )

    rows = []
    for section_index, section in enumerate(sections):
        include_end = section_index == len(sections) - 1
        if len(aligned) == 0:
            mask = np.array([], dtype=bool)
        else:
            t_all = aligned['time_s'].to_numpy()
            if include_end:
                mask = (
                    (t_all >= section['start_time_s'])
                    & (t_all <= section['end_time_s'])
                )
            else:
                mask = (
                    (t_all >= section['start_time_s'])
                    & (t_all < section['end_time_s'])
                )

        sample_count = int(np.sum(mask))

        if sample_count == 0:
            right_rmse = np.nan
            left_rmse = np.nan
            mean_rmse = np.nan
        else:
            t_eval = aligned.loc[mask, 'time_s'].to_numpy()
            true_heading_eval = make_true_heading(t_eval)
            right_err = angle_diff_pm180(
                aligned.loc[mask, 'theta_deg_R'].to_numpy(),
                true_heading_eval
            )
            left_err = angle_diff_pm180(
                aligned.loc[mask, 'theta_deg_L'].to_numpy(),
                true_heading_eval
            )
            mean_err = angle_diff_pm180(
                aligned.loc[mask, 'theta_mean_deg'].to_numpy(),
                true_heading_eval
            )

            right_rmse = rms_or_nan(right_err)
            left_rmse = rms_or_nan(left_err)
            mean_rmse = rms_or_nan(mean_err)

        rows.append({
            'method': method_name,
            'section_name': section['section_name'],
            'section_kind': section['section_kind'],
            'section_kind_label': section['section_kind_label'],
            'start_time_s': section['start_time_s'],
            'end_time_s': section['end_time_s'],
            'true_start_deg': section['true_start_deg'],
            'true_end_deg': section['true_end_deg'],
            'sample_count': sample_count,
            'right_rmse_deg': right_rmse,
            'left_rmse_deg': left_rmse,
            'mean_rmse_deg': mean_rmse
        })

    return rows


def make_section_rmse_summary(section_rmse_df):
    summary_rows = []
    rmse_cols = ['right_rmse_deg', 'left_rmse_deg', 'mean_rmse_deg']

    for method_name in section_rmse_df['method'].drop_duplicates():
        method_df = section_rmse_df[section_rmse_df['method'] == method_name]
        summary_specs = [
            ('直進歩行区間平均', method_df['section_kind'] == 'straight'),
            ('方向転換区間平均', method_df['section_kind'] == 'turn'),
            ('全15区間平均', np.ones(len(method_df), dtype=bool))
        ]

        for summary_name, mask in summary_specs:
            target_df = method_df.loc[mask]
            row = {
                'method': method_name,
                'summary': summary_name
            }
            for col in rmse_cols:
                row[col] = finite_mean(target_df[col].to_numpy())
            summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# =========================================================
# 推定進行方向の時系列描画
# =========================================================
def align_heading_for_plot(
    heading_R,
    heading_L,
    use_weighted_mean,
    use_simple_mean=False
):
    aligned = align_headings_for_section_rmse(
        heading_R,
        heading_L,
        use_weighted_mean=use_weighted_mean,
        use_simple_mean=use_simple_mean
    )

    if aligned.empty:
        return None

    t_plot = aligned['time_s'].to_numpy()
    theta_R_plot = aligned['theta_deg_R'].to_numpy()
    theta_L_plot = aligned['theta_deg_L'].to_numpy()
    theta_mean_plot = aligned['theta_mean_deg'].to_numpy()

    true_heading_all = make_true_heading(t_plot)

    if is_mask == 1:
        mask = (limit_min_time <= t_plot) & (t_plot <= limit_max_time)

        t_plot = t_plot[mask]
        theta_R_plot = theta_R_plot[mask]
        theta_L_plot = theta_L_plot[mask]
        theta_mean_plot = theta_mean_plot[mask]
        true_heading_all = true_heading_all[mask]

    if len(t_plot) == 0:
        return None

    return {
        'time_s': t_plot,
        'theta_R': wrap_pm180(theta_R_plot),
        'theta_L': wrap_pm180(theta_L_plot),
        'theta_mean': wrap_pm180(theta_mean_plot),
        'true_heading': wrap_pm180(true_heading_all)
    }


def plot_heading_estimate(ax, t_plot, theta_plot, label, color, marker_size=12):
    if PLOT_HEADING_AS_POINTS:
        ax.scatter(
            t_plot,
            theta_plot,
            label=label,
            c=color,
            s=marker_size,
            alpha=0.75,
            edgecolors='none'
        )
    else:
        ax.plot(
            t_plot,
            theta_plot,
            label=label,
            c=color,
            alpha=0.8
        )


def plot_heading_on_axis(
    ax,
    heading_R,
    heading_L,
    title_str,
    use_weighted_mean,
    use_simple_mean=False
):
    plot_data = align_heading_for_plot(
        heading_R,
        heading_L,
        use_weighted_mean=use_weighted_mean,
        use_simple_mean=use_simple_mean
    )

    if plot_data is None:
        ax.text(
            0.5,
            0.5,
            'プロットできるデータがありません．',
            ha='center',
            va='center',
            transform=ax.transAxes
        )
        ax.set_title(title_str)
        return

    t_plot = plot_data['time_s']

    if SHOW_INDIVIDUAL_HEADINGS:
        plot_heading_estimate(
            ax,
            t_plot,
            plot_data['theta_L'],
            label='左手の端末',
            color='b',
            marker_size=4
        )
        plot_heading_estimate(
            ax,
            t_plot,
            plot_data['theta_R'],
            label='右手の端末',
            color='r',
            marker_size=4
        )

    mean_label = '左右平均'

    plot_heading_estimate(
        ax,
        t_plot,
        plot_data['theta_mean'],
        label=mean_label,
        color='g',
        marker_size=6
    )
    ax.plot(
        t_plot,
        plot_data['true_heading'],
        label='真値',
        c='k',
        linestyle='--',
        alpha=0.8
    )
   
    ax.set_xlabel('時間 [s]')
    ax.set_ylabel('推定進行方向 [deg]')
    ax.set_title(title_str)
    ax.grid(True)
    ax.legend()


# =========================================================
# 3手法の比較・結果表示
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


def calculate_rmse_summary(method_specs):
    """直進8・方向転換7・全15区間のRMSE平均を、左右平均列だけ返す。"""
    section_rows = []
    for method_name, heading_R, heading_L, _ in method_specs:
        section_rows.extend(
            calc_section_rmse_rows(
                method_name,
                heading_R,
                heading_L,
                use_weighted_mean=False,
                use_simple_mean=False
            )
        )

    # 全サンプルをまとめたRMSEではなく、既存の区間RMSEの単純平均。
    summary = make_section_rmse_summary(pd.DataFrame(section_rows))
    return summary[['method', 'summary', 'mean_rmse_deg']].copy()


def print_rmse_summary(summary):
    """手法見出しと、左右平均のRMSEを合計9個だけ表示する。"""
    summary_labels = [
        ('直進歩行区間平均', '直進区間平均RMSE'),
        ('方向転換区間平均', '方向転換区間平均RMSE'),
        ('全15区間平均', '全区間平均RMSE')
    ]
    for method_name in summary['method'].drop_duplicates():
        values = (
            summary.loc[summary['method'] == method_name]
            .set_index('summary')['mean_rmse_deg']
        )
        print(f'\n=== {method_name} ===')
        for summary_name, display_label in summary_labels:
            print(f'{display_label}: {values.loc[summary_name]:.4f} deg')


def make_cdf_data(method_specs):
    """左右のベクトル平均から、既存の評価時間・角度差・CDF計算を使う。"""
    cdf_specs = []
    for method_name, heading_R, heading_L, color in method_specs:
        cdf_df = make_absolute_error_cdf_data(
            heading_R,
            heading_L,
            use_weighted_mean=False,
            use_simple_mean=False
        )
        cdf_specs.append((method_name, cdf_df, color))
    return cdf_specs


def configure_cdf_axis(ax):
    """2つのFigureでCDFの表示範囲と軸設定をそろえる。"""
    ax.set_xlabel('左右平均の絶対誤差 [deg]')
    ax.set_ylabel('累積確率')
    ax.set_xlim(0.0, 50.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.grid(True, which='both', linestyle='--', alpha=0.5)


def plot_cdf_separately(cdf_specs):
    """固定窓・閾値型・前時刻比較型のCDFを横3列に表示する。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    for ax, (method_name, cdf_df, color) in zip(axes, cdf_specs):
        ax.set_title(method_name)
        configure_cdf_axis(ax)
        if cdf_df.empty:
            ax.text(
                0.5, 0.5, 'CDFを計算できるデータがありません。',
                ha='center', va='center', transform=ax.transAxes
            )
        else:
            ax.step(
                cdf_df['absolute_error_deg'], cdf_df['cdf'],
                where='post', color=color, linewidth=2.0, alpha=0.9
            )
    fig.tight_layout()
    return fig, axes


def plot_cdf_combined(cdf_specs):
    """同じCDFデータを1つのAxesに重ねる。凡例は作成しない。"""
    fig, ax = plt.subplots(figsize=(11, 7))
    configure_cdf_axis(ax)
    missing_methods = []
    for method_name, cdf_df, color in cdf_specs:
        if cdf_df.empty:
            missing_methods.append(method_name)
            continue
        ax.step(
            cdf_df['absolute_error_deg'], cdf_df['cdf'],
            where='post', color=color, linewidth=3.0, alpha=0.9
        )
    if missing_methods:
        ax.text(
            0.5, 0.5,
            '\n'.join(f'{name}: CDFデータなし' for name in missing_methods),
            ha='center', va='center', transform=ax.transAxes
        )
    fig.tight_layout()
    return fig, ax


def plot_heading_comparison(method_specs):
    """3手法の補正済み推定方位と真値を横一列に表示する。"""
    fig, axes = plt.subplots(
        1, 3, figsize=(21, 6), sharex=True, sharey=True
    )
    for ax, (method_name, heading_R, heading_L, _) in zip(axes, method_specs):
        plot_heading_on_axis(
            ax,
            heading_R,
            heading_L,
            title_str=method_name,
            use_weighted_mean=False,
            use_simple_mean=False
        )
    fig.tight_layout()
    return fig, axes


def main():
    validate_true_heading_settings()

    # 同期計算はそのまま実行し、その関数が出力する診断printだけ抑制する。
    with redirect_stdout(StringIO()):
        sync_data = prepare_synchronized_sensor_data()

    # 角速度累積法は180度補正の基準としてのみ使用する。
    gyro_heading_L = compute_heading_gyro_integral_from_synced(
        sync_data['L'], sync_data['initial_quat_L'], initial_heading_deg=90.0
    )
    gyro_heading_R = compute_heading_gyro_integral_from_synced(
        sync_data['R'], sync_data['initial_quat_R'], initial_heading_deg=90.0
    )
    gyro_mean_heading = make_gyro_mean_heading(gyro_heading_R, gyro_heading_L)

    method_specs = calculate_method_headings(sync_data, gyro_mean_heading)
    summary = calculate_rmse_summary(method_specs)
    print_rmse_summary(summary)

    cdf_specs = make_cdf_data(method_specs)
    plot_cdf_separately(cdf_specs)
    plot_cdf_combined(cdf_specs)
    plot_heading_comparison(method_specs)
    plt.show()


if __name__ == '__main__':
    main()
