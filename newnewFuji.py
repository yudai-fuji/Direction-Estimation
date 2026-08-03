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
file_L = r'260630香川共同研究/KL1.csv'
file_R = r'260630香川共同研究/KR1.csv'
delay_time = 2.055392
video_to_sensor_offset = 10.704
limit_min_time = 3.51 + video_to_sensor_offset
limit_max_time = 50.60 + video_to_sensor_offset
#真値
true_headings = [90.0, -180.0, 90.0, 0.0, -90.0, -180.0, -90.0, 0.0]
turn_start_times_video = [12.63, 17.07, 22.21, 26.79, 31.86, 36.44, 46.00]
turn_end_times_video = [13.73, 18.27, 23.32, 27.86, 33.05, 37.57, 47.26]
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
WINDOW_SIZE = 50

# 可変窓のPCA適用本数の最大値と最小値
PCA_WINDOW_MAX = 60
PCA_WINDOW_MIN = 30

# 第一主成分の寄与率がこの値を下回ったら窓幅を最小にする。
PCA_PC1_LOW_THRESHOLD = 0.75

# 左右両方の第一主成分寄与率がこの値を上回ったら窓幅回復の条件を満たす。
PCA_PC1_HIGH_THRESHOLD = 0.75

# 窓幅回復開始からこの本数ごとに窓幅を回復させる。
WINDOW_RECOVERY_STEP = 1

# 窓幅回復開始からこの本数ごとに窓幅を回復させる条件を満たすとみなす。
WINDOW_RECOVERY_INTERVAL = 5

# Falseでは左右共通のAND条件，Trueでは左右独立の条件で窓幅を決定する。
USE_INDEPENDENT_PREVIOUS_PC1_WINDOW = True

# センサ値を共通グリッドへ最近傍で割り当てる際の許容時間差の最大値
SAMPLE_INTERVAL = 0.02

NEAREST_TOLERANCE = SAMPLE_INTERVAL

TIME_ZERO_SENSORS = ['Lacc', 'Gyro', 'GameRo']

is_mask = 1
is_mask_g = 1

PRINT_SYNC_DIAGNOSTICS = True

# 左右単体の結果も表示させるか
SHOW_INDIVIDUAL_HEADINGS = False

# 推定結果を点で表示するか
PLOT_HEADING_AS_POINTS = True

# 回転後の水平加速度にローパスフィルタをかけてからPCAに渡すか
APPLY_ACC_LOWPASS = False
ACC_LOWPASS_ALPHA = 0.2


# 角度を-180度以上180度未満の範囲へ正規化する。
def wrap_pm180(theta_deg):
    if is_mask_g == 1:
        theta_deg = np.asarray(theta_deg, dtype=float)
        return np.mod(theta_deg + 180.0, 360.0) - 180.0

    return theta_deg


# 推定角度と真値角度の差を-180度以上180度未満で求める。
def angle_diff_pm180(pred_deg, true_deg):
    pred_deg = np.asarray(pred_deg, dtype=float)
    true_deg = np.asarray(true_deg, dtype=float)
    return np.mod((pred_deg - true_deg) + 180.0, 360.0) - 180.0


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


# 8回の直進歩行区間と7回の方向転換区間の設定を確認する。
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

def add_turn_regions_to_axis(ax, turn_starts, turn_ends):
    if len(turn_starts) != len(turn_ends):
        raise ValueError('方向転換開始時刻と終了時刻の個数が一致していません')

    for i, (start, end) in enumerate(zip(turn_starts, turn_ends)):
        region_label = '方向転換区間' if i == 0 else None

        ax.axvspan(start, end, facecolor = 'green', alpha = 0.12, edgecolor = 'none',
                   linewidth = 0, zorder = 0, label = region_label)



def kyoyaku(gx, gy, gz, gw):
    return (-gx, -gy, -gz, gw)


# 初期姿勢を基準にした相対クォータニオンを計算する。
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


# GameRoの姿勢情報で3軸ベクトルを初期姿勢基準の座標系へ回転する。
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
# Sensor synchronization
# =========================================================
# センサCSVを読み込み，基準時刻からの秒時刻と初期姿勢を取得する。
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


# 指定センサの有効な時刻範囲を取得する。
def get_sensor_time_range(df, sensor_name):
    sensor_df = df[df['Sensor'] == sensor_name]

    if len(sensor_df) == 0:
        raise ValueError(f'{sensor_name} がファイル内に存在しません．')

    return sensor_df['time_s'].min(), sensor_df['time_s'].max()


# 左右センサが共通して持つ時間範囲に等間隔の時刻グリッドを作る。
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


# 指定センサの値を共通時刻グリッドへ最近傍で割り当てる。
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


# 片側の複数センサ値を共通時刻グリッド上に同期した表へまとめる。
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


# 同期後データの採用点数や時刻ずれを表示する。
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


# 左右CSVを読み込み，同期済みデータと初期姿勢をまとめて返す。
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


# 回転補正したZ軸角速度を時間積分して方位角を推定する。
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
# Acceleration PCA
# =========================================================
# 回転補正済みの水平加速度をPCA入力用の時系列として作る。
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


# 左右のPC1寄与率から，共通のPCA窓幅時系列を作る。
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


# 前時刻比較型可変窓の空スケジュールを作る。
def empty_previous_pc1_window_schedule_df():
    return pd.DataFrame(columns=[
        'time_s',
        'window_size',
        'pc1_ratio_L',
        'pc1_ratio_R',
        'removed_count',
        'forced_min_window'
    ])


# 前時刻に採用した左右PC1寄与率との比較から，共通のPCA窓幅時系列を作る。
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


# 前時刻比較型可変窓の採用状況を表示する。
def print_previous_pc1_window_diagnostics(window_schedule_previous_pc1):
    print('\n=== 前時刻寄与率比較型可変窓の診断 ===')

    if (
        window_schedule_previous_pc1 is None
        or len(window_schedule_previous_pc1) == 0
    ):
        print('診断できる窓幅スケジュールがありません。')
        return

    finite_mask = (
        np.isfinite(window_schedule_previous_pc1['pc1_ratio_L'])
        & np.isfinite(window_schedule_previous_pc1['pc1_ratio_R'])
    )
    valid_schedule = (
        window_schedule_previous_pc1.loc[finite_mask]
        .reset_index(drop=True)
    )

    valid_count = len(valid_schedule)
    print(f'PCA寄与率を計算できた時刻数: {valid_count}')

    if valid_count == 0:
        print('窓幅統計を計算できるデータがありません。')
        return

    mean_window = valid_schedule['window_size'].mean()
    min_window = int(valid_schedule['window_size'].min())
    max_window = int(valid_schedule['window_size'].max())
    mean_removed_count = valid_schedule['removed_count'].mean()
    min_window_ratio = (
        valid_schedule['window_size'] == PCA_WINDOW_MIN
    ).mean()
    forced_count = int(valid_schedule['forced_min_window'].sum())
    forced_ratio = valid_schedule['forced_min_window'].mean()

    print(f'平均窓幅: {mean_window:.4f}')
    print(f'最小窓幅: {min_window}')
    print(f'最大窓幅: {max_window}')
    print(f'平均 removed_count: {mean_removed_count:.4f}')
    print(
        f'窓幅が{PCA_WINDOW_MIN}だった割合: '
        f'{min_window_ratio:.2%}'
    )
    print(f'forced_min_window が True だった回数: {forced_count}')
    print(
        'forced_min_window が True だった割合: '
        f'{forced_ratio:.2%}'
    )


# 左右独立方式で使う片側分の空スケジュールを作る。
def empty_independent_previous_pc1_window_schedule_df():
    return pd.DataFrame(columns=[
        'time_s',
        'window_size',
        'pc1_ratio',
        'removed_count',
        'forced_min_window'
    ])


# 片側の前時刻PC1寄与率との比較から，独立したPCA窓幅時系列を作る。
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


# 左右独立方式の窓幅採用状況を片側ずつ表示する。
def print_independent_previous_pc1_window_diagnostics(
    window_schedule_previous_pc1_L,
    window_schedule_previous_pc1_R
):
    print('\n=== 前時刻寄与率比較型可変窓・左右独立方式の診断 ===')

    side_schedules = [
        ('左手', window_schedule_previous_pc1_L),
        ('右手', window_schedule_previous_pc1_R)
    ]

    for side_label, schedule in side_schedules:
        print(f'\n--- {side_label} ---')

        if schedule is None or len(schedule) == 0:
            print('診断できる窓幅スケジュールがありません。')
            continue

        finite_mask = np.isfinite(schedule['pc1_ratio'])
        valid_schedule = schedule.loc[finite_mask].reset_index(drop=True)

        valid_count = len(valid_schedule)
        print(f'PCA寄与率を計算できた時刻数: {valid_count}')

        if valid_count == 0:
            print('窓幅統計を計算できるデータがありません。')
            continue

        mean_window = valid_schedule['window_size'].mean()
        min_window = int(valid_schedule['window_size'].min())
        max_window = int(valid_schedule['window_size'].max())
        mean_removed_count = valid_schedule['removed_count'].mean()
        min_window_ratio = (
            valid_schedule['window_size'] == PCA_WINDOW_MIN
        ).mean()
        forced_count = int(valid_schedule['forced_min_window'].sum())
        forced_ratio = valid_schedule['forced_min_window'].mean()

        print(f'平均窓幅: {mean_window:.4f}')
        print(f'最小窓幅: {min_window}')
        print(f'最大窓幅: {max_window}')
        print(f'平均 removed_count: {mean_removed_count:.4f}')
        print(
            f'窓幅が{PCA_WINDOW_MIN}だった割合: '
            f'{min_window_ratio:.2%}'
        )
        print(f'forced_min_window が True だった回数: {forced_count}')
        print(
            'forced_min_window が True だった割合: '
            f'{forced_ratio:.2%}'
        )


# 加速度PCA結果が空になる場合の列構造だけを持つDataFrameを作る。
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


# PCA計算用データへ固定窓または可変窓の窓幅を付与する。
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


# 回転補正した水平加速度にPCAをかけて進行方向を推定する中核処理を行う。
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


# 固定窓の通常加速度PCAで進行方向を推定する。
def compute_heading_acc_pca_from_synced(sync_df, initial_quat, window_size=WINDOW_SIZE):
    return compute_acc_pca_core(
        sync_df,
        initial_quat,
        window_size=window_size,
        window_schedule_df=None,
        use_ratio_weight=False
    )


# 固定窓の寄与率重み付き加速度PCAで進行方向を推定する。
def compute_heading_acc_pca_proposed_from_synced(
    sync_df,
    initial_quat,
    window_size=WINDOW_SIZE
):
    return compute_acc_pca_core(
        sync_df,
        initial_quat,
        window_size=window_size,
        window_schedule_df=None,
        use_ratio_weight=True
    )


# PC1寄与率で決めた可変窓を使って加速度PCAの進行方向を推定する。
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


# ジャイロ積分方位を基準にPCA方位の180度反転を補正する。
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
# RMSE and improvement
# =========================================================
# 各手法のRMSE比較表を表示する。
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


def print_section_rmse_tables(section_rmse_rows):
    if len(section_rmse_rows) == 0:
        print('\n=== 区間別RMSE ===')
        print('表示するRMSEがありません。')
        return

    section_rmse_df = pd.DataFrame(section_rmse_rows)
    detail_cols = [
        'method',
        'section_name',
        'section_kind_label',
        'start_time_s',
        'end_time_s',
        'right_rmse_deg',
        'left_rmse_deg',
        'mean_rmse_deg'
    ]

    detail_df = section_rmse_df[detail_cols].rename(columns={
        'method': '手法',
        'section_name': '区間',
        'section_kind_label': '区間種類',
        'start_time_s': '開始時刻[s]',
        'end_time_s': '終了時刻[s]',
        'right_rmse_deg': '右手RMSE[deg]',
        'left_rmse_deg': '左手RMSE[deg]',
        'mean_rmse_deg': '左右平均RMSE[deg]'
    })

    print('\n=== 15区間それぞれのRMSE ===')
    print(
        detail_df.to_string(
            index=False,
            float_format=lambda x: f'{x:.4f}'
        )
    )

    summary_df = make_section_rmse_summary(section_rmse_df)
    summary_order = ['直進歩行区間平均', '方向転換区間平均', '全15区間平均']
    method_order = [
        '固定窓PCA',
        '固定窓・寄与率重み付きPCA',
        '可変窓PCA',
        '可変窓・寄与率重み付きPCA',
        '前時刻比較型可変窓PCA',
        '前時刻比較型可変窓・寄与率重み付きPCA',
        '左右独立・前時刻比較型可変窓PCA',
        '左右独立・前時刻比較型可変窓・寄与率重み付きPCA',
        '角速度累積法'
    ]
    side_specs = [
        ('右手', 'right_rmse_deg'),
        ('左手', 'left_rmse_deg'),
        ('左右平均', 'mean_rmse_deg')
    ]

    for side_label, rmse_col in side_specs:
        side_summary = (
            summary_df
            .pivot(index='method', columns='summary', values=rmse_col)
            .reset_index()
        )
        ordered_methods = [
            method for method in method_order
            if method in set(side_summary['method'])
        ]
        remaining_methods = [
            method for method in side_summary['method']
            if method not in method_order
        ]
        side_summary['method'] = pd.Categorical(
            side_summary['method'],
            categories=ordered_methods + remaining_methods,
            ordered=True
        )
        side_summary = side_summary.sort_values('method').reset_index(drop=True)
        side_summary['method'] = side_summary['method'].astype(str)
        side_summary = side_summary[['method'] + summary_order]
        side_summary = side_summary.rename(columns={'method': '手法'})

        print(f'\n=== {side_label}RMSEの区間平均 ===')
        print(
            side_summary.to_string(
                index=False,
                float_format=lambda x: f'{x:.4f}'
            )
        )


# =========================================================
# Plotting
# =========================================================

def plot_pca_method_absolute_error_cdf(method_specs):
    # 手法ごとの色を固定する。
    color_map = {
        '固定窓PCA': 'tab:blue',
        '固定窓・寄与率重み付きPCA': 'tab:orange',
        '可変窓PCA': 'tab:green',
        '可変窓・寄与率重み付きPCA': 'tab:red',
        '前時刻比較型可変窓PCA': 'tab:purple',
        '前時刻比較型可変窓・寄与率重み付きPCA': 'tab:brown',
        '左右独立・前時刻比較型可変窓PCA': 'tab:pink',
        '左右独立・前時刻比較型可変窓・寄与率重み付きPCA': 'tab:gray'
    }

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(11, 7)
    )

    plotted_method_count = 0

    for (
        method_name,
        heading_R,
        heading_L,
        use_weighted_mean,
        use_simple_mean
    ) in method_specs:
        cdf_df = make_absolute_error_cdf_data(
            heading_R,
            heading_L,
            use_weighted_mean=use_weighted_mean,
            use_simple_mean=use_simple_mean
        )

        if len(cdf_df) == 0:
            print(
                f'{method_name}: '
                'CDFを計算できるデータがありません。'
            )
            continue

        ax.step(
            cdf_df['absolute_error_deg'],
            cdf_df['cdf'],
            where='post',
            label=(
                f'{method_name} '
                f'(n={len(cdf_df)})'
            ),
            color=color_map.get(method_name),
            linewidth=2.0,
            alpha=0.9
        )

        plotted_method_count += 1

    if plotted_method_count == 0:
        ax.text(
            0.5,
            0.5,
            'プロットできるCDFデータがありません。',
            ha='center',
            va='center',
            transform=ax.transAxes
        )

    ax.set_xlabel(
        '左右平均方位の絶対誤差 [deg]',
        fontsize=16
    )
    ax.set_ylabel(
        '累積確率',
        fontsize=16
    )
    ax.set_title(
        'PCA手法の左右平均方位絶対誤差の累積分布関数'
    )

    # 絶対角度誤差なので、横軸の最小値は0度。
    ax.set_xlim(left=0.0)

    # CDFの値域は0～1。
    ax.set_ylim(0.0, 1.0)

    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.tick_params(axis='both', labelsize=13)



    ax.grid(
        True,
        which='both',
        linestyle='--',
        alpha=0.5
    )

    if plotted_method_count > 0:
        ax.legend()

    plt.tight_layout()
    plt.show()


# 左右の推定方位を時刻でそろえ，描画用の真値と平均方位を作る。
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


# 1つの軸に左右推定方位，平均方位，真値方位を描画する。
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


# 通常PCAと寄与率重み付きPCAの方位時系列を横並びで描画する。
def plot_pca_comparison_figure(
    acc_heading_R,
    acc_heading_L,
    prop_heading_R,
    prop_heading_L,
    figure_title
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    fig.suptitle(figure_title, fontsize=16)

    plot_heading_on_axis(
        axes[0, 0],
        acc_heading_R,
        acc_heading_L,
        '加速度PCA',
        use_weighted_mean=False
    )

    plot_heading_on_axis(
        axes[0, 1],
        prop_heading_R,
        prop_heading_L,
        '寄与率重み付き加速度PCA',
        use_weighted_mean=True
    )

    plt.tight_layout()
    plt.show()


# 角速度累積法の左右単体と左右平均を描画する。
def plot_gyro_heading_figure(gyro_heading_R, gyro_heading_L):
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(12, 6)
    )

    plot_heading_on_axis(
        ax,
        gyro_heading_R,
        gyro_heading_L,
        '角速度累積法',
        use_weighted_mean=False,
        use_simple_mean=True
    )

    fig.suptitle('時系列変化（角速度累積法）', fontsize=16)
    plt.tight_layout()
    plt.show()


# PC1寄与率の変化を描画範囲に絞る。
def prepare_pc1_window_plot_df(window_schedule_common):
    if window_schedule_common is None or len(window_schedule_common) == 0:
        return empty_pc1_window_schedule_df()

    plot_window = window_schedule_common.copy()
    if is_mask == 1:
        mask = (
            (plot_window['time_s'] >= limit_min_time)
            & (plot_window['time_s'] <= limit_max_time)
        )
        plot_window = plot_window.loc[mask].reset_index(drop=True)

    return plot_window


# PC1寄与率の変化を描画する。
def plot_pc1_window_control(window_schedule_common):
    plot_window = prepare_pc1_window_plot_df(window_schedule_common)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 5),
        sharex=True,
        sharey=True
    )

    fig.suptitle('PC1寄与率の時系列変化', fontsize=16)

    plot_specs = [
        (axes[0], 'pc1_ratio_L', '左手', 'b'),
        (axes[1], 'pc1_ratio_R', '右手', 'r')
    ]

    for ax_ratio, ratio_column, hand_label, line_color in plot_specs:
        add_turn_regions_to_axis(ax_ratio, turn_start_times, turn_end_times)

        if len(plot_window) > 0:
            ax_ratio.plot(
                plot_window['time_s'],
                plot_window[ratio_column],
                label=f'{hand_label} PC1寄与率',
                c=line_color,
                alpha=0.8
            )

        ax_ratio.axhline(
            PCA_PC1_LOW_THRESHOLD,
            c='k',
            linestyle='--',
            alpha=0.7,
            label=f'窓幅縮小閾値 {PCA_PC1_LOW_THRESHOLD:g}'
        )
        ax_ratio.axhline(
            PCA_PC1_HIGH_THRESHOLD,
            c='k',
            linestyle=':',
            alpha=0.7,
            label=f'窓幅回復閾値 {PCA_PC1_HIGH_THRESHOLD:g}'
        )
  
        ax_ratio.set_xlabel('時間 [s]')
        ax_ratio.set_ylabel('PC1寄与率')
        ax_ratio.set_title(hand_label)
        ax_ratio.set_ylim(0.5, 1.0)
        ax_ratio.grid(True)
        ax_ratio.legend()

    plt.tight_layout()
    plt.show()


# =========================================================
# Main
# =========================================================
# データ同期から各手法の推定，評価，描画までの全体処理を実行する。
def main():
    validate_true_heading_settings()

    sync_data = prepare_synchronized_sensor_data()

    sync_L = sync_data['L']
    sync_R = sync_data['R']
    initial_quat_L = sync_data['initial_quat_L']
    initial_quat_R = sync_data['initial_quat_R']

    gyro_heading_L = compute_heading_gyro_integral_from_synced(
        sync_L,
        initial_quat_L,
        initial_heading_deg=90.0
    )
    gyro_heading_R = compute_heading_gyro_integral_from_synced(
        sync_R,
        initial_quat_R,
        initial_heading_deg=90.0
    )
    gyro_mean_heading = make_gyro_mean_heading(
        gyro_heading_R,
        gyro_heading_L
    )

    window_schedule_common = make_common_window_schedule_from_pc1(
        sync_L,
        initial_quat_L,
        sync_R,
        initial_quat_R
    )
    if USE_INDEPENDENT_PREVIOUS_PC1_WINDOW:
        previous_pc1_method_name = (
            '左右独立・前時刻比較型可変窓PCA'
        )
        previous_pc1_weighted_method_name = (
            '左右独立・前時刻比較型可変窓・寄与率重み付きPCA'
        )
        previous_pc1_figure_title = (
            '左右独立・前時刻寄与率比較型可変窓PCA'
        )
        previous_pc1_progress_message = (
            '=== 左右独立・前時刻寄与率比較型可変窓PCAを計算中 ==='
        )
        window_schedule_previous_pc1_L = (
            make_independent_window_schedule_by_previous_pc1(
                sync_L,
                initial_quat_L
            )
        )
        window_schedule_previous_pc1_R = (
            make_independent_window_schedule_by_previous_pc1(
                sync_R,
                initial_quat_R
            )
        )
        print_independent_previous_pc1_window_diagnostics(
            window_schedule_previous_pc1_L,
            window_schedule_previous_pc1_R
        )
    else:
        previous_pc1_method_name = (
            '前時刻比較型可変窓PCA'
        )
        previous_pc1_weighted_method_name = (
            '前時刻比較型可変窓・寄与率重み付きPCA'
        )
        previous_pc1_figure_title = (
            '前時刻寄与率比較型可変窓PCA'
        )
        previous_pc1_progress_message = (
            '=== 前時刻寄与率比較型可変窓PCAを計算中 ==='
        )
        window_schedule_previous_pc1 = (
            make_common_window_schedule_by_previous_pc1(
                sync_L,
                initial_quat_L,
                sync_R,
                initial_quat_R
            )
        )
        window_schedule_previous_pc1_L = window_schedule_previous_pc1
        window_schedule_previous_pc1_R = window_schedule_previous_pc1
        print_previous_pc1_window_diagnostics(
            window_schedule_previous_pc1
        )

    print('\n=== 固定窓PCAを計算中 ===')
    heading_L_pca_fixed = compute_heading_acc_pca_from_synced(
        sync_L,
        initial_quat_L,
        window_size=WINDOW_SIZE
    )
    heading_R_pca_fixed = compute_heading_acc_pca_from_synced(
        sync_R,
        initial_quat_R,
        window_size=WINDOW_SIZE
    )

    heading_L_prop_fixed = compute_heading_acc_pca_proposed_from_synced(
        sync_L,
        initial_quat_L,
        window_size=WINDOW_SIZE
    )
    heading_R_prop_fixed = compute_heading_acc_pca_proposed_from_synced(
        sync_R,
        initial_quat_R,
        window_size=WINDOW_SIZE
    )

    print('\n=== PC1寄与率可変窓PCAを計算中 ===')
    heading_L_pca_variable = compute_heading_acc_pca_variable_from_synced(
        sync_L,
        initial_quat_L,
        window_schedule_common,
        use_ratio_weight=False
    )
    heading_R_pca_variable = compute_heading_acc_pca_variable_from_synced(
        sync_R,
        initial_quat_R,
        window_schedule_common,
        use_ratio_weight=False
    )

    heading_L_prop_variable = compute_heading_acc_pca_variable_from_synced(
        sync_L,
        initial_quat_L,
        window_schedule_common,
        use_ratio_weight=True
    )
    heading_R_prop_variable = compute_heading_acc_pca_variable_from_synced(
        sync_R,
        initial_quat_R,
        window_schedule_common,
        use_ratio_weight=True
    )

    print(f'\n{previous_pc1_progress_message}')
    heading_L_pca_previous_pc1 = compute_heading_acc_pca_variable_from_synced(
        sync_L,
        initial_quat_L,
        window_schedule_previous_pc1_L,
        use_ratio_weight=False
    )
    heading_R_pca_previous_pc1 = compute_heading_acc_pca_variable_from_synced(
        sync_R,
        initial_quat_R,
        window_schedule_previous_pc1_R,
        use_ratio_weight=False
    )

    heading_L_prop_previous_pc1 = compute_heading_acc_pca_variable_from_synced(
        sync_L,
        initial_quat_L,
        window_schedule_previous_pc1_L,
        use_ratio_weight=True
    )
    heading_R_prop_previous_pc1 = compute_heading_acc_pca_variable_from_synced(
        sync_R,
        initial_quat_R,
        window_schedule_previous_pc1_R,
        use_ratio_weight=True
    )

    print('\n=== 角速度累積法を用いたPCA 180度補正中 ===')
    heading_L_pca_fixed = resolve_pca_180_by_gyro(
        heading_L_pca_fixed,
        gyro_mean_heading,
        use_ratio_weight=False
    )
    heading_R_pca_fixed = resolve_pca_180_by_gyro(
        heading_R_pca_fixed,
        gyro_mean_heading,
        use_ratio_weight=False
    )
    heading_L_prop_fixed = resolve_pca_180_by_gyro(
        heading_L_prop_fixed,
        gyro_mean_heading,
        use_ratio_weight=True
    )
    heading_R_prop_fixed = resolve_pca_180_by_gyro(
        heading_R_prop_fixed,
        gyro_mean_heading,
        use_ratio_weight=True
    )
    heading_L_pca_variable = resolve_pca_180_by_gyro(
        heading_L_pca_variable,
        gyro_mean_heading,
        use_ratio_weight=False
    )
    heading_R_pca_variable = resolve_pca_180_by_gyro(
        heading_R_pca_variable,
        gyro_mean_heading,
        use_ratio_weight=False
    )
    heading_L_prop_variable = resolve_pca_180_by_gyro(
        heading_L_prop_variable,
        gyro_mean_heading,
        use_ratio_weight=True
    )
    heading_R_prop_variable = resolve_pca_180_by_gyro(
        heading_R_prop_variable,
        gyro_mean_heading,
        use_ratio_weight=True
    )
    heading_L_pca_previous_pc1 = resolve_pca_180_by_gyro(
        heading_L_pca_previous_pc1,
        gyro_mean_heading,
        use_ratio_weight=False
    )
    heading_R_pca_previous_pc1 = resolve_pca_180_by_gyro(
        heading_R_pca_previous_pc1,
        gyro_mean_heading,
        use_ratio_weight=False
    )
    heading_L_prop_previous_pc1 = resolve_pca_180_by_gyro(
        heading_L_prop_previous_pc1,
        gyro_mean_heading,
        use_ratio_weight=True
    )
    heading_R_prop_previous_pc1 = resolve_pca_180_by_gyro(
        heading_R_prop_previous_pc1,
        gyro_mean_heading,
        use_ratio_weight=True
    )

    section_method_specs = [
        (
            '固定窓PCA',
            heading_R_pca_fixed,
            heading_L_pca_fixed,
            False,
            False
        ),
        (
            '固定窓・寄与率重み付きPCA',
            heading_R_prop_fixed,
            heading_L_prop_fixed,
            True,
            False
        ),
        (
            '可変窓PCA',
            heading_R_pca_variable,
            heading_L_pca_variable,
            False,
            False
        ),
        (
            '可変窓・寄与率重み付きPCA',
            heading_R_prop_variable,
            heading_L_prop_variable,
            True,
            False
        ),
        (
            previous_pc1_method_name,
            heading_R_pca_previous_pc1,
            heading_L_pca_previous_pc1,
            False,
            False
        ),
        (
            previous_pc1_weighted_method_name,
            heading_R_prop_previous_pc1,
            heading_L_prop_previous_pc1,
            True,
            False
        ),
        (
            '角速度累積法',
            gyro_heading_R,
            gyro_heading_L,
            False,
            True
        )
    ]

    section_rmse_rows = []
    for (
        method_name,
        heading_R,
        heading_L,
        use_weighted_mean,
        use_simple_mean
    ) in section_method_specs:
        section_rmse_rows.extend(
            calc_section_rmse_rows(
                method_name,
                heading_R,
                heading_L,
                use_weighted_mean=use_weighted_mean,
                use_simple_mean=use_simple_mean
            )
        )

    print_section_rmse_tables(section_rmse_rows)

    cdf_method_specs = [spec
                        for spec in section_method_specs
                        if spec[0] != '角速度累積法']

    plot_pca_method_absolute_error_cdf(cdf_method_specs)

    plot_gyro_heading_figure(
        gyro_heading_R,
        gyro_heading_L
    )

    plot_pca_comparison_figure(
        heading_R_pca_fixed,
        heading_L_pca_fixed,
        heading_R_prop_fixed,
        heading_L_prop_fixed,
        figure_title=f'固定窓PCA（窓幅: {WINDOW_SIZE}サンプル）'
    )

    plot_pca_comparison_figure(
        heading_R_pca_variable,
        heading_L_pca_variable,
        heading_R_prop_variable,
        heading_L_prop_variable,
        figure_title='PC1寄与率可変窓PCA'
    )

    plot_pca_comparison_figure(
        heading_R_pca_previous_pc1,
        heading_L_pca_previous_pc1,
        heading_R_prop_previous_pc1,
        heading_L_prop_previous_pc1,
        figure_title=previous_pc1_figure_title
    )

    plot_pc1_window_control(window_schedule_common)


if __name__ == '__main__':
    main()
