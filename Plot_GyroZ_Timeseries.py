#端末座標系の角速度を世界基準の座標系へ回転変換し、Z軸角速度成分 Gyz の時系列変化を可視化する。
#回転後Z軸角速度 Gyz、Gyz の絶対値、|Gyz| の移動平均をプロット

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import japanize_matplotlib

file_L = r'260518栁澤共同研究/KL2.csv'
file_R = r'260518栁澤共同研究/KR2.csv'
delay_time = 0.449970
cod_times = [17.96, 22.5, 32.45]
limit_min_time = 8
limit_max_time = 39.31

SAMPLE_INTERVAL = 0.02

NEAREST_TOLERANCE = SAMPLE_INTERVAL

GYRO_SMOOTH_WINDOW = 30

is_mask = 1

PRINT_SYNC_DIAGNOSTICS = True

#四元数の共役を返す
def kyouyaku(gx, gy, gz, gw):
    return (-gx, -gy, -gz, gw)

#初期姿勢を基準に，逆回転する
def calc_relative_quaternion(gx, gy, gz, gw, initial_quat):
    gx0, gy0, gz0, gw0 = initial_quat

    Gx0, Gy0, Gz0, Gw0 = kyouyaku(gx0, gy0, gz0, gw0)

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

#センサ値を世界座標系に変換する
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

#時刻とセンサ値を読み込む処理
def load_sensor_file_with_time(file_path, time_offset_s=0.0):

    df = pd.read_csv(file_path)

    required_sensors_for_zero = ['Gyro', 'GameRo']
    zero_df = df[df['Sensor'].isin(required_sensors_for_zero)].copy()

    if len(zero_df) == 0:
        raise ValueError(
            f'{file_path} に Gyro または GameRo が存在しません．'
        )

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

#センサの時刻範囲を取得
def get_sensor_time_range(df, sensor_name):
    sensor_df = df[df['Sensor'] == sensor_name].copy()

    if len(sensor_df) == 0:
        raise ValueError(f'{sensor_name} がファイル内に存在しません．')

    return sensor_df['time_s'].min(), sensor_df['time_s'].max()

# =========================================================
# 6) 左右両方にGyroとGameRoが存在する重複区間から共通時刻グリッドを作る処理
# =========================================================

def make_common_grid(df_L, df_R):
    required_sensors = ['Gyro', 'GameRo']

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
            '左右のGyro/GameRoの重複区間がありません．'
            ' delay_time やCSVを確認してください．'
        )

    n_grid = int(np.floor((overlap_end - overlap_start) / SAMPLE_INTERVAL)) + 1
    grid_time = overlap_start + np.arange(n_grid) * SAMPLE_INTERVAL

    grid_df = pd.DataFrame({
        'time_s': grid_time
    })

    return grid_df, overlap_start, overlap_end


# =========================================================
# 7) 1つのセンサを共通時刻グリッドへ割り当てる処理
# =========================================================

def assign_sensor_to_grid(df, sensor_name, grid_df):
    """
    指定センサを共通時刻グリッドへ割り当てる．

    direction='nearest' を使う理由:
        共通時刻に最も近いセンサ値を使うため．
        backward の場合，常に過去側の値だけを使うため，
        ここでは角速度の確認用として nearest を使う．
    """
    sensor_df = (
        df[df['Sensor'] == sensor_name]
        .copy()
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
        sensor_df.sort_values('time_s'),
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


# =========================================================
# 8) 片側端末のGyroとGameRoを共通時刻グリッドへ同期する処理
# =========================================================

def build_synced_gyro_gamero(df, grid_df):
    synced = grid_df.copy()

    for sensor_name in ['Gyro', 'GameRo']:
        matched = assign_sensor_to_grid(df, sensor_name, grid_df)
        synced = pd.merge(
            synced,
            matched,
            on='time_s',
            how='left'
        )

    return synced


# =========================================================
# 9) 同期状況を表示する処理
# =========================================================

def print_sync_diagnostics(sync_df, side_label):
    print(f'\n=== {side_label}端末のGyro/GameRo同期状況 ===')
    print(f'共通グリッド点数: {len(sync_df)}')

    for sensor_name in ['Gyro', 'GameRo']:
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


# =========================================================
# 10) |Gyz| の移動平均を計算する処理
# =========================================================

def moving_average_abs_gyro(omega_z_deg_s, smooth_window=10):

    omega_z_deg_s = np.asarray(omega_z_deg_s, dtype=float)

    omega_abs_smooth = (
        pd.Series(np.abs(omega_z_deg_s))
        .rolling(
            window=smooth_window,
            center=False,
            min_periods=1
        )
        .mean()
        .to_numpy()
    )

    return omega_abs_smooth

# ----------絶対値化しない移動平均関数----------------
def moving_average_gyro(omega_z_deg_s, smooth_window=10):
    omega_z_deg_s = np.asarray(omega_z_deg_s, dtype=float)

    omega_smooth = (
        pd.Series(omega_z_deg_s)
        .rolling(
            window=smooth_window,
            center=False,
            min_periods=1
        )
        .mean()
        .to_numpy()
    )

    return omega_smooth


# =========================================================
# 11) 同期済みデータから回転後Gyzを計算する処理
# =========================================================

def compute_rotated_gyro_z(sync_df, initial_quat):
    """
    共通時刻グリッドに同期済みのGyroとGameRoから，
    初期姿勢基準座標系におけるZ軸角速度 Gyz [deg/s] を求める．
    """
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
        return pd.DataFrame(
            columns=[
                'time_s',
                'Gyz_deg_s',
                'Gyz_smooth_deg_s',
                'abs_Gyz_deg_s',
                'abs_Gyz_smooth_deg_s'
            ]
        )

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

    # AndroidのGyroは通常 rad/s なので deg/s に変換
    Gyz_deg_s = Gyz * 180.0 / math.pi

    # 生のGyzを絶対値化したもの
    abs_Gyz_deg_s = np.abs(Gyz_deg_s)

    # 生のGyzの移動平均
    Gyz_smooth_deg_s = moving_average_gyro(
        Gyz_deg_s,
        smooth_window=GYRO_SMOOTH_WINDOW
    )

    # |Gyz| の移動平均
    abs_Gyz_smooth = moving_average_abs_gyro(
        Gyz_deg_s,
        smooth_window=GYRO_SMOOTH_WINDOW
    )

    result = pd.DataFrame({
        'time_s': data['time_s'].to_numpy(),
        'Gyz_deg_s': Gyz_deg_s,
        'Gyz_smooth_deg_s': Gyz_smooth_deg_s,
        'abs_Gyz_deg_s': abs_Gyz_deg_s,
        'abs_Gyz_smooth_deg_s': abs_Gyz_smooth        
    })

    return result


# =========================================================
# 12) 方向変換時刻の縦線を追加する処理
# =========================================================

def add_cod_time_lines_to_plot(ax):
    for i, ct in enumerate(cod_times):
        label = '方向変換時刻' if i == 0 else None
        ax.axvline(
            ct,
            c='black',
            linestyle=':',
            alpha=0.9,
            label=label
        )


# =========================================================
# 13) 回転後Gyzと |Gyz| 平滑化値をプロットする処理
# =========================================================

def prepare_gyro_z_plot_df(gyro_z_df, hand_label):
    if gyro_z_df is None or len(gyro_z_df) == 0:
        print(f'{hand_label}: プロットできる角速度データがありません．')
        return None

    plot_df = (
        gyro_z_df
        .copy()
        .sort_values('time_s')
        .reset_index(drop=True)
    )

    if is_mask == 1:
        mask = (
            (plot_df['time_s'] >= limit_min_time)
            & (plot_df['time_s'] <= limit_max_time)
        )

        plot_df = plot_df.loc[mask].reset_index(drop=True)

    if len(plot_df) == 0:
        print(f'{hand_label}: 指定範囲内に角速度データがありません．')
        return None

    return plot_df


def plot_gyro_z_timeseries(
    gyro_z_L,
    gyro_z_R,
    left_label,
    right_label
):
    plot_items = [
        (prepare_gyro_z_plot_df(gyro_z_L, left_label), left_label),
        (prepare_gyro_z_plot_df(gyro_z_R, right_label), right_label)
    ]

    plot_specs = [
        (
            'Gyz_smooth_deg_s',
            'Gyz移動平均 [deg/s]',
            'Gyz移動平均',
            {'linewidth': 2, 'alpha': 0.9}
        ),
        (
            'abs_Gyz_smooth_deg_s',
            '|Gyz|移動平均 [deg/s]',
            '|Gyz|移動平均',
            {'linewidth': 2, 'alpha': 0.9}
        )
    ]

    fig, axes = plt.subplots(
        len(plot_specs),
        2,
        figsize=(16, 8),
        sharex=True,
        squeeze=False
    )
    fig.suptitle(
        f'角速度Z成分の移動平均（移動平均窓幅: {GYRO_SMOOTH_WINDOW}サンプル）',
        fontsize=16
    )

    for col_index, (plot_df, hand_label) in enumerate(plot_items):
        axes[0, col_index].set_title(hand_label)

        for row_index, (value_col, ylabel, series_label, plot_kwargs) in enumerate(plot_specs):
            ax = axes[row_index, col_index]

            if plot_df is not None:
                ax.plot(
                    plot_df['time_s'],
                    plot_df[value_col],
                    label=series_label,
                    **plot_kwargs
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    '表示できるデータがありません',
                    ha='center',
                    va='center',
                    transform=ax.transAxes
                )

            add_cod_time_lines_to_plot(ax)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend()

    for ax in axes[-1, :]:
        ax.set_xlabel('時間 [s]')

    plt.tight_layout()
    plt.show()


# =========================================================
# 14) メイン処理
# =========================================================

def main():
    # -----------------------------------------------------
    # 1．CSV読み込みと time_s 作成
    # -----------------------------------------------------
    df_L, initial_quat_L = load_sensor_file_with_time(
        file_L,
        time_offset_s=delay_time
    )

    df_R, initial_quat_R = load_sensor_file_with_time(
        file_R,
        time_offset_s=0.0
    )

    # -----------------------------------------------------
    # 2．共通時刻グリッド作成
    # -----------------------------------------------------
    grid_df, overlap_start, overlap_end = make_common_grid(
        df_L,
        df_R
    )

    print('\n=== 共通時刻グリッド ===')
    print(f'重複開始時刻: {overlap_start:.6f} s')
    print(f'重複終了時刻: {overlap_end:.6f} s')
    print(f'グリッド点数: {len(grid_df)}')
    print(f'サンプリング間隔: {SAMPLE_INTERVAL:.6f} s')

    # -----------------------------------------------------
    # 3．左右のGyro/GameRoを共通グリッドへ同期
    # -----------------------------------------------------
    sync_L = build_synced_gyro_gamero(df_L, grid_df)
    sync_R = build_synced_gyro_gamero(df_R, grid_df)

    if PRINT_SYNC_DIAGNOSTICS:
        print_sync_diagnostics(sync_L, '左')
        print_sync_diagnostics(sync_R, '右')

    # -----------------------------------------------------
    # 4．回転後Gyzを計算
    # -----------------------------------------------------
    gyro_z_L = compute_rotated_gyro_z(
        sync_L,
        initial_quat_L
    )

    gyro_z_R = compute_rotated_gyro_z(
        sync_R,
        initial_quat_R
    )

    # -----------------------------------------------------
    # 5．プロット
    # -----------------------------------------------------
    print('\n=== 角速度Z成分の移動平均を表示します ===')
    plot_gyro_z_timeseries(
        gyro_z_L,
        gyro_z_R,
        left_label='左手の端末',
        right_label='右手の端末'
    )
    

# =========================================================
# 15) 実行
# =========================================================

if __name__ == '__main__':
    main()


