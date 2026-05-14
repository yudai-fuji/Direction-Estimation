# --- 角速度の時刻間角度変化量プロット用 ---
# 目的:
#   GyroをGameRoでD座標系からW座標系へ変換し，
#   W座標系Z軸まわりの角速度を時間積分する．
#   ただし，累積はせず，各時刻間の角度変化量 dtheta のみをプロットする．

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib
except ImportError:
    pass


# =========================================================
# CSVファイル指定
# 右手CSVと左手CSVは別ファイルとして指定する
# =========================================================
file_L = 'LLLL_2.csv'
file_R = 'RRRR_2.csv'



# =========================================================
# GyroからW座標系Z軸まわりの角度変化量を計算する処理
# 戻り値:
#   time_s           : 開始時刻を0秒とした時間 [s]
#   omega_z_deg_s    : W座標系Z軸まわりの角速度 [deg/s]
#   dt_s             : 直前サンプルからの時間差 [s]
#   delta_angle_deg  : 時刻間の角度変化量 [°]
# =========================================================
def compute_wz_delta_angle_from_gyro(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f'{file_path} が見つかりませんでした')
        raise

    # センサ種別で分離
    gyro_df = df[df['Sensor'] == 'Gyro'].copy()
    gamerot_df = df[df['Sensor'] == 'GameRo'].copy()

    if gyro_df.empty:
        raise ValueError(f'{file_path} に Gyro データがありません')

    if gamerot_df.empty:
        raise ValueError(f'{file_path} に GameRo データがありません')

    # Timestampを基準に，直前のGameRoをGyroに結合
    merged_df = pd.merge_asof(
        gyro_df.sort_values('Timestamp'),
        gamerot_df.sort_values('Timestamp'),
        on='Timestamp',
        direction='backward',
        suffixes=('_gyro', '_ro')
    )

    # GameRoが結合できなかった行を削除
    merged_df = merged_df.dropna(subset=['X_ro', 'Y_ro', 'Z_ro', 'W_ro']).reset_index(drop=True)

    if merged_df.empty:
        raise ValueError(f'{file_path} で Gyro と GameRo を時刻同期できませんでした')

    # 端末座標系の角速度 [rad/s]
    gyx = merged_df['X_gyro']
    gyy = merged_df['Y_gyro']
    gyz = merged_df['Z_gyro']

    # Game Rotation Vectorのクォータニオン
    gx = merged_df['X_ro']
    gy = merged_df['Y_ro']
    gz = merged_df['Z_ro']
    gw = merged_df['W_ro']

    # 基準姿勢
    gx0, gy0, gz0, gw0 = gamerot_df.iloc[0][['X', 'Y', 'Z', 'W']]

    # 共役クォータニオン
    def kyoyaku(qx, qy, qz, qw):
        return -qx, -qy, -qz, qw

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

    # D座標系の角速度をW座標系へ変換
    Gyx = (
        (2 * gwc * gwc + 2 * gxc * gxc - 1) * gyx
        + (2 * gxc * gyc - 2 * gzc * gwc) * gyy
        + (2 * gxc * gzc + 2 * gyc * gwc) * gyz
    )

    Gyy = (
        (2 * gxc * gyc + 2 * gzc * gwc) * gyx
        + (2 * gwc * gwc + 2 * gyc * gyc - 1) * gyy
        + (2 * gyc * gzc - 2 * gxc * gwc) * gyz
    )

    Gyz = (
        (2 * gxc * gzc - 2 * gyc * gwc) * gyx
        + (2 * gyc * gzc + 2 * gxc * gwc) * gyy
        + (2 * gwc * gwc + 2 * gzc * gzc - 1) * gyz
    )

    # rad/s から deg/s へ変換
    Gyx = Gyx * 180.0 / math.pi
    Gyy = Gyy * 180.0 / math.pi
    Gyz = Gyz * 180.0 / math.pi

    merged_df['X_rotated'] = Gyx
    merged_df['Y_rotated'] = Gyy
    merged_df['Z_rotated'] = Gyz

    merged_df = merged_df.dropna(subset=['X_rotated', 'Y_rotated', 'Z_rotated']).reset_index(drop=True)

    # 時間 [s]
    s_timestamp = merged_df['Timestamp'] / 1_000_000_000.0
    time_s = s_timestamp - s_timestamp.iloc[0]

    # サンプリング間隔 [s]
    time_np = time_s.to_numpy()
    dt_s = np.diff(time_np, prepend=time_np[0])

    # W座標系Z軸まわりの角速度 [deg/s]
    omega_z_deg_s = merged_df['Z_rotated'].to_numpy()

    # 時刻間の角度変化量 [°]
    # 符号付きで保持する
    delta_angle_deg = omega_z_deg_s * dt_s

    result_df = pd.DataFrame({
        'time_s': time_s,
        'omega_z_deg_s': omega_z_deg_s,
        'dt_s': dt_s,
        'delta_angle_deg': delta_angle_deg
    })

    return result_df



# =========================================================
# 指定した時刻区間の角度変化量を累積する処理
# 注意:
#   delta_angle_deg は「直前サンプルから現在サンプルまでの角度変化量」である．
#   ここでは，time_s が start_time_s 以上 end_time_s 以下の行を対象に合計する．
# 戻り値:
#   accumulated_angle_deg : 指定区間内の符号付き累積角度 [°]
#   section_df            : 指定区間に含まれるデータ
# =========================================================
def accumulate_delta_angle_in_range(delta_df, start_time_s, end_time_s):
    if start_time_s > end_time_s:
        raise ValueError('start_time_s は end_time_s 以下にしてください')

    section_df = delta_df[
        (delta_df['time_s'] >= start_time_s)
        & (delta_df['time_s'] <= end_time_s)
    ].copy()

    if section_df.empty:
        raise ValueError(
            f'{start_time_s:.3f}s から {end_time_s:.3f}s の範囲にデータがありません'
        )

    accumulated_angle_deg = section_df['delta_angle_deg'].sum()

    return accumulated_angle_deg, section_df


# =========================================================
# 指定した時刻区間の累積角度を表示する処理
# =========================================================
def print_accumulated_angle_in_range(delta_df, hand_label, start_time_s, end_time_s):
    accumulated_angle_deg, section_df = accumulate_delta_angle_in_range(
        delta_df,
        start_time_s,
        end_time_s
    )

    print(f'\n=== {hand_label}端末: 指定区間の累積角度 ===')
    print(f'区間: {start_time_s:.3f} s ～ {end_time_s:.3f} s')
    print(f'データ数: {len(section_df)}')
    print(f'累積角度: {accumulated_angle_deg:.4f} [°]')

    return accumulated_angle_deg, section_df

# =========================================================
# Wz軸まわり角度変化量を1端末分だけプロットする処理
# 左右を同じグラフには描かない
# =========================================================
def plot_wz_delta_angle(delta_df, hand_label):
    # 色設定
    # 左手: 青，右手: 赤
    if '左' in hand_label:
        plot_color = 'blue'
    elif '右' in hand_label:
        plot_color = 'red'
    else:
        plot_color = 'black'

    plt.figure(figsize=(10, 6))

    plt.plot(
        delta_df['time_s'],
        delta_df['delta_angle_deg'],
        color=plot_color,
        linewidth=1.5,
        label=f'{hand_label}の端末'
    )

    # 0°の基準線
    plt.axhline(0.0, linestyle='--', linewidth=1.0, alpha=0.7)

    plt.xlabel('時間 [s]')
    plt.ylabel('角度変化量 [°]')
    plt.title(f'{hand_label}端末のWz軸まわり角度変化量')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# メイン処理
# =========================================================
if __name__ == '__main__':
    left_delta_df = compute_wz_delta_angle_from_gyro(file_L)
    right_delta_df = compute_wz_delta_angle_from_gyro(file_R)

    print('\n=== 左手端末 ===')
    print(left_delta_df[['time_s', 'omega_z_deg_s', 'dt_s', 'delta_angle_deg']].head())

    print('=== 右手端末 ===')
    print(right_delta_df[['time_s', 'omega_z_deg_s', 'dt_s', 'delta_angle_deg']].head())

    

    # =====================================================
    # 指定区間の累積角度を確認する場合
    # ここを確認したい時刻に変更する
    # =====================================================

    

    start_time_s = 5.00
    end_time_s = 6.42

    print_accumulated_angle_in_range(
        left_delta_df,
        '左手',
        start_time_s,
        end_time_s
    )

    start_time_s = 5.00
    end_time_s = 6.25

    print_accumulated_angle_in_range(
        right_delta_df,
        '右手',
        start_time_s,
        end_time_s
    )

    
    plot_wz_delta_angle(left_delta_df, '左手')
    plot_wz_delta_angle(right_delta_df, '右手')
    
