# --- 右手・左手・左右平均の角速度累積法とRMS誤差 ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import japanize_matplotlib

file_R = 'HwR2.csv'
file_L = 'HL2.csv'
cod_time = 21.99 #方向転換した時間
delay_time = 3.06 #計測時間のずれ
limit_min_time = 12.25 #時間の適用範囲の下限
limit_max_time = 27.7 #時間の適用範囲の上限


def compute_heading(file_path, initial_heading_deg):
  
    # CSV 読み込み
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f'{file_path} が見つかりませんでした')
        raise

    # Gyro と GameRo を抽出
    gyro_df = df[df['Sensor'] == 'Gyro'].copy()
    gamerot_df = df[df['Sensor'] == 'GameRo'].copy()

    # 時刻で同期（Gyro を基準に最も近い GameRo を結合）
    merged_df = pd.merge_asof(
        gyro_df.sort_values('Timestamp'),
        gamerot_df.sort_values('Timestamp'),
        on='Timestamp',
        direction='backward',
        suffixes=('_gyro', '_ro')
    )

    # 角速度 (D座標系)
    gyx = merged_df['X_gyro']
    gyy = merged_df['Y_gyro']
    gyz = merged_df['Z_gyro']

    # クォータニオン (GameRotationVector)
    gx = merged_df['X_ro']
    gy = merged_df['Y_ro']
    gz = merged_df['Z_ro']
    gw = merged_df['W_ro']

    # 初期姿勢のクォータニオン（GameRoの1行目）
    gx0, gy0, gz0, gw0 = gamerot_df.iloc[0][['X', 'Y', 'Z', 'W']]

    # 共役な四元数
    def kyoyaku(gx, gy, gz, gw):
        return (-gx, -gy, -gz, gw)

    Gx0, Gy0, Gz0, Gw0 = kyoyaku(gx0, gy0, gz0, gw0)

    # 相対クォータニオン
    gwc = gw*Gw0 - gx*Gx0 - gy*Gy0 - gz*Gz0
    gxc = gw*Gx0 + gx*Gw0 - gy*Gz0 + gz*Gy0
    gyc = gw*Gy0 + gx*Gz0 + gy*Gw0 - gz*Gx0
    gzc = gw*Gz0 - gx*Gy0 + gy*Gx0 + gz*Gw0

    # 回転行列による D→W 変換（角速度ベクトルの回転）
    Gyx = (2*gwc*gwc + 2*gxc*gxc - 1)*gyx + (2*gxc*gyc - 2*gzc*gwc)*gyy + (2*gxc*gzc + 2*gyc*gwc)*gyz
    Gyy = (2*gxc*gyc + 2*gzc*gwc)*gyx + (2*gwc*gwc + 2*gyc*gyc - 1)*gyy + (2*gyc*gzc - 2*gxc*gwc)*gyz
    Gyz = (2*gxc*gzc - 2*gyc*gwc)*gyx + (2*gyc*gzc + 2*gxc*gwc)*gyy + (2*gwc*gwc + 2*gzc*gzc - 1)*gyz

    # 単位を rad/s → deg/s に変換
    Gyx = Gyx * 180.0 / math.pi
    Gyy = Gyy * 180.0 / math.pi
    Gyz = Gyz * 180.0 / math.pi

    merged_df['X_rotated'] = Gyx
    merged_df['Y_rotated'] = Gyy
    merged_df['Z_rotated'] = Gyz

    # NaN を含む行を削除
    merged_df = merged_df.dropna(
        subset=['X_rotated', 'Y_rotated', 'Z_rotated']
    ).reset_index(drop=True)
    
    # 時刻[s] に変換して 0始まりに
    s_timestamp = merged_df['Timestamp'] / 1_000_000_000.0

    t = s_timestamp - s_timestamp.iloc[0]

    # Δt の計算（先頭は0）
    #to_numpy()で1次元配列へ
    t_np = t.to_numpy()

    #prependで先頭に，指定した値をつけてから差分の計算(データ数を減らさないため)をスタート
    dt = np.diff(t_np, prepend=t_np[0])

    # Wz軸角速度 [°/s]　を1次元配列へ
    omega_z = merged_df['Z_rotated'].to_numpy()

    # Δθ_i = ω_z,i * Δt_i
    dtheta = omega_z * dt

    # 初期角度を加えて累積
    theta = initial_heading_deg + np.cumsum(dtheta)

    # 結果をdfで返す．列がtime_s,theta_degでindexが0,1,2...
    result = pd.DataFrame({
        'time_s': t,
        'theta_deg': theta
        })
    
    return result

#それぞれdfが格納
heading_R = compute_heading(file_R, initial_heading_deg=90.0)  # 右手
heading_L = compute_heading(file_L, initial_heading_deg=90.0)  # 左手

#計測時間の差を埋める
heading_L['time_s'] = heading_L['time_s'] + delay_time

# 念のため time_s でソート
heading_R = heading_R.sort_values('time_s').reset_index(drop=True)
heading_L = heading_L.sort_values('time_s').reset_index(drop=True)

#heading_Rと_Lを結合
aligned = pd.merge_asof(
    heading_R,     # left: 基準（右手）
    heading_L,     # right: 合わせる側（左手）
    on='time_s',   # time_s でマージ
    direction='backward',
    suffixes=('_R', '_L'),
    tolerance=0.05   # 許容差(timeR - timeL) 0.05 s（必要なら調整）
)

# マッチできなかった行（NaN）は削除
aligned = aligned.dropna(subset=['theta_deg_R', 'theta_deg_L']).reset_index(drop=True)

# 左右平均 θ
aligned['theta_mean'] = (aligned['theta_deg_R'] + aligned['theta_deg_L']) / 2.0

#time_s列が 0s ~ 13s ならTrue,それ以外はFalseを返す
mask_eval = (aligned['time_s'] >= limit_min_time) & (aligned['time_s'] <= limit_max_time)

#trueの範囲の各データを変数へ格納
t_eval          = aligned.loc[mask_eval, 'time_s'].to_numpy()
theta_R_eval    = aligned.loc[mask_eval, 'theta_deg_R'].to_numpy()   # 右手
theta_L_eval    = aligned.loc[mask_eval, 'theta_deg_L'].to_numpy()   # 左手
theta_mean_eval = aligned.loc[mask_eval, 'theta_mean'].to_numpy()    # 左右平均

# 真値とそれ以降
true_heading_masked = np.where(t_eval <= cod_time, 90.0, 0.0)

# 推定角度 - 真値
err_R    = theta_R_eval    - true_heading_masked   # 右手
err_L    = theta_L_eval    - true_heading_masked   # 左手
err_mean = theta_mean_eval - true_heading_masked   # 左右平均

# RMS 誤差 (平均をとって平方根)
RMS_R_deg    = np.sqrt(np.mean(err_R**2))
RMS_L_deg    = np.sqrt(np.mean(err_L**2))
RMS_mean_deg = np.sqrt(np.mean(err_mean**2))

print(f"右手の端末のRMS誤差:      {RMS_R_deg:.3f} [deg]")
print(f"左手の端末のRMS誤差:      {RMS_L_deg:.3f} [deg]")
print(f"左右平均のRMS誤差: {RMS_mean_deg:.3f} [deg]")

t_plot          = aligned['time_s'].to_numpy()
theta_R_plot    = aligned['theta_deg_R'].to_numpy()
theta_L_plot    = aligned['theta_deg_L'].to_numpy()
theta_mean_plot = aligned['theta_mean'].to_numpy()

true_heading_all = np.where(t_plot <= cod_time, 90.0, 0.0)

# --- プロットの設定 ---
plt.figure(figsize=(10, 6))

plt.plot(t_plot, theta_R_plot,
         label='右手の端末', alpha=0.8, c = 'r', linestyle='-')
plt.plot(t_plot, theta_L_plot,
         label='左手の端末', alpha=0.8, c = 'b', linestyle='-')
plt.plot(t_plot, theta_mean_plot,
         label='左右平均', c='g', linewidth=2, alpha = 0.8, linestyle='-')
plt.plot(t_plot, true_heading_all,
         label='真値', c='k', linestyle='--', alpha = 0.8)

plt.xlabel('時刻 (s)')
plt.ylabel('方位 (°)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
