import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import japanize_matplotlib

# ====== 入力（あなたが設定） ======
file_L = 'HL6.csv'
file_R = 'HR6.csv'
delay_time = 5.056
cod_time = 22.06
limit_min_time = 10.85
limit_max_time = 28.16
WINDOW_SIZE = 40

# ECDF表示範囲（見やすさ用。分布を壊したくないのでclipはしない）
PCA_X_MAX_PLOT = 60
GYRO_X_MAX_PLOT = 30


# =========================================================
# 共通：時刻同期して、評価区間内の「絶対誤差 |推定-真値|」配列を返す
# 返り値: err_abs_R, err_abs_L, err_abs_mean (いずれも1次元np.array)
# =========================================================
def abs_errors_from_headings(heading_R, heading_L):
    heading_L = heading_L.copy()
    heading_R = heading_R.copy()

    # 時刻ずれ補正（あなたのコードと同じ）
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

    t_all = aligned['time_s'].to_numpy()
    mask = (t_all >= limit_min_time) & (t_all <= limit_max_time)

    # eval...evaluation(評価)，評価区間の意味
    t_eval = t_all[mask]
    theta_R_eval = aligned.loc[mask, 'theta_deg_R'].to_numpy()
    theta_L_eval = aligned.loc[mask, 'theta_deg_L'].to_numpy()
    theta_mean_eval = (theta_R_eval + theta_L_eval) / 2.0

    # 真値（あなたのコードと同じ）
    true_heading_eval = np.where(t_eval <= cod_time, 90.0, 0.0)

    # 絶対誤差（wrapなし：今のまま）
    err_abs_R = np.abs(theta_R_eval - true_heading_eval)
    err_abs_L = np.abs(theta_L_eval - true_heading_eval)
    err_abs_mean = np.abs(theta_mean_eval - true_heading_eval)

    return err_abs_R, err_abs_L, err_abs_mean


# =========================================================
# CDF描画
# =========================================================
def plot_ecdf_three(err_R, err_L, err_M, title_str, x_max_plot=None):
    def ecdf(y):
    
        # Numpy配列(ndarray)として扱える形に変換する
        y = np.asarray(y)

        # isnan()で各要素がNanかどうかを判定し，Nanの要素は削除して返す．[~np.]の~は否定
        y = y[~np.isnan(y)]
    
        # 昇順に並べる
        y_sorted = np.sort(y)
        
        # 1 ~ len(y_sorted)までを小さいものから順位付けされた数列(1,2,3,,,)を作り，サンプル数で割ることでCDFを求めている
        p = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
        
        return y_sorted, p

    # 左(x)が誤差を昇順に並べたもので，右(p)がCDFの結果
    xR, pR = ecdf(err_R)
    xL, pL = ecdf(err_L)
    xM, pM = ecdf(err_M)

    plt.figure(figsize=(10, 6))
    plt.step(xR, pR, where='post', color='r', linewidth=2.0)
    plt.step(xL, pL, where='post', color='b', linewidth=2.0)
    plt.step(xM, pM, where='post', color='g', linewidth=2.0)

    plt.xlabel('絶対誤差 [°]')
    plt.ylabel('累積確率')
    plt.title(title_str)
    plt.grid(True)
    plt.ylim(0, 1)

    if x_max_plot is not None:
        plt.xlim(0, x_max_plot)

    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# PCA法：CSV -> (time_s, theta_deg)
# =========================================================
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

    merged_df = merged_df.assign(X_rotated=Ax, Y_rotated=Ay, Z_rotated=Az)
    merged_df = merged_df.dropna(subset=['X_rotated', 'Y_rotated', 'Z_rotated']).reset_index(drop=True)

    time_data = merged_df['Timestamp'].values
    time_start_ns = time_data[0]

    x_rotated = merged_df['X_rotated'].values
    y_rotated = merged_df['Y_rotated'].values

    pca = PCA(n_components=1)

    time_list = []
    deg_list = []

    for frame in range(len(x_rotated)):
        if frame + 1 < window_size:
            continue

        start_index = frame - (window_size - 1)
        end_index   = frame + 1

        data_window = np.column_stack((x_rotated[start_index:end_index], y_rotated[start_index:end_index]))
        pca.fit(data_window)
        vx, vy = pca.components_[0]

        angle_deg = np.degrees(np.arctan2(vy, vx))
        t_sec = (time_data[frame] - time_start_ns) / 1_000_000_000
        time_list.append(t_sec)
        deg_list.append(angle_deg)

    return pd.DataFrame({'time_s': time_list, 'theta_deg': deg_list})


# =========================================================
# 角速度累積法：CSV -> (time_s, theta_deg)
# =========================================================
def compute_heading(file_path, initial_heading_deg):
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

    merged_df = merged_df.assign(X_rotated=Gyx, Y_rotated=Gyy, Z_rotated=Gyz)
    merged_df = merged_df.dropna(subset=['X_rotated', 'Y_rotated', 'Z_rotated']).reset_index(drop=True)

    s_timestamp = merged_df['Timestamp'] / 1_000_000_000.0
    t = s_timestamp - s_timestamp.iloc[0]

    t_np = t.to_numpy()
    dt = np.diff(t_np, prepend=t_np[0])

    omega_z = merged_df['Z_rotated'].to_numpy()  # deg/s
    dtheta = omega_z * dt
    theta = initial_heading_deg + np.cumsum(dtheta)

    return pd.DataFrame({'time_s': t, 'theta_deg': theta})


# =========================================================
# 2手法融合：推定角度を時刻ごとに平均
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
    return pd.DataFrame({'time_s': fused['time_s'], 'theta_deg': fused_theta})

def ratio(err,deg = 5):
    err = np.asarray(err)
    err = err[~np.isnan(err)]
    if len(err) == 0:
        return np.nan
    return np.mean(err <= deg)*100

def abs_error_at_cdf(err,cdf=0.8):
    err = err[~np.isnan(err)]
    if len(err) == 0:
        return np.nan
    err = np.sort(err)
    p = np.arange(1, len(err)+1) / len(err)
    idx = np.searchsorted(p,cdf)
    return err[idx]


# =========================================================
# 1) PCA のECDF
# =========================================================
heading_L_pca = compute_heading_from_file(file_L, window_size=WINDOW_SIZE)
heading_R_pca = compute_heading_from_file(file_R, window_size=WINDOW_SIZE)

errR, errL, errM = abs_errors_from_headings(heading_R_pca, heading_L_pca)

print('')
print('PCA方位推定法')
print('絶対誤差が5°となる確率')
print(f'Right : {ratio(errR, 5):.1f}% ')
print(f'Left : {ratio(errL, 5):.1f}% ')
print(f'Mean : {ratio(errM, 5):.1f}% ')

print('累積確率が80%となる絶対誤差')
print(f'Right : {abs_error_at_cdf(errR, 0.8):.1f}° ')
print(f'Left : {abs_error_at_cdf(errL, 0.8):.1f}° ')
print(f'Mean : {abs_error_at_cdf(errM, 0.8):.1f}° ')

plot_ecdf_three(errR, errL, errM, title_str='PCA方位推定法：絶対誤差のCDF', x_max_plot=PCA_X_MAX_PLOT)


# =========================================================
# 2) 角速度累積 のECDF
# =========================================================
heading_R_gyro = compute_heading(file_R, initial_heading_deg=90.0)
heading_L_gyro = compute_heading(file_L, initial_heading_deg=90.0)

errR, errL, errM = abs_errors_from_headings(heading_R_gyro, heading_L_gyro)

print('')
print('角速度累積法')
print('絶対誤差が5°となる確率')
print(f'Right : {ratio(errR, 5):.1f}% ')
print(f'Left : {ratio(errL, 5):.1f}% ')
print(f'Mean : {ratio(errM, 5):.1f}% ')

print('累積確率が80%となる絶対誤差')
print(f'Right : {abs_error_at_cdf(errR, 0.8):.1f}° ')
print(f'Left : {abs_error_at_cdf(errL, 0.8):.1f}° ')
print(f'Mean : {abs_error_at_cdf(errM, 0.8):.1f}° ')

plot_ecdf_three(errR, errL, errM, title_str='角速度累積法：絶対誤差のCDF', x_max_plot=GYRO_X_MAX_PLOT)


# =========================================================
# 3) 2手法平均 のECDF
# =========================================================
heading_R_avg = fuse_two_methods_heading(heading_R_pca, heading_R_gyro)
heading_L_avg = fuse_two_methods_heading(heading_L_pca, heading_L_gyro)

errR, errL, errM = abs_errors_from_headings(heading_R_avg, heading_L_avg)
#plot_ecdf_three(errR, errL, errM, title_str='2手法平均：絶対誤差のCDF', x_max_plot=GYRO_X_MAX_PLOT)