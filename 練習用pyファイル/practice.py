import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# =========================================================
# 設定
# =========================================================
FILE_PATH = r"研究用csvファイル/NLS036_SelfPace.csv"

# PCA固定窓
PCA_WINDOW = 40

# 初期進行方向 [deg]
# 分かっている場合は 0, 90, -90 などを設定する。
# Noneなら最初のPCA結果から自動設定する。
INITIAL_HEADING_DEG = 90


# =========================================================
# 角度処理
# =========================================================

def wrap_pm180(angle):
    """角度を [-180, 180) にする"""
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def angle_diff(a, b):
    """aとbの最小角度差"""
    return wrap_pm180(a - b)


def circular_mean_two(angle1, angle2):
    """2つの角度の円周平均"""
    rad1 = np.radians(angle1)
    rad2 = np.radians(angle2)

    x = np.cos(rad1) + np.cos(rad2)
    y = np.sin(rad1) + np.sin(rad2)

    return np.degrees(np.arctan2(y, x))


# =========================================================
# 加速度PCA
# =========================================================

def compute_pca_heading(east, north, window_size):
    """
    FreeAcc_E/N に固定窓PCAを適用する。
    この段階では180°の方向不確定性が残っている。
    """

    headings = np.full(len(east), np.nan)

    pca = PCA(n_components=2)

    for i in range(window_size - 1, len(east)):

        window = np.column_stack([
            east[i - window_size + 1:i + 1],
            north[i - window_size + 1:i + 1]
        ])

        if np.isnan(window).any():
            continue

        pca.fit(window)

        # 第一主成分
        vx, vy = pca.components_[0]

        # E軸=0°, N軸=90°
        heading = np.degrees(np.arctan2(vy, vx))

        headings[i] = heading

    return headings


# =========================================================
# 角速度累積法
# =========================================================

def gyro_to_world_up(gx, gy, gz, roll, pitch, yaw):
    """
    D座標系の角速度をW座標系へ変換し、
    W座標系の鉛直方向(U)の角速度を取得する。

    Roll/Pitch/Yaw : deg
    Gyro           : rad/s
    """

    phi = np.radians(roll)
    theta = np.radians(pitch)
    psi = np.radians(yaw)

    world_up_gyro = np.full(len(gx), np.nan)

    for i in range(len(gx)):

        if np.any(np.isnan([
            gx[i], gy[i], gz[i],
            phi[i], theta[i], psi[i]
        ])):
            continue

        # Rx
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi[i]), -np.sin(phi[i])],
            [0, np.sin(phi[i]),  np.cos(phi[i])]
        ])

        # Ry
        Ry = np.array([
            [ np.cos(theta[i]), 0, np.sin(theta[i])],
            [0, 1, 0],
            [-np.sin(theta[i]), 0, np.cos(theta[i])]
        ])

        # Rz
        Rz = np.array([
            [np.cos(psi[i]), -np.sin(psi[i]), 0],
            [np.sin(psi[i]),  np.cos(psi[i]), 0],
            [0, 0, 1]
        ])

        # D → W
        R = Rz @ Ry @ Rx

        gyro_d = np.array([
            gx[i],
            gy[i],
            gz[i]
        ])

        gyro_w = R @ gyro_d

        # ENUのU成分
        world_up_gyro[i] = gyro_w[2]

    return world_up_gyro


def integrate_gyro_heading(time, gyro_up, initial_heading):
    """W座標系鉛直角速度を時間積分する"""

    heading = np.full(len(time), np.nan)

    heading[0] = initial_heading

    for i in range(1, len(time)):

        if np.isnan(gyro_up[i]):
            heading[i] = heading[i - 1]
            continue

        dt = time[i] - time[i - 1]

        # rad/s → deg/s
        omega_deg = np.degrees(gyro_up[i])

        heading[i] = heading[i - 1] + omega_deg * dt

    return heading


# =========================================================
# PCAの180°不確定性を角速度累積法で解消
# =========================================================

def resolve_pca_180(pca_heading, gyro_heading):

    corrected = np.full(len(pca_heading), np.nan)

    for i in range(len(pca_heading)):

        if np.isnan(pca_heading[i]) or np.isnan(gyro_heading[i]):
            continue

        # PCAが示す2候補
        candidate1 = pca_heading[i]
        candidate2 = pca_heading[i] + 180.0

        # 角速度累積法に近い方を採用
        error1 = abs(angle_diff(candidate1, gyro_heading[i]))
        error2 = abs(angle_diff(candidate2, gyro_heading[i]))

        if error1 <= error2:
            corrected[i] = candidate1
        else:
            corrected[i] = candidate2

    return wrap_pm180(corrected)


# =========================================================
# Main
# =========================================================

df = pd.read_csv(FILE_PATH)


# ---------------------------------------------------------
# Timeを数値化
# ---------------------------------------------------------
df["Time"] = (
    df["Time"]
    .astype(str)
    .str.replace(" sec", "", regex=False)
    .astype(float)
)


# ---------------------------------------------------------
# 最初のWalkから解析
# ---------------------------------------------------------
walk_indices = df.index[df["GeneralEvent"] == "Walk"]

if len(walk_indices) == 0:
    raise ValueError("Walkデータがありません。")

start_index = walk_indices[0]

df = df.loc[start_index:].reset_index(drop=True)

time = df["Time"].to_numpy()

# 時刻を0秒始まりにする
time = time - time[0]


# =========================================================
# 1. 左右の加速度PCA
# =========================================================

pca_R = compute_pca_heading(
    df["R_Wrist_FreeAcc_E"].to_numpy(),
    df["R_Wrist_FreeAcc_N"].to_numpy(),
    PCA_WINDOW
)

pca_L = compute_pca_heading(
    df["L_Wrist_FreeAcc_E"].to_numpy(),
    df["L_Wrist_FreeAcc_N"].to_numpy(),
    PCA_WINDOW
)


# =========================================================
# 2. 初期方向を決める
# =========================================================

first_valid = np.where(
    np.isfinite(pca_R) & np.isfinite(pca_L)
)[0][0]

if INITIAL_HEADING_DEG is None:

    # 左PCAについて180°違う2候補から
    # 右PCAに近い方を選ぶ
    left1 = pca_L[first_valid]
    left2 = pca_L[first_valid] + 180.0

    if abs(angle_diff(left2, pca_R[first_valid])) < \
       abs(angle_diff(left1, pca_R[first_valid])):
        left_initial = left2
    else:
        left_initial = left1

    # 左右の円周平均
    initial_heading = circular_mean_two(
        pca_R[first_valid],
        left_initial
    )

else:
    initial_heading = INITIAL_HEADING_DEG


print(f"初期方向: {initial_heading:.2f} deg")


# =========================================================
# 3. 左右の角速度をW座標系へ変換
# =========================================================

gyro_up_R = gyro_to_world_up(
    df["R_Wrist_Gyr_X"].to_numpy(),
    df["R_Wrist_Gyr_Y"].to_numpy(),
    df["R_Wrist_Gyr_Z"].to_numpy(),
    df["R_Wrist_Roll"].to_numpy(),
    df["R_Wrist_Pitch"].to_numpy(),
    df["R_Wrist_Yaw"].to_numpy()
)

gyro_up_L = gyro_to_world_up(
    df["L_Wrist_Gyr_X"].to_numpy(),
    df["L_Wrist_Gyr_Y"].to_numpy(),
    df["L_Wrist_Gyr_Z"].to_numpy(),
    df["L_Wrist_Roll"].to_numpy(),
    df["L_Wrist_Pitch"].to_numpy(),
    df["L_Wrist_Yaw"].to_numpy()
)


# =========================================================
# 4. 角速度累積法
# =========================================================

gyro_heading_R = integrate_gyro_heading(
    time,
    gyro_up_R,
    initial_heading
)

gyro_heading_L = integrate_gyro_heading(
    time,
    gyro_up_L,
    initial_heading
)


# 左右角速度累積結果も円周平均
gyro_heading_mean = circular_mean_two(
    gyro_heading_R,
    gyro_heading_L
)


# =========================================================
# 5. PCAの180°不確定性を解消
# =========================================================

heading_R = resolve_pca_180(
    pca_R,
    gyro_heading_mean
)

heading_L = resolve_pca_180(
    pca_L,
    gyro_heading_mean
)


# =========================================================
# 6. 左右の円周平均
# =========================================================

heading_mean = circular_mean_two(
    heading_R,
    heading_L
)


# =========================================================
# 7. プロット
# =========================================================

plt.figure(figsize=(12, 6))

plt.scatter(
    time,
    heading_R,
    label="Right wrist",
    s=8
)

plt.scatter(
    time,
    heading_L,
    label="Left wrist",
    s=8
)

plt.scatter(
    time,
    heading_mean,
    label="Circular mean",
    s=8
)

plt.xlabel("Time [s]")
plt.ylabel("Estimated heading [deg]")

plt.ylim(-180, 180)
plt.yticks([-180, -90, 0, 90, 180])

plt.grid()
plt.legend()

plt.tight_layout()
plt.show()