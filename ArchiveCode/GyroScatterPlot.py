# 角速度センサ値をプロット

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Meiryo"


# =========================
# 設定
# =========================
csv_path = "stra2_L.csv"  

# プロット範囲を秒で指定．None にすると全範囲
plot_min_time = 5.44
plot_max_time = 16.18


# =========================
# 列名補正
# =========================
def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"列が見つかりませんでした．候補: {candidates}")


# =========================
# クォータニオン関数
# 形式は [x, y, z, w]
# =========================
def normalize_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm == 0, np.nan, norm)
    return q / norm


def quat_conjugate(q):
    q = np.asarray(q, dtype=np.float64)
    qc = q.copy()
    qc[..., 0] *= -1.0
    qc[..., 1] *= -1.0
    qc[..., 2] *= -1.0
    return qc


def quat_multiply(q1, q2):
    """
    q1 ⊗ q2
    q1, q2 は shape=(4,) または shape=(N,4)
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.stack([x, y, z, w], axis=-1)


def rotate_vectors_by_quaternion(v, q):
    """
    ベクトル v をクォータニオン q で回転する．
    v: shape=(N,3)
    q: shape=(N,4) または shape=(4,)
    """
    v = np.asarray(v, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q = normalize_quaternion(q)

    if q.ndim == 1:
        q = np.repeat(q[np.newaxis, :], len(v), axis=0)

    q_xyz = q[:, :3]
    q_w = q[:, 3:4]

    t = 2.0 * np.cross(q_xyz, v)
    v_rot = v + q_w * t + np.cross(q_xyz, t)
    return v_rot


# =========================
# CSV読み込み
# =========================
try:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding="cp932")

timestamp_col = find_column(df, ["Timestamp", "Timestap", "timestamp", "timestap"])
sensor_col = find_column(df, ["Sensor", "Senser", "sensor", "senser"])
x_col = find_column(df, ["X", "x"])
y_col = find_column(df, ["Y", "y"])
z_col = find_column(df, ["Z", "z"])
w_col = find_column(df, ["W", "w"])

df = df.rename(columns={
    timestamp_col: "Timestamp",
    sensor_col: "Sensor",
    x_col: "X",
    y_col: "Y",
    z_col: "Z",
    w_col: "W",
})

# 数値変換
for c in ["Timestamp", "X", "Y", "Z", "W"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Sensor"] = df["Sensor"].astype(str).str.strip().str.lower()

# =========================
# Gyro と GameRo を抽出
# =========================
gyro_df = df[df["Sensor"] == "gyro"][["Timestamp", "X", "Y", "Z"]].copy()
gamero_df = df[df["Sensor"] == "gamero"][["Timestamp", "X", "Y", "Z", "W"]].copy()

gyro_df = gyro_df.dropna(subset=["Timestamp", "X", "Y", "Z"]).sort_values("Timestamp").reset_index(drop=True)
gamero_df = gamero_df.dropna(subset=["Timestamp", "X", "Y", "Z", "W"]).sort_values("Timestamp").reset_index(drop=True)

if len(gyro_df) == 0:
    raise ValueError("Gyroデータが見つかりませんでした．")
if len(gamero_df) == 0:
    raise ValueError("GameRoデータが見つかりませんでした．")

# ユーザー要望どおり，変数としても保持
gyro_timestamp = gyro_df["Timestamp"].to_numpy()
gyro_x = gyro_df["X"].to_numpy()
gyro_y = gyro_df["Y"].to_numpy()
gyro_z = gyro_df["Z"].to_numpy()

gamero_timestamp = gamero_df["Timestamp"].to_numpy()
gamero_x = gamero_df["X"].to_numpy()
gamero_y = gamero_df["Y"].to_numpy()
gamero_z = gamero_df["Z"].to_numpy()
gamero_w = gamero_df["W"].to_numpy()

# =========================
# backward結合
# Gyroの各時刻に対して，その時刻以前で最も近いGameRoを結合
# =========================
merged = pd.merge_asof(
    gyro_df.sort_values("Timestamp"),
    gamero_df.sort_values("Timestamp"),
    on="Timestamp",
    direction="backward",
    suffixes=("_gyro", "_gamero")
)

merged = merged.dropna(subset=["X_gamero", "Y_gamero", "Z_gamero", "W"]).reset_index(drop=True)

# merge_asof後，GameRo側のW列名は "W" のまま残る
merged = merged.rename(columns={"W": "W_gamero"})

# =========================
# 初期姿勢補正
# 最初のGameRoを基準にして，初期姿勢を identity に戻す
# q_rel(t) = q0^{-1} ⊗ q(t)
# =========================
q0 = gamero_df.loc[0, ["X", "Y", "Z", "W"]].to_numpy(dtype=np.float64)
q0 = normalize_quaternion(q0)
q0_inv = quat_conjugate(q0)

q_t = merged[["X_gamero", "Y_gamero", "Z_gamero", "W_gamero"]].to_numpy(dtype=np.float64)
q_t = normalize_quaternion(q_t)

q0_inv_batch = np.repeat(q0_inv[np.newaxis, :], len(q_t), axis=0)
q_rel = quat_multiply(q0_inv_batch, q_t)
q_rel = normalize_quaternion(q_rel)

# =========================
# Gyroを世界座標系へ変換
# w_world = R_rel @ w_device
# =========================
gyro_device = merged[["X_gyro", "Y_gyro", "Z_gyro"]].to_numpy(dtype=np.float64)
gyro_world = rotate_vectors_by_quaternion(gyro_device, q_rel)

merged["Wx"] = gyro_world[:, 0]
merged["Wy"] = gyro_world[:, 1]
merged["Wz"] = gyro_world[:, 2]

# =========================
# 単位変換（rad → deg）
# =========================
merged["Wx_deg"] = np.rad2deg(merged["Wx"])
merged["Wy_deg"] = np.rad2deg(merged["Wy"])
merged["Wz_deg"] = np.rad2deg(merged["Wz"])

# =========================
# 時間軸を秒に変換
# 先頭時刻を 0 秒にそろえる
# =========================
merged["time_s"] = (merged["Timestamp"] - merged["Timestamp"].iloc[0]) / 1_000_000_000.0

print("time_s 最小 =", merged["time_s"].min())
print("time_s 最大 =", merged["time_s"].max())
print(merged["time_s"].head(10))

# =========================
# プロット範囲で絞り込み
# =========================
mask = np.ones(len(merged), dtype=bool)

if plot_min_time is not None:
    mask &= (merged["time_s"].to_numpy() >= plot_min_time)

if plot_max_time is not None:
    mask &= (merged["time_s"].to_numpy() <= plot_max_time)

plot_df = merged.loc[mask].copy()

if len(plot_df) == 0:
    raise ValueError("指定したプロット範囲にデータがありませんでした．")

# =========================
# プロット1．散布図
# 横軸 Wx，縦軸 Wy
# =========================
plt.figure(figsize=(7, 7))
plt.scatter(plot_df["Wx_deg"], plot_df["Wy_deg"], s=10)
plt.xlabel("Wx [°]")
plt.ylabel("Wy [°]")
plt.title("角速度散布図 (世界座標系)")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()