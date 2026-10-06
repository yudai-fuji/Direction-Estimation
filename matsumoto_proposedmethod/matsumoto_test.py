import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

file_path = r"260630香川共同研究/KL1.csv"

# センサ値をこの間隔で並べる
SAMPLE_INTERVAL = 0.02

# PCA適用サンプル数
WINDOW_SIZE = 40

# 初期方向
TRUE_INITIAL_HEADING = 90.0

# ファイルの読み込みと必要なセンサ値を抽出
def load_sensors(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values("Timestamp")

    lacc = df[df["Sensor"] == "Lacc"].reset_index(drop=True)
    gamero = df[df["Sensor"] == "GameRo"].reset_index(drop=True)

    return lacc,gamero

# 加速度サンプルとGameを0.02sでグリッドした時刻に割り当てて同期する
def synchronize_sensors(lacc,gamero):
    acc = lacc[["Timestamp", "X", "Y", "Z"]].copy()
    rot = gamero[["Timestamp", "X", "Y", "Z", "W"]].copy()

    acc.columns = ["time", "ax", "ay", "az"]
    rot.columns = ["time", "qx", "qy", "qz", "qw"]

    t0 = min(acc["time"].iloc[0], rot["time"].iloc[0])

    for data in (acc,rot):
        data["time"] = (data["time"] - t0) / 1000_000_000

    start = max(acc["time"].iloc[0], rot["time"].iloc[0])
    end = min(acc["time"].iloc[-1], rot["time"].iloc[-1])

    # floorで歩行時間を整数にし，0.02秒刻みにして，刻まれた個数を数える
    count = int(np.floor((end - start) / SAMPLE_INTERVAL)) + 1

    # 元サンプルを識別するための番号
    acc["acc_id"] = np.arange(len(acc))
    rot["rot_id"] = np.arange(len(rot))

    acc["acc_time"] = acc["time"]
    rot["rot_time"] = rot["time"]

    times = start + np.arange(count) * SAMPLE_INTERVAL
    grid = pd.DataFrame({"time":times})

    synced = pd.merge_asof(grid, acc, on="time", direction="backward", tolerance=SAMPLE_INTERVAL)
    synced = pd.merge_asof(synced, rot, on="time", direction="backward", tolerance=SAMPLE_INTERVAL)

    # ============サンプル数を数える==============

    # isna()は欠損地(Nan)を判定し(True/False)を返す
    missing_acc = synced["acc_id"].isna().sum()
    missing_rot = synced["rot_id"].isna().sum()

    # dropna()はNanの行を削除，nuniqueで加速度サンプルにつけたidを重複なしで数える(0.04gridに同じセンサ値が当てはまる可能性がある)
    used_acc = synced["acc_id"].dropna().nunique()
    used_rot = synced["rot_id"].dropna().nunique()

    unused_acc = len(acc) - used_acc
    unused_rot = len(rot) - used_rot

    print(f"Laccで欠損したサンプル数は{missing_acc}個です")
    print(f"Gameroで欠損したサンプル数は{missing_rot}個です")
    
    print(f"Laccで同期時に未使用のサンプル数は{unused_acc}個です")
    print(f"Gameroで同期時に未使用のサンプル数は{unused_rot}個です")

    # ===========================================

    return synced  

# クォータニオンによる回転
def rotate_acceleration(synced, gamero):
    q = synced[["qx", "qy", "qz", "qw"]].to_numpy()
    q0 = gamero[["X", "Y", "Z", "W"]].iloc[0].to_numpy()

    # 回転行列
    rotation = R.from_quat(q)

    # 時刻0におけるGameRoの回転行列
    initial_rotation = R.from_quat(q0)

    # 初期エラーを考慮した相対回転
    relative_rotation = initial_rotation.inv() * rotation

    acc_rotated = synced[["ax", "ay", "az"]].to_numpy()

    return relative_rotation.apply(acc_rotated)

# PCAを計算
def calculate_pca_axis(window):
    if len(window) < 2:
        return None

    xy = window[["Wx", "Wy"]].to_numpy()

    pca = PCA(n_components=2)
    pca.fit(xy)

    return pca.components_[0]

# 従来の方向不確定性の処理
def choose_traditional_direction(v1, previous_heading):
    angle = np.degrees(np.arctan2(v1[1], v1[0]))

    candidates = np.array([angle, angle + 180.0])
    candidates = (candidates + 180.0) % 360.0 - 180.0

    differences = (candidates - previous_heading + 180.0) % 360.0 - 180.0

    index = np.argmin(np.abs(differences))
    return candidates[index]

# 松本さん提案手法
def choose_matsumoto_direction(window, v1):
    if not np.isfinite(v1).all():
        return np.nan
    
    acc = window[["Wx", "Wy", "Wz"]].to_numpy()

    # 第一主成分と加速度サンプルの内積を格納
    s = acc[:,:2] @ v1

    # 加速度ノルムを計算
    norms = np.linalg.norm(acc, axis=1)

    plus = norms[s > 0]
    minus = norms[s < 0]

    if len(plus) == 0 or len(minus) == 0:
        return np.nan

    d_plus = plus.mean()
    d_minus = minus.mean()

    if d_plus == d_minus:
        return np.nan

    direction = v1 if d_plus > d_minus else -v1

    angle = np.degrees(np.arctan2(direction[1], direction[0]))

    return (angle + 180.0) % 360.0 - 180.0
   

# csvファイルからLaccとGameRoのサンプルを抽出
lacc, gamero = load_sensors(file_path)

# LaccとGameroを0.02sグリッドで同期
synced = synchronize_sensors(lacc,gamero)

# 欠損前のgrid時刻列を保存
grid_times = synced["time"].to_numpy(copy=True)

# 欠損行を削除
synced = synced.dropna(subset=["ax","ay","az","qx","qy","qz","qw"]).copy()

# LaccをW座標系基準のセンサデータへ
synced[["Wx", "Wy", "Wz"]] = rotate_acceleration(synced,gamero)

results = []

for k in range(WINDOW_SIZE - 1, len(grid_times)):
    start_time = grid_times[k - WINDOW_SIZE + 1]
    end_time = grid_times[k]

    window = synced.loc[synced["time"].between(start_time, end_time)]

    v1 = calculate_pca_axis(window)

    if v1 is None:
        v1 = [np.nan, np.nan]

    results.append([v1[0], v1[1], len(window)])

# PCAの結果をデータフレームへ
pca_results = pd.DataFrame(results, columns=["vx", "vy", "sample_count"])

# 初期方向
previous_heading = TRUE_INITIAL_HEADING

traditional_headings = []

for v1 in pca_results[["vx","vy"]].to_numpy():
    if not np.isfinite(v1).all():
        traditional_headings.append(np.nan)
        continue

    heading = choose_traditional_direction(v1, previous_heading)

    traditional_headings.append(heading)
    previous_heading = heading

traditional_headings = np.array(traditional_headings)

matsumoto_headings = []
pca_axes = pca_results[["vx","vy"]].to_numpy()

for i,k in enumerate(range(WINDOW_SIZE - 1, len(grid_times))):
    start_time = grid_times[k - WINDOW_SIZE + 1]
    end_time = grid_times[k]

    window = synced.loc[synced["time"].between(start_time, end_time)]

    v1 = pca_axes[i]
    heading = choose_matsumoto_direction(window, v1)

    matsumoto_headings.append(heading)

matsumoto_headings = np.array(matsumoto_headings)



