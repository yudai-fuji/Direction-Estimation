import pandas as pd
import numpy as np

file_path = r"260630香川共同研究/KL1.csv"

# センサ値をこの間隔で並べる
SAMPLE_INTERVAL = 0.02

# ファイルの読み込みと必要なセンサ値を抽出
def load_sensors(file_path):
    df = pd.read_csv(file_path)
    df.sort_values("Timestamp")

    lacc = df[df["Sensor"] == "Lacc"].reset_index(drop=True)
    gamero = df[df["Sensor"] == "GameRo"].reset_index(drop=True)

    return lacc,gamero

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

    count = int(np.floor((end - start) / SAMPLE_INTERVAL)) + 1

    times = start + np.arange(count) * SAMPLE_INTERVAL
    grid = pd.DataFrame({"time": times})

    synced = pd.merge_asof(grid, acc, on="time", direction="nearest")
    synced = pd.merge_asof(synced, rot, on="time", direction="nearest")

    return synced


lacc,gamero = load_sensors(file_path)
synchronize_sensors(lacc,gamero)
