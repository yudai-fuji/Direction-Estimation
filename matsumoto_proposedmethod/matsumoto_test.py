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

    # floorで歩行時間を整数にし，0.02秒刻みにして，刻まれた個数を数える
    count = int(np.floor((end - start) / SAMPLE_INTERVAL)) + 1

    # 元サンプルを識別するための番号
    acc["acc_id"] = np.arange(len(acc))
    rot["rot_id"] = np.arange(len(rot))

    times = start + np.arange(count) * SAMPLE_INTERVAL
    grid = pd.DataFrame({"time":times})

    synced = pd.merge_asof(grid, acc, on="time", direction="backward", tolerance=SAMPLE_INTERVAL)
    synced = pd.merge_asof(synced, rot, on="time", direction="backward", tolerance=SAMPLE_INTERVAL)

    

    # ① グリッド側で値が見つからなかった点
    missing_acc = synced["acc_id"].isna().sum()
    missing_rot = synced["rot_id"].isna().sum()

    # ② 元データ側で一度も採用されなかったサンプル
    used_acc = synced["acc_id"].dropna().nunique()
    used_rot = synced["rot_id"].dropna().nunique()

    unused_acc = len(acc) - used_acc
    unused_rot = len(rot) - used_rot

    print("=== 同期結果 ===")
    print("Lacc グリッド欠損数:", missing_acc)
    print("GameRo グリッド欠損数:", missing_rot)

    print("Lacc 未使用サンプル数:", unused_acc)
    print("GameRo 未使用サンプル数:", unused_rot)

    return synced   


lacc,gamero = load_sensors(file_path)
synchronize_sensors(lacc,gamero)
