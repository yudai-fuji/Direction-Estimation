# =========================================================
# 左右端末の整合用コード：加速度Ver
# 右腕端末を先にStartし，左腕端末を後からStartした前提
#
# 目的:
#   歩行前に行った「両腕を同時に上げる動作」の
#   Lacc合成加速度ピークを用いて，左右端末の時刻ずれを推定する．
#
# 出力:
#   1．推定 delay_time をコンソールに表示
#   2．同期前のLacc合成加速度グラフを表示
#   3．同期後のLacc合成加速度グラフを表示
#
# 注意:
#   このコードはファイル保存を一切行わない．
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib
except ImportError:
    pass


# =========================================================
# 設定
# =========================================================

FILE_L = "0428_1L.csv"
FILE_R = "0428_1R.csv"

# 同期に使うセンサ
# 加速度Verなので Lacc を使用する
SYNC_SENSOR = "Lacc"

# 平滑化窓
# 加速度はノイズが出やすいため，5〜11程度を推奨
SMOOTH_WINDOW = 5

# 実験条件
# 右腕Start → 左腕Start なら True
# 左腕Start → 右腕Start なら False
RIGHT_STARTS_FIRST = True

# 同期動作の探索範囲
# ここを実験ごとに変更する
#
# 例:
#   右腕端末のLacc波形上で，同期動作が 8.0〜8.8秒 にある
#   左腕端末のLacc波形上で，同期動作が 3.0〜3.8秒 にある
#
# 範囲を広くしすぎると，歩行中の腕振りピークを拾う可能性がある．
SYNC_RIGHT_WINDOW = (0.0, 15.0)
SYNC_LEFT_WINDOW  = (0.0, 15.0)

# グラフ表示範囲
PLOT_MIN_TIME = 0.0
PLOT_MAX_TIME = 20.0


# =========================================================
# 同期用信号の作成
# =========================================================

def make_sync_signal(file_path, sensor_name="Lacc", smooth_window=5):
    """
    CSVから同期用の加速度合成値を作成する．

    Parameters
    ----------
    file_path : str
        CSVファイル名
    sensor_name : str
        使用するセンサ名．加速度Verでは基本的に 'Lacc'
    smooth_window : int
        平滑化窓幅

    Returns
    -------
    pd.DataFrame
        time_s, norm, norm_smooth を持つDataFrame
    """

    df = pd.read_csv(file_path)

    required_cols = ["Timestamp", "Sensor", "X", "Y", "Z"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{file_path} に必要な列 '{col}' がありません．")

    sensor_df = df[df["Sensor"] == sensor_name].copy()
    sensor_df = sensor_df.sort_values("Timestamp").reset_index(drop=True)

    if len(sensor_df) == 0:
        raise ValueError(f"{file_path} に Sensor == '{sensor_name}' のデータがありません．")

    for col in ["Timestamp", "X", "Y", "Z"]:
        sensor_df[col] = pd.to_numeric(sensor_df[col], errors="coerce")

    sensor_df = sensor_df.dropna(subset=["Timestamp", "X", "Y", "Z"]).reset_index(drop=True)

    if len(sensor_df) == 0:
        raise ValueError(f"{file_path} の {sensor_name} データが数値として読み込めません．")

    # 各端末内のセンサ開始時刻を0秒とする
    t0 = sensor_df["Timestamp"].iloc[0]
    sensor_df["time_s"] = (sensor_df["Timestamp"] - t0) / 1_000_000_000.0

    # 3軸合成加速度
    sensor_df["norm"] = np.sqrt(
        sensor_df["X"] ** 2 +
        sensor_df["Y"] ** 2 +
        sensor_df["Z"] ** 2
    )

    # 平滑化
    if smooth_window is not None and smooth_window > 1:
        sensor_df["norm_smooth"] = sensor_df["norm"].rolling(
            window=smooth_window,
            center=True,
            min_periods=1
        ).mean()
    else:
        sensor_df["norm_smooth"] = sensor_df["norm"]

    return sensor_df[["time_s", "norm", "norm_smooth"]]


# =========================================================
# 同期ピークの検出
# =========================================================

def find_sync_peak(sync_df, start_s, end_s, value_col="norm_smooth"):
    """
    指定した時間範囲内で最大ピークを検出する．
    """

    target = sync_df[
        (sync_df["time_s"] >= start_s) &
        (sync_df["time_s"] <= end_s)
    ].copy()

    if len(target) == 0:
        raise ValueError(f"{start_s:.3f}〜{end_s:.3f}秒の範囲にデータがありません．")

    peak_idx = target[value_col].idxmax()

    peak_time = float(sync_df.loc[peak_idx, "time_s"])
    peak_value = float(sync_df.loc[peak_idx, value_col])

    return peak_time, peak_value


# =========================================================
# delay_time の推定
# =========================================================

def estimate_delay_time(
    file_R,
    file_L,
    right_window,
    left_window,
    sensor_name="Lacc",
    smooth_window=5,
    right_starts_first=True
):
    """
    同期動作の加速度ピーク時刻差から delay_time を推定する．

    右腕Start → 左腕Start の場合:
        delay_time = 右腕ピーク時刻 - 左腕ピーク時刻
        左腕 time_s に delay_time を加える

    左腕Start → 右腕Start の場合:
        delay_time = 左腕ピーク時刻 - 右腕ピーク時刻
        右腕 time_s に delay_time を加える
    """

    sync_R = make_sync_signal(
        file_R,
        sensor_name=sensor_name,
        smooth_window=smooth_window
    )

    sync_L = make_sync_signal(
        file_L,
        sensor_name=sensor_name,
        smooth_window=smooth_window
    )

    t_R_peak, v_R_peak = find_sync_peak(
        sync_R,
        start_s=right_window[0],
        end_s=right_window[1]
    )

    t_L_peak, v_L_peak = find_sync_peak(
        sync_L,
        start_s=left_window[0],
        end_s=left_window[1]
    )

    if right_starts_first:
        delay_time = t_R_peak - t_L_peak
        delayed_side = "L"
        explanation = "右腕Start → 左腕Start のため，左腕 time_s に delay_time を加算"
    else:
        delay_time = t_L_peak - t_R_peak
        delayed_side = "R"
        explanation = "左腕Start → 右腕Start のため，右腕 time_s に delay_time を加算"

    result = {
        "delay_time": float(delay_time),
        "t_R_peak": float(t_R_peak),
        "t_L_peak": float(t_L_peak),
        "v_R_peak": float(v_R_peak),
        "v_L_peak": float(v_L_peak),
        "delayed_side": delayed_side,
        "explanation": explanation,
    }

    return result, sync_R, sync_L


# =========================================================
# 同期前プロット
# =========================================================

def plot_before_alignment(
    sync_R,
    sync_L,
    t_R_peak,
    t_L_peak,
    plot_min_time=None,
    plot_max_time=None
):
    """
    補正前の左右Lacc合成加速度をプロットする．
    """

    R_plot = sync_R.copy()
    L_plot = sync_L.copy()

    if plot_min_time is not None:
        R_plot = R_plot[R_plot["time_s"] >= plot_min_time]
        L_plot = L_plot[L_plot["time_s"] >= plot_min_time]

    if plot_max_time is not None:
        R_plot = R_plot[R_plot["time_s"] <= plot_max_time]
        L_plot = L_plot[L_plot["time_s"] <= plot_max_time]

    plt.figure(figsize=(10, 5))

    plt.plot(
        R_plot["time_s"],
        R_plot["norm_smooth"],
        label="右腕 Lacc norm",
        c="r",
        alpha=0.8
    )

    plt.plot(
        L_plot["time_s"],
        L_plot["norm_smooth"],
        label="左腕 Lacc norm",
        c="b",
        alpha=0.8
    )

    plt.axvline(
        t_R_peak,
        color="r",
        linestyle="--",
        alpha=0.8,
        label="右腕ピーク"
    )

    plt.axvline(
        t_L_peak,
        color="b",
        linestyle="--",
        alpha=0.8,
        label="左腕ピーク"
    )

    plt.xlabel("時間 [s]")
    plt.ylabel("合成加速度 [m/s²]")
    plt.title("同期前のLacc合成加速度")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 同期後プロット
# =========================================================

def plot_after_alignment(
    sync_R,
    sync_L,
    delay_time,
    delayed_side,
    t_R_peak,
    t_L_peak,
    plot_min_time=None,
    plot_max_time=None
):
    """
    補正後の左右Lacc合成加速度をプロットする．
    """

    R_plot = sync_R.copy()
    L_plot = sync_L.copy()

    if delayed_side == "L":
        L_plot["time_s_aligned"] = L_plot["time_s"] + delay_time
        R_plot["time_s_aligned"] = R_plot["time_s"]

        t_R_peak_aligned = t_R_peak
        t_L_peak_aligned = t_L_peak + delay_time

    elif delayed_side == "R":
        R_plot["time_s_aligned"] = R_plot["time_s"] + delay_time
        L_plot["time_s_aligned"] = L_plot["time_s"]

        t_R_peak_aligned = t_R_peak + delay_time
        t_L_peak_aligned = t_L_peak

    else:
        raise ValueError("delayed_side は 'L' または 'R' である必要があります．")

    if plot_min_time is not None:
        R_plot = R_plot[R_plot["time_s_aligned"] >= plot_min_time]
        L_plot = L_plot[L_plot["time_s_aligned"] >= plot_min_time]

    if plot_max_time is not None:
        R_plot = R_plot[R_plot["time_s_aligned"] <= plot_max_time]
        L_plot = L_plot[L_plot["time_s_aligned"] <= plot_max_time]

    plt.figure(figsize=(10, 5))

    plt.plot(
        R_plot["time_s_aligned"],
        R_plot["norm_smooth"],
        label="右腕 Lacc norm",
        c="r",
        alpha=0.8
    )

    plt.plot(
        L_plot["time_s_aligned"],
        L_plot["norm_smooth"],
        label="左腕 Lacc norm",
        c="b",
        alpha=0.8
    )

    plt.axvline(
        t_R_peak_aligned,
        color="r",
        linestyle="--",
        alpha=0.8,
        label="右腕ピーク補正後"
    )

    plt.axvline(
        t_L_peak_aligned,
        color="b",
        linestyle="--",
        alpha=0.8,
        label="左腕ピーク補正後"
    )

    plt.xlabel("時間 [s]")
    plt.ylabel("合成加速度 [m/s²]")
    plt.title("同期後のLacc合成加速度")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 実行部
# =========================================================

if __name__ == "__main__":
    print("=== 左右端末の整合処理：加速度Ver ===")
    print(f"右腕CSV: {FILE_R}")
    print(f"左腕CSV: {FILE_L}")
    print(f"同期センサ: {SYNC_SENSOR}")
    print(f"右腕探索範囲: {SYNC_RIGHT_WINDOW[0]:.3f}〜{SYNC_RIGHT_WINDOW[1]:.3f} [s]")
    print(f"左腕探索範囲: {SYNC_LEFT_WINDOW[0]:.3f}〜{SYNC_LEFT_WINDOW[1]:.3f} [s]")

    result, sync_R, sync_L = estimate_delay_time(
        file_R=FILE_R,
        file_L=FILE_L,
        right_window=SYNC_RIGHT_WINDOW,
        left_window=SYNC_LEFT_WINDOW,
        sensor_name=SYNC_SENSOR,
        smooth_window=SMOOTH_WINDOW,
        right_starts_first=RIGHT_STARTS_FIRST
    )

    print("\n=== 整合結果：加速度Ver ===")
    print(f"右腕ピーク時刻: {result['t_R_peak']:.6f} [s]")
    print(f"左腕ピーク時刻: {result['t_L_peak']:.6f} [s]")
    print(f"右腕ピーク値: {result['v_R_peak']:.6f} [m/s²]")
    print(f"左腕ピーク値: {result['v_L_peak']:.6f} [m/s²]")
    print(f"推定 delay_time: {result['delay_time']:.6f} [s]")
    print(f"delay_time ミリ秒換算: {result['delay_time'] * 1000:.3f} [ms]")
    print(f"補正対象: {result['delayed_side']}")
    print(result["explanation"])

    if result["delay_time"] < 0:
        print("\n警告: delay_time が負になっています．")
        print("探索範囲が間違っているか，Start順序の設定が逆の可能性があります．")
        print("RIGHT_STARTS_FIRST または SYNC_RIGHT_WINDOW / SYNC_LEFT_WINDOW を確認してください．")

    plot_before_alignment(
        sync_R=sync_R,
        sync_L=sync_L,
        t_R_peak=result["t_R_peak"],
        t_L_peak=result["t_L_peak"],
        plot_min_time=PLOT_MIN_TIME,
        plot_max_time=PLOT_MAX_TIME
    )

    plot_after_alignment(
        sync_R=sync_R,
        sync_L=sync_L,
        delay_time=result["delay_time"],
        delayed_side=result["delayed_side"],
        t_R_peak=result["t_R_peak"],
        t_L_peak=result["t_L_peak"],
        plot_min_time=PLOT_MIN_TIME,
        plot_max_time=PLOT_MAX_TIME
    )

    print("\n=== 完了 ===")
    print("ファイル保存は行っていません．")
    print("本解析コードには，次の値を delay_time として入れてください．")
    print(f"delay_time = {result['delay_time']:.6f}")