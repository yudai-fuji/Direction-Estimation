# =========================================================
# 左右端末の世界座標系加速度 Wx-Wy を ax1, ax2 に分けてGIF表示
# CCA適用前の確認用
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import japanize_matplotlib
import math


# =========================================================
# 0. ユーザー設定
# =========================================================

# 左手・右手CSV
FILE_L = "CCA_L.csv"
FILE_R = "CCA_R.csv"

# 出力ファイル
OUTPUT_GIF_PATH = "world_acc_left_right_check.gif"
OUTPUT_SYNC_CSV_PATH = "world_acc_left_right_synced.csv"

# 左手側に加える時刻補正 [s]
# 例：左手が右手より0.892716秒早く始まっているなら、左手に +0.892716 を加える
DELAY_TIME_L_TO_R = 0.53620

# 共通時刻グリッドの間隔 [s]
SAMPLE_INTERVAL = 0.02

# 最近傍として採用する最大許容ずれ [s]
NEAREST_TOLERANCE = SAMPLE_INTERVAL

# GIFで表示する直近点数
WINDOW_SIZE = 40

# GIFのFPS
# 実世界時間に近づけるなら None
# 見やすさ優先なら 10 などにする
GIF_FPS = None

# センサ名
ACC_SENSOR_NAME = "Lacc"
GAMERO_SENSOR_NAME = "GameRo"

# GIFで間引くか
# 1なら全フレーム使用
# 2なら2点に1点，5なら5点に1点
FRAME_STEP = 1


# =========================================================
# 1. 四元数関連関数
# =========================================================

def kyoyaku(gx, gy, gz, gw):
    """
    四元数の共役を返す
    q = (x, y, z, w) に対して q* = (-x, -y, -z, w)
    """
    return -gx, -gy, -gz, gw


def calc_relative_quaternion(gx, gy, gz, gw, initial_quat):
    """
    初期姿勢を基準にした相対四元数を計算する
    q_relative = q_current * q_initial_conjugate
    """
    gx0, gy0, gz0, gw0 = initial_quat

    Gx0, Gy0, Gz0, Gw0 = kyoyaku(gx0, gy0, gz0, gw0)

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


def rotate_xyz_by_gamero(x, y, z, gx, gy, gz, gw, initial_quat):
    """
    加速度ベクトル x, y, z を，GameRoの相対四元数を用いて
    初期姿勢基準の世界座標系へ回転する
    """
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


# =========================================================
# 2. CSV読み込み・時刻作成
# =========================================================

def load_sensor_file_with_time(file_path, time_offset_s=0.0):
    """
    CSVを読み込み，ファイル内の相対時刻 time_s を作る
    time_offset_s により左右の時刻差を補正する
    """
    df = pd.read_csv(file_path)

    required_cols = {"Sensor", "Timestamp", "X", "Y", "Z"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{file_path} に必要な列がありません: {missing_cols}")

    sensor_df = df[df["Sensor"].isin([ACC_SENSOR_NAME, GAMERO_SENSOR_NAME])].copy()

    if len(sensor_df) == 0:
        raise ValueError(f"{file_path} に {ACC_SENSOR_NAME} または {GAMERO_SENSOR_NAME} がありません。")

    time0_ns = sensor_df["Timestamp"].min()

    df["time_s"] = (df["Timestamp"] - time0_ns) / 1_000_000_000.0 + time_offset_s

    gamerot_df = df[df["Sensor"] == GAMERO_SENSOR_NAME].sort_values("Timestamp").reset_index(drop=True)

    if len(gamerot_df) == 0:
        raise ValueError(f"{file_path} に {GAMERO_SENSOR_NAME} がありません。")

    if "W" not in gamerot_df.columns:
        raise ValueError(f"{file_path} の {GAMERO_SENSOR_NAME} に W 列がありません。")

    initial_quat = gamerot_df.iloc[0][["X", "Y", "Z", "W"]].astype(float).to_numpy()

    return df, initial_quat


def get_sensor_time_range(df, sensor_name):
    """
    指定センサの time_s の範囲を取得する
    """
    sensor_df = df[df["Sensor"] == sensor_name].copy()

    if len(sensor_df) == 0:
        raise ValueError(f"{sensor_name} が存在しません。")

    return sensor_df["time_s"].min(), sensor_df["time_s"].max()


# =========================================================
# 3. 共通時刻グリッド作成
# =========================================================

def make_common_grid(df_L, df_R):
    """
    左右端末の Lacc と GameRo がすべて存在する重複区間を求め，
    SAMPLE_INTERVAL ごとの共通時刻グリッドを作る
    """
    starts = []
    ends = []

    for df in [df_L, df_R]:
        for sensor_name in [ACC_SENSOR_NAME, GAMERO_SENSOR_NAME]:
            sensor_start, sensor_end = get_sensor_time_range(df, sensor_name)
            starts.append(sensor_start)
            ends.append(sensor_end)

    overlap_start = max(starts)
    overlap_end = min(ends)

    if overlap_start >= overlap_end:
        raise ValueError("左右端末の重複区間がありません。delay_time やファイルを確認してください。")

    n_grid = int(np.floor((overlap_end - overlap_start) / SAMPLE_INTERVAL)) + 1
    grid_time = overlap_start + np.arange(n_grid) * SAMPLE_INTERVAL

    grid_df = pd.DataFrame({
        "time_s": grid_time
    })

    return grid_df, overlap_start, overlap_end


# =========================================================
# 4. センサ値を共通時刻へ最近傍同期
# =========================================================

def nearest_sensor_to_grid(df, sensor_name, grid_df, prefix):
    """
    1つのセンサを共通時刻グリッドに最近傍割当する
    """
    sensor_df = df[df["Sensor"] == sensor_name].copy().sort_values("time_s").reset_index(drop=True)

    if len(sensor_df) == 0:
        raise ValueError(f"{sensor_name} が存在しません。")

    value_cols = [c for c in ["X", "Y", "Z", "W"] if c in sensor_df.columns]

    sensor_df = sensor_df[["time_s"] + value_cols].copy()
    sensor_df[f"{prefix}_{sensor_name}_source_time_s"] = sensor_df["time_s"]

    matched = pd.merge_asof(
        grid_df.sort_values("time_s"),
        sensor_df.sort_values("time_s"),
        on="time_s",
        direction="nearest",
        tolerance=NEAREST_TOLERANCE
    )

    rename_dict = {
        c: f"{prefix}_{sensor_name}_{c}"
        for c in value_cols
    }

    matched = matched.rename(columns=rename_dict)

    matched[f"{prefix}_{sensor_name}_dt_s"] = np.abs(
        matched["time_s"] - matched[f"{prefix}_{sensor_name}_source_time_s"]
    )

    use_cols = (
        ["time_s"]
        + list(rename_dict.values())
        + [
            f"{prefix}_{sensor_name}_source_time_s",
            f"{prefix}_{sensor_name}_dt_s"
        ]
    )

    return matched[use_cols]


def build_synced_side(df, grid_df, prefix):
    """
    片側端末について Lacc と GameRo を共通時刻グリッドへ同期する
    """
    synced = grid_df.copy()

    acc_matched = nearest_sensor_to_grid(df, ACC_SENSOR_NAME, grid_df, prefix)
    gamero_matched = nearest_sensor_to_grid(df, GAMERO_SENSOR_NAME, grid_df, prefix)

    synced = pd.merge(synced, acc_matched, on="time_s", how="left")
    synced = pd.merge(synced, gamero_matched, on="time_s", how="left")

    return synced


# =========================================================
# 5. 世界座標系加速度を作る
# =========================================================

def add_world_acceleration(sync_df, prefix, initial_quat):
    """
    同期済みデータに世界座標系加速度を追加する
    出力列:
        {prefix}_Wx
        {prefix}_Wy
        {prefix}_Wz
    """
    required_cols = [
        f"{prefix}_{ACC_SENSOR_NAME}_X",
        f"{prefix}_{ACC_SENSOR_NAME}_Y",
        f"{prefix}_{ACC_SENSOR_NAME}_Z",
        f"{prefix}_{GAMERO_SENSOR_NAME}_X",
        f"{prefix}_{GAMERO_SENSOR_NAME}_Y",
        f"{prefix}_{GAMERO_SENSOR_NAME}_Z",
        f"{prefix}_{GAMERO_SENSOR_NAME}_W",
    ]

    missing_cols = [c for c in required_cols if c not in sync_df.columns]
    if missing_cols:
        raise ValueError(f"{prefix} 側に必要な列がありません: {missing_cols}")

    df = sync_df.copy()

    valid_df = df.dropna(subset=required_cols).copy()

    ax = valid_df[f"{prefix}_{ACC_SENSOR_NAME}_X"].astype(float).to_numpy()
    ay = valid_df[f"{prefix}_{ACC_SENSOR_NAME}_Y"].astype(float).to_numpy()
    az = valid_df[f"{prefix}_{ACC_SENSOR_NAME}_Z"].astype(float).to_numpy()

    gx = valid_df[f"{prefix}_{GAMERO_SENSOR_NAME}_X"].astype(float).to_numpy()
    gy = valid_df[f"{prefix}_{GAMERO_SENSOR_NAME}_Y"].astype(float).to_numpy()
    gz = valid_df[f"{prefix}_{GAMERO_SENSOR_NAME}_Z"].astype(float).to_numpy()
    gw = valid_df[f"{prefix}_{GAMERO_SENSOR_NAME}_W"].astype(float).to_numpy()

    Wx, Wy, Wz = rotate_xyz_by_gamero(
        ax, ay, az,
        gx, gy, gz, gw,
        initial_quat
    )

    df[f"{prefix}_Wx"] = np.nan
    df[f"{prefix}_Wy"] = np.nan
    df[f"{prefix}_Wz"] = np.nan

    df.loc[valid_df.index, f"{prefix}_Wx"] = Wx
    df.loc[valid_df.index, f"{prefix}_Wy"] = Wy
    df.loc[valid_df.index, f"{prefix}_Wz"] = Wz

    return df


# =========================================================
# 6. 同期済み左右データを作る
# =========================================================

def prepare_left_right_world_acc():
    """
    左右CSVを読み込み，共通時刻同期し，世界座標系加速度を計算する
    """
    print("CSVを読み込みます。")

    df_L_raw, initial_quat_L = load_sensor_file_with_time(
        FILE_L,
        time_offset_s=DELAY_TIME_L_TO_R
    )

    df_R_raw, initial_quat_R = load_sensor_file_with_time(
        FILE_R,
        time_offset_s=0.0
    )

    print("共通時刻グリッドを作成します。")

    grid_df, overlap_start, overlap_end = make_common_grid(df_L_raw, df_R_raw)

    print(f"重複区間: {overlap_start:.3f} s ～ {overlap_end:.3f} s")
    print(f"共通時刻点数: {len(grid_df)}")

    print("左手・右手のセンサ値を共通時刻へ同期します。")

    sync_L = build_synced_side(df_L_raw, grid_df, prefix="L")
    sync_R = build_synced_side(df_R_raw, grid_df, prefix="R")

    print("世界座標系加速度に変換します。")

    sync_L = add_world_acceleration(sync_L, prefix="L", initial_quat=initial_quat_L)
    sync_R = add_world_acceleration(sync_R, prefix="R", initial_quat=initial_quat_R)

    merged = pd.merge(
        sync_L[["time_s", "L_Wx", "L_Wy", "L_Wz"]],
        sync_R[["time_s", "R_Wx", "R_Wy", "R_Wz"]],
        on="time_s",
        how="inner"
    )

    merged = merged.dropna(
        subset=["L_Wx", "L_Wy", "L_Wz", "R_Wx", "R_Wy", "R_Wz"]
    ).reset_index(drop=True)

    if len(merged) == 0:
        raise ValueError("同期後に有効な左右データがありません。")

    merged.to_csv(OUTPUT_SYNC_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"同期済み世界座標系加速度CSVを保存しました: {OUTPUT_SYNC_CSV_PATH}")

    return merged


# =========================================================
# 7. GIF作成
# =========================================================

def create_left_right_world_acc_gif(df):
    """
    左手 ax1，右手 ax2 に分けて世界座標系加速度 Wx-Wy のGIFを作成する
    """
    time_s = df["time_s"].to_numpy()

    L_Wx = df["L_Wx"].to_numpy()
    L_Wy = df["L_Wy"].to_numpy()

    R_Wx = df["R_Wx"].to_numpy()
    R_Wy = df["R_Wy"].to_numpy()

    # -----------------------------------------------------
    # 実世界時間に近いFPSを計算
    # -----------------------------------------------------
    total_duration_sec = time_s[-1] - time_s[0]
    total_frame = len(df)

    if total_duration_sec <= 0:
        real_time_fps = 10
    else:
        real_time_fps = total_frame / total_duration_sec

    if GIF_FPS is None:
        gif_fps = int(round(real_time_fps / FRAME_STEP))
        gif_fps = max(1, gif_fps)
    else:
        gif_fps = GIF_FPS

    print(f"計測時間: {total_duration_sec:.2f} 秒")
    print(f"元データFPS相当: {real_time_fps:.2f}")
    print(f"GIF保存FPS: {gif_fps}")

    # -----------------------------------------------------
    # 表示範囲を左右共通スケールで決める
    # -----------------------------------------------------
    x_all = np.concatenate([L_Wx, R_Wx])
    y_all = np.concatenate([L_Wy, R_Wy])

    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)

    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    if max_range == 0:
        max_range = 1.0

    padding = max_range * 0.15

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    x_lim = (
        x_center - max_range / 2 - padding,
        x_center + max_range / 2 + padding
    )

    y_lim = (
        y_center - max_range / 2 - padding,
        y_center + max_range / 2 + padding
    )

    # -----------------------------------------------------
    # 図の作成
    # -----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 左手 ax1
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.axhline(0, color="black", linewidth=1.2)
    ax1.axvline(0, color="black", linewidth=1.2)
    ax1.set_title("左手")
    ax1.set_xlabel("Wx")
    ax1.set_ylabel("Wy")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")

    # 右手 ax2
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.axhline(0, color="black", linewidth=1.2)
    ax2.axvline(0, color="black", linewidth=1.2)
    ax2.set_title("右手")
    ax2.set_xlabel("Wx")
    ax2.set_ylabel("Wy")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")

    # アニメーション要素
    left_points, = ax1.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=4,
        alpha=0.65,
        label="直近窓"
    )

    right_points, = ax2.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=4,
        alpha=0.65,
        label="直近窓"
    )

    left_current, = ax1.plot(
        [],
        [],
        marker="*",
        color="red",
        markersize=14,
        linestyle="None",
        label="現在点"
    )

    right_current, = ax2.plot(
        [],
        [],
        marker="*",
        color="red",
        markersize=14,
        linestyle="None",
        label="現在点"
    )

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    time_text = fig.text(
        0.5,
        0.95,
        "",
        ha="center",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    frames = list(range(0, len(df), FRAME_STEP))

    # -----------------------------------------------------
    # 更新関数
    # -----------------------------------------------------
    def update(frame):
        start_index = max(0, frame - WINDOW_SIZE + 1)
        end_index = frame + 1

        left_points.set_data(
            L_Wx[start_index:end_index],
            L_Wy[start_index:end_index]
        )

        right_points.set_data(
            R_Wx[start_index:end_index],
            R_Wy[start_index:end_index]
        )

        left_current.set_data(
            [L_Wx[frame]],
            [L_Wy[frame]]
        )

        right_current.set_data(
            [R_Wx[frame]],
            [R_Wy[frame]]
        )

        time_text.set_text(f"time = {time_s[frame]:.2f} s")

        return (
            left_points,
            right_points,
            left_current,
            right_current,
            time_text
        )

    # -----------------------------------------------------
    # GIF生成
    # -----------------------------------------------------
    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        blit=False
    )

    print("GIF保存を開始します。")

    animation.save(
        OUTPUT_GIF_PATH,
        writer="pillow",
        fps=gif_fps
    )

    plt.close(fig)

    print(f"GIF保存が完了しました: {OUTPUT_GIF_PATH}")


# =========================================================
# 8. メイン処理
# =========================================================

if __name__ == "__main__":
    synced_world_acc_df = prepare_left_right_world_acc()
    create_left_right_world_acc_gif(synced_world_acc_df)