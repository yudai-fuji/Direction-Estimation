from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation, FFMpegWriter, writers

# 日本語を表示するための設定
plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

SAMPLE_INTERVAL = 0.02

# PCAに用いる時間窓のグリッド点数

WINDOW_SIZE = 40

# W座標系での真の初期進行方向

TRUE_INITIAL_HEADING = 90.0


# =========================================================
# 時刻整合
# =========================================================

file_L = r'260630香川共同研究/ML2.csv'
file_R = r'260630香川共同研究/MR2.csv'
delay_time = 3.513334
video_to_sensor_offset = 12.458
limit_min_time = 2.88 + video_to_sensor_offset
limit_max_time = 48.30 + video_to_sensor_offset
#真値
true_headings = [90.0, -180.0, 90.0, 0.0, -90.0, -180.0, -90.0, 0.0]
turn_start_times_video = [10.79, 15.20, 20.59, 24.87, 29.66, 34.01, 43.20]
turn_end_times_video = [11.96, 16.31, 21.72, 25.98, 30.82, 35.16, 44.32]
turn_start_times = [round(t + video_to_sensor_offset, 3) for t in turn_start_times_video]
turn_end_times = [round(t + video_to_sensor_offset, 3) for t in turn_end_times_video]


# =========================================================
# ffmpeg
# =========================================================

# ffmpegがPATHにない場合、ffmpeg.exeの絶対パスを指定する
# 例：
# FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"


FFMPEG_PATH = ""


# =========================================================
# CSV読み込みとセンサ抽出
# =========================================================


def load_sensors(file_path):

    df = pd.read_csv(file_path).sort_values("Timestamp")

    lacc = df[df["Sensor"] == "Lacc"].reset_index(drop=True)

    gamero = df[df["Sensor"] == "GameRo"].reset_index(drop=True)

    if lacc.empty or gamero.empty:
        raise ValueError("LaccまたはGameRoがありません。")
    # newnewFujiと時刻原点を統一する。
    # Gyroは時刻原点の確認だけに使い、
    # 測定値自体は今回の進行方向推定には使用しない。

    t0 = df.loc[df["Sensor"].isin(["Lacc", "Gyro", "GameRo"]), "Timestamp"].min()

    return lacc, gamero, t0


# =========================================================
# 0.02秒グリッドへ同期
# =========================================================


def synchronize_sensors(lacc, gamero, t0, time_offset):

    acc = lacc[["Timestamp", "X", "Y", "Z"]].copy()

    rot = gamero[["Timestamp", "X", "Y", "Z", "W"]].copy()

    acc.columns = ["time", "ax", "ay", "az"]

    rot.columns = ["time", "qx", "qy", "qz", "qw"]

    # 左腕はdelay_timeを加算。
    # 右腕は0を加算。

    for data in (acc, rot):

        data["time"] = (data["time"] - t0) / 1_000_000_000 + time_offset
    # 同期状況を確認するためのID

    acc["acc_id"] = np.arange(len(acc))

    rot["rot_id"] = np.arange(len(rot))

    # LaccとGameRoの共通計測範囲

    start = max(acc["time"].iloc[0], rot["time"].iloc[0])

    end = min(acc["time"].iloc[-1], rot["time"].iloc[-1])

    if start > end:
        raise ValueError("LaccとGameRoの計測期間が" "重なっていません。")
    count = int(np.floor((end - start) / SAMPLE_INTERVAL)) + 1

    grid = pd.DataFrame({"time": start + np.arange(count) * SAMPLE_INTERVAL})

    # 現在のグリッド時刻以前で、
    # 最も近いLaccサンプルを割り当てる。
    # 0.02秒以内にサンプルがなければNaNになる。

    synced = pd.merge_asof(
        grid, acc, on="time", direction="backward", tolerance=SAMPLE_INTERVAL
    )

    # GameRoも同様に割り当てる。

    synced = pd.merge_asof(
        synced, rot, on="time", direction="backward", tolerance=SAMPLE_INTERVAL
    )

    print("Laccの未割当グリッド数:", synced["acc_id"].isna().sum())

    print("GameRoの未割当グリッド数:", synced["rot_id"].isna().sum())

    print("Laccの未使用サンプル数:", len(acc) - synced["acc_id"].nunique())

    print("GameRoの未使用サンプル数:", len(rot) - synced["rot_id"].nunique())

    return synced


# =========================================================
# 初期姿勢を基準とした座標へ加速度を回転
# =========================================================


def rotate_acceleration(synced, gamero):

    q = synced[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)

    q0 = gamero[["X", "Y", "Z", "W"]].iloc[0].to_numpy(dtype=float)

    if not np.isfinite(q0).all() or np.linalg.norm(q0) == 0:

        raise ValueError("最初のGameRoが無効なため" "初期姿勢を決められません。")
    rotation = R.from_quat(q)

    initial_rotation = R.from_quat(q0)

    # q0の逆回転 × 現在の回転

    relative_rotation = initial_rotation.inv() * rotation

    acc = synced[["ax", "ay", "az"]].to_numpy()

    return relative_rotation.apply(acc)


# =========================================================
# 水平成分のPCA
# =========================================================


def calculate_pca_axis(window):

    if len(window) < 2:
        return None
    xy = window[["Wx", "Wy"]].to_numpy()

    # 全点が同じ場合、
    # ばらつきの方向を定義できない。

    if np.all(xy == xy[0]):
        return None
    pca = PCA(n_components=2)

    pca.fit(xy)

    return pca.components_[0]


# =========================================================
# 従来手法
# 前時刻の推定方向に近い候補を選択
# =========================================================


def choose_traditional_direction(v1, previous_heading):

    angle = np.degrees(np.arctan2(v1[1], v1[0]))

    candidates = np.array([angle, angle + 180.0])

    candidates = (candidates + 180.0) % 360.0 - 180.0

    differences = (candidates - previous_heading + 180.0) % 360.0 - 180.0

    return candidates[np.argmin(np.abs(differences))]


# =========================================================
# 松本手法
# 各領域の3次元加速度ノルム平均を比較
# =========================================================


def choose_matsumoto_direction(window, v1):

    if not np.isfinite(v1).all():
        return np.nan
    acc = window[["Wx", "Wy", "Wz"]].to_numpy()

    # 水平成分と第一主成分の内積

    s = acc[:, :2] @ v1

    # 射影前の3次元加速度ノルム

    norms = np.linalg.norm(acc, axis=1)

    # 第一主成分軸の正側

    plus = norms[s > 0]

    # 第一主成分軸の負側

    minus = norms[s < 0]

    # s == 0 の点は
    # どちらの領域にも含めない。

    if len(plus) == 0 or len(minus) == 0:
        return np.nan
    d_plus = plus.mean()

    d_minus = minus.mean()

    if d_plus == d_minus:
        return np.nan
    # 平均ノルムの大きい側を
    # 進行方向として採用する。

    direction = v1 if d_plus > d_minus else -v1

    angle = np.degrees(np.arctan2(direction[1], direction[0]))

    return (angle + 180.0) % 360.0 - 180.0


# =========================================================
# 7回の方向転換を含む真値
# =========================================================


def make_true_heading(time):

    # ±180度境界をまたいで
    # 正しく補間できるようunwrapする。

    angles = np.degrees(np.unwrap(np.radians(true_headings)))

    times = np.column_stack([turn_start_times, turn_end_times]).ravel()

    values = np.column_stack([angles[:-1], angles[1:]]).ravel()

    heading = np.interp(time, times, values)

    return (heading + 180.0) % 360.0 - 180.0


# =========================================================
# 全時間窓のPCAと両手法の方向選択
# =========================================================


def calculate_results(synced, grid_times, frame_indices):

    windows = []

    results = []

    traditional_headings = []

    matsumoto_headings = []

    # 真値を方向選択に使うのは最初だけ。

    previous_heading = TRUE_INITIAL_HEADING

    for k in frame_indices:

        start_time = grid_times[k - WINDOW_SIZE + 1]

        end_time = grid_times[k]

        window = synced.loc[synced["time"].between(start_time, end_time)]

        # 描画でも同じサンプルを
        # 使用するため保存する。

        windows.append(window)

        v1 = calculate_pca_axis(window)

        if v1 is None:

            v1 = np.array([np.nan, np.nan])

            traditional_heading = np.nan
        else:

            traditional_heading = choose_traditional_direction(v1, previous_heading)

            # 推定できた場合だけ、
            # 次時刻の基準方向を更新する。

            previous_heading = traditional_heading
        results.append([v1[0], v1[1]])

        traditional_headings.append(traditional_heading)

        matsumoto_headings.append(choose_matsumoto_direction(window, v1))
    pca_results = pd.DataFrame(results, columns=["vx", "vy"])

    return (
        windows,
        pca_results,
        np.array(traditional_headings),
        np.array(matsumoto_headings),
    )


# =========================================================
# 左：従来手法
# 右：松本手法
# =========================================================


def make_animation(
    windows, pca_results, traditional_headings, matsumoto_headings, frame_times
):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 左右のグラフ間隔

    fig.subplots_adjust(wspace=0.25)

    pca_axes = pca_results[["vx", "vy"]].to_numpy()

    true_angles = make_true_heading(frame_times)

    artists = []

    # グラフを一度作り、
    # 各フレームでは内容だけ更新する。

    for ax in axes:

        ax.set(xlim=(-10, 10), ylim=(-10, 10))

        ax.set_aspect("equal")

        ax.grid(True)

        points = ax.scatter([], [], s=10, alpha=0.4,label="W座標系加速度")

        (line,) = ax.plot([], [], "k--", alpha=0.4, label="第一主成分軸")

        true_arrow = ax.quiver(
            0,
            0,
            0,
            0,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="green",
            label="真の進行方向",
        )

        estimated_arrow = ax.quiver(
            0,
            0,
            0,
            0,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="red",
            label="推定進行方向",
        )

        ax.legend(loc="upper right")

        artists.append((points, line, true_arrow, estimated_arrow))

    def update(i):

        window = windows[i]

        xy = window[["Wx", "Wy"]].to_numpy()

        # ±10の範囲外を除外するのは
        # 描画だけ。

        visible = xy[(np.abs(xy) <= 10).all(axis=1)]

        v1 = pca_axes[i]

        estimates = [traditional_headings[i], matsumoto_headings[i]]

        for items, heading in zip(artists, estimates):

            (points, line, true_arrow, estimated_arrow) = items

            points.set_offsets(visible)

            # PCA軸は両方向へ伸びる
            # 破線として表示する。

            if np.isfinite(v1).all():

                line.set_data([-15 * v1[0], 15 * v1[0]], [-15 * v1[1], 15 * v1[1]])
            else:

                line.set_data([], [])
            # 矢印の長さは描画用。
            # 真値を少し長くして
            # 推定方向との重なりを確認しやすくする。

            for arrow, angle, length in [
                (true_arrow, true_angles[i], 7.5),
                (estimated_arrow, heading, 6.0),
            ]:

                arrow.set_visible(bool(np.isfinite(angle)))

                if np.isfinite(angle):

                    rad = np.radians(angle)

                    arrow.set_UVC(length * np.cos(rad), length * np.sin(rad))

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_times),
        interval=(SAMPLE_INTERVAL * 1000),
        repeat=False,
        cache_frame_data=False,
    )

    return fig, animation


# =========================================================
# Main
# =========================================================


def main():

    print("左腕CSV:", file_L)

    print("右腕CSV:", file_R)

    while True:

        side = (
            input(
                "出力したいファイルは"
                "左の結果ですか？右の結果ですか？"
                "(left or right): "
            )
            .strip()
            .lower()
        )

        if side in ("left", "right"):
            break
        print("left または right を" "入力してください。")
    # 選択に応じて、
    # CSVと時刻補正を切り替える。

    if side == "left":

        file_path = DATA_DIR / file_L

        time_offset = delay_time
    else:

        file_path = DATA_DIR / file_R

        time_offset = 0.0
    print("使用CSV:", file_path)

    print("センサ時刻への加算:", time_offset, "秒")

    # ffmpeg.exeを
    # 個別指定している場合

    if FFMPEG_PATH:

        plt.rcParams["animation.ffmpeg_path"] = FFMPEG_PATH
    if not writers.is_available("ffmpeg"):

        raise RuntimeError(
            "ffmpegが見つかりません。"
            "インストールしてPATHを通すか、"
            "FFMPEG_PATHにffmpeg.exeの"
            "絶対パスを指定してください。"
        )
    lacc, gamero, t0 = load_sensors(file_path)

    synced = synchronize_sensors(lacc, gamero, t0, time_offset)

    # 欠損行を削除する前に、
    # 元の0.02秒グリッド時刻を保存する。

    grid_times = synced["time"].to_numpy(copy=True)

    # 無効な行だけを
    # 計算対象から外す。
    # グリッド時刻自体は削除しない。

    columns = ["ax", "ay", "az", "qx", "qy", "qz", "qw"]

    valid = np.isfinite(synced[columns].to_numpy()).all(axis=1)

    quaternions = synced[["qx", "qy", "qz", "qw"]].to_numpy()

    valid &= np.linalg.norm(quaternions, axis=1) > 0

    synced = synced.loc[valid].copy()

    if synced.empty:

        raise ValueError("回転に使用できる" "サンプルがありません。")
    synced[["Wx", "Wy", "Wz"]] = rotate_acceleration(synced, gamero)

    # 窓の終端が歩行区間内にある
    # フレームだけを採用する。
    #
    # 最初の窓には、
    # 歩行開始前のサンプルも含まれる。

    frame_indices = np.flatnonzero(
        (grid_times >= limit_min_time) & (grid_times <= limit_max_time)
    )

    # WINDOW_SIZE個分の
    # 過去サンプルが存在する
    # フレームだけ使用する。

    frame_indices = frame_indices[frame_indices >= WINDOW_SIZE - 1]

    if len(frame_indices) == 0:

        raise ValueError("指定した歩行区間に" "PCAの対象フレームがありません。")
    frame_times = grid_times[frame_indices]

    (windows, pca_results, traditional_headings, matsumoto_headings) = (
        calculate_results(synced, grid_times, frame_indices)
    )

    print("フレーム数:", len(frame_times))

    print("従来手法の推定なし:", np.isnan(traditional_headings).sum())

    print("松本手法の推定なし:", np.isnan(matsumoto_headings).sum())

    fig, animation = make_animation(
        windows, pca_results, traditional_headings, matsumoto_headings, frame_times
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 再実行時に
    # 前の動画を上書きしないよう
    # 日時をファイル名に付ける。

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = OUTPUT_DIR / (f"{file_path.stem}_" f"{side}_" f"{stamp}.mp4")

    writer = FFMpegWriter(
        fps=(1 / SAMPLE_INTERVAL),
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast"],
    )

    def show_progress(current, total):

        if current % 500 == 0 or current == total - 1:

            print(f"MP4保存中: " f"{current + 1}/" f"{total}", flush=True)

    try:

        animation.save(
            str(output_path), writer=writer, dpi=100, progress_callback=(show_progress)
        )
    finally:

        plt.close(fig)
    print("保存完了:", output_path)


if __name__ == "__main__":
    main()
