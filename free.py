#3,2次元のPCAの過程をGif表示
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

# =========================================================
# ユーザー設定
# =========================================================
file_L = "0428_1L.csv"
file_R = "0428_1R.csv"

# 歩行開始時刻～歩行終了時刻
limit_min_time = 12.91
limit_max_time = 35.49

# PCA窓長
WINDOW_SIZE = 40

# GIFの見やすさ調整
FRAME_STEP = 2          # 何フレームおきにGIF化するか
FPS = 12                # GIFのフレームレート
DOT_SIZE = 16           # 散布図の点サイズ
PC1_ARROW_LENGTH = 2.0  # PC1矢印の固定長
FIGSIZE = (12, 6)

# 3D表示の固定視点
ELEV = 24
AZIM = -58

# 軸に持たせる余白率
AXIS_MARGIN_RATIO = 0.08

# PC1矢印の色
PC1_COLOR = "red"

# 出力GIF名
OUTPUT_GIF_L = "acc_world_3d_pca_left_simple.gif"
OUTPUT_GIF_R = "acc_world_3d_pca_right_simple.gif"


# =========================================================
# 四元数の共役
# =========================================================
def quat_conjugate(x, y, z, w):
    return (-x, -y, -z, w)


# =========================================================
# Device座標系の3軸値を，GameRotationVectorを用いて
# 初期姿勢基準のWorld座標系へ回転
# 戻り値: Xw, Yw, Zw
# =========================================================
def rotate_xyz_by_gamero(vx, vy, vz, gx, gy, gz, gw, initial_quat):
    gx0, gy0, gz0, gw0 = initial_quat
    Gx0, Gy0, Gz0, Gw0 = quat_conjugate(gx0, gy0, gz0, gw0)

    gwc = gw * Gw0 - gx * Gx0 - gy * Gy0 - gz * Gz0
    gxc = gw * Gx0 + gx * Gw0 - gy * Gz0 + gz * Gy0
    gyc = gw * Gy0 + gx * Gz0 + gy * Gw0 - gz * Gx0
    gzc = gw * Gz0 - gx * Gy0 + gy * Gx0 + gz * Gw0

    norm = np.sqrt(gwc**2 + gxc**2 + gyc**2 + gzc**2)
    gwc = gwc / norm
    gxc = gxc / norm
    gyc = gyc / norm
    gzc = gzc / norm

    Xw = (2*gwc*gwc + 2*gxc*gxc - 1)*vx + (2*gxc*gyc - 2*gzc*gwc)*vy + (2*gxc*gzc + 2*gyc*gwc)*vz
    Yw = (2*gxc*gyc + 2*gzc*gwc)*vx + (2*gwc*gwc + 2*gyc*gyc - 1)*vy + (2*gyc*gzc - 2*gxc*gwc)*vz
    Zw = (2*gxc*gzc - 2*gyc*gwc)*vx + (2*gyc*gzc + 2*gxc*gwc)*vy + (2*gwc*gwc + 2*gzc*gzc - 1)*vz

    return Xw, Yw, Zw


# =========================================================
# 1端末分のCSVから，
# 歩行区間のWorld座標系加速度DataFrameを作る
# 戻り値列:
# time_s, Xw, Yw, Zw
# =========================================================
def load_world_acc_df(file_path, start_time, end_time):
    df = pd.read_csv(file_path)

    acc_df = df[df["Sensor"] == "Lacc"].copy().sort_values("Timestamp")
    gamerot_df = df[df["Sensor"] == "GameRo"].copy().sort_values("Timestamp")

    if len(acc_df) == 0:
        raise ValueError(f"{file_path} に Lacc がありません。")
    if len(gamerot_df) == 0:
        raise ValueError(f"{file_path} に GameRo がありません。")

    merged = pd.merge_asof(
        acc_df,
        gamerot_df,
        on="Timestamp",
        direction="backward",
        suffixes=("_acc", "_ro")
    )

    merged = merged.dropna(subset=["X_acc", "Y_acc", "Z_acc", "X_ro", "Y_ro", "Z_ro", "W_ro"]).reset_index(drop=True)

    ts_sec = merged["Timestamp"].to_numpy(dtype=float) / 1_000_000_000.0
    ts_sec = ts_sec - ts_sec[0]
    merged["time_s"] = ts_sec

    # 歩行区間だけ使う
    merged = merged[(merged["time_s"] >= start_time) & (merged["time_s"] <= end_time)].copy().reset_index(drop=True)

    if len(merged) < WINDOW_SIZE:
        raise ValueError(
            f"{file_path} の歩行区間データ数が不足しています。"
            f" rows={len(merged)}, WINDOW_SIZE={WINDOW_SIZE}"
        )

    gx0, gy0, gz0, gw0 = gamerot_df.iloc[0][["X", "Y", "Z", "W"]]
    initial_quat = (gx0, gy0, gz0, gw0)

    Xw, Yw, Zw = rotate_xyz_by_gamero(
        merged["X_acc"].to_numpy(dtype=float),
        merged["Y_acc"].to_numpy(dtype=float),
        merged["Z_acc"].to_numpy(dtype=float),
        merged["X_ro"].to_numpy(dtype=float),
        merged["Y_ro"].to_numpy(dtype=float),
        merged["Z_ro"].to_numpy(dtype=float),
        merged["W_ro"].to_numpy(dtype=float),
        initial_quat
    )

    out = pd.DataFrame({
        "time_s": merged["time_s"].to_numpy(dtype=float),
        "Xw": Xw,
        "Yw": Yw,
        "Zw": Zw
    })

    return out


# =========================================================
# 軸範囲を固定で作る
# 全区間の最小・最大から余白をつけて決める
# 3Dは見やすさのため3軸共通幅でそろえる
# =========================================================
def build_fixed_axis_ranges(df, margin_ratio=0.08):
    mins = df[["Xw", "Yw", "Zw"]].min()
    maxs = df[["Xw", "Yw", "Zw"]].max()

    centers = (mins + maxs) / 2.0
    spans = (maxs - mins).to_numpy(dtype=float)

    max_span = float(np.max(spans))
    if max_span == 0:
        max_span = 1.0

    half = (max_span / 2.0) * (1.0 + margin_ratio)

    ranges = {
        "Xw": (centers["Xw"] - half, centers["Xw"] + half),
        "Yw": (centers["Yw"] - half, centers["Yw"] + half),
        "Zw": (centers["Zw"] - half, centers["Zw"] + half),
    }
    return ranges


# =========================================================
# 直近WINDOW_SIZE点からPCAを行い，
# 散布点群とPC1を描くGIFを生成
# 3D + 2D投影(Wx-Wy)
# ※ Wy-Wz, Wz-Wx は表示しない
# ※ 主成分矢印は赤色・固定長
# ※ 中央の赤点は表示しない
# =========================================================
def create_world_acc_pca_gif(df, output_path, title_prefix, window_size=40, frame_step=1, fps=12):
    df = df.reset_index(drop=True).copy()

    ranges = build_fixed_axis_ranges(df, margin_ratio=AXIS_MARGIN_RATIO)

    all_times = df["time_s"].to_numpy()
    X = df["Xw"].to_numpy()
    Y = df["Yw"].to_numpy()
    Z = df["Zw"].to_numpy()

    frame_indices = list(range(window_size - 1, len(df), frame_step))
    if frame_indices[-1] != len(df) - 1:
        frame_indices.append(len(df) - 1)

    fig = plt.figure(figsize=FIGSIZE)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(1, 2, 2)

    pca = PCA(n_components=3)

    def draw_frame(frame_no):
        idx = frame_indices[frame_no]
        start = idx - (window_size - 1)
        end = idx + 1

        wx = X[start:end]
        wy = Y[start:end]
        wz = Z[start:end]

        points = np.column_stack((wx, wy, wz))
        pca.fit(points)

        pc1 = pca.components_[0]
        pc1_ratio = pca.explained_variance_ratio_[0]

        cx = float(np.mean(wx))
        cy = float(np.mean(wy))
        cz = float(np.mean(wz))

        # PC1を固定長にそろえる
        norm_pc1 = np.linalg.norm(pc1)
        if norm_pc1 == 0:
            vx, vy, vz = 0.0, 0.0, 0.0
        else:
            unit_pc1 = pc1 / norm_pc1
            vx = unit_pc1[0] * PC1_ARROW_LENGTH
            vy = unit_pc1[1] * PC1_ARROW_LENGTH
            vz = unit_pc1[2] * PC1_ARROW_LENGTH

        ax3d.cla()
        ax_xy.cla()

        # -------- 3D --------
        ax3d.scatter(wx, wy, wz, s=DOT_SIZE)
        ax3d.quiver(
            cx, cy, cz, vx, vy, vz,
            color=PC1_COLOR,
            arrow_length_ratio=0.15,
            linewidth=2.0
        )
        ax3d.set_title(f"{title_prefix}\n3D Scatter + PC1")
        ax3d.set_xlabel("Wx")
        ax3d.set_ylabel("Wy")
        ax3d.set_zlabel("Wz")
        ax3d.set_xlim(ranges["Xw"])
        ax3d.set_ylim(ranges["Yw"])
        ax3d.set_zlim(ranges["Zw"])
        ax3d.view_init(elev=ELEV, azim=AZIM)

        # -------- Wx-Wy --------
        ax_xy.scatter(wx, wy, s=DOT_SIZE)
        ax_xy.arrow(
            cx, cy, vx, vy,
            color=PC1_COLOR,
            length_includes_head=True,
            head_width=0.03 * (ranges["Yw"][1] - ranges["Yw"][0]),
            linewidth=2.0
        )
        ax_xy.set_title("Projection: Wx-Wy")
        ax_xy.set_xlabel("Wx")
        ax_xy.set_ylabel("Wy")
        ax_xy.set_xlim(ranges["Xw"])
        ax_xy.set_ylim(ranges["Yw"])
        ax_xy.grid(True)

        fig.suptitle(
            f"{title_prefix}\n"
            f"time={all_times[idx]:.2f} s, window={window_size}, "
            f"PC1 ratio={pc1_ratio:.3f}",
            fontsize=13
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

    anim = FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=1000 / fps)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)


def main():
    # 実行時のカレントディレクトリを基準にCSVを探す．
    # 例: G:\... で python C:\Users\...\free.py を実行した場合，G:\...\0428_1L.csv を読む．
    base_dir = Path.cwd()

    file_path_L = base_dir / file_L
    file_path_R = base_dir / file_R

    if not file_path_L.exists():
        raise FileNotFoundError(f"左手CSVが見つかりません: {file_path_L}")
    if not file_path_R.exists():
        raise FileNotFoundError(f"右手CSVが見つかりません: {file_path_R}")

    print("左手データ読込中...")
    df_L = load_world_acc_df(file_path_L, limit_min_time, limit_max_time)
    print(f"左手データ数: {len(df_L)}")

    print("右手データ読込中...")
    df_R = load_world_acc_df(file_path_R, limit_min_time, limit_max_time)
    print(f"右手データ数: {len(df_R)}")

    out_L = base_dir / OUTPUT_GIF_L
    out_R = base_dir / OUTPUT_GIF_R

    print("左手GIF生成中...")
    create_world_acc_pca_gif(
        df_L,
        output_path=out_L,
        title_prefix="Left Device (World Acceleration)",
        window_size=WINDOW_SIZE,
        frame_step=FRAME_STEP,
        fps=FPS
    )

    print("右手GIF生成中...")
    create_world_acc_pca_gif(
        df_R,
        output_path=out_R,
        title_prefix="Right Device (World Acceleration)",
        window_size=WINDOW_SIZE,
        frame_step=FRAME_STEP,
        fps=FPS
    )

    print("生成完了")
    print(f"左手GIF: {out_L}")
    print(f"右手GIF: {out_R}")


if __name__ == "__main__":
    from pathlib import Path
    main()
