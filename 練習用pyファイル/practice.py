import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
import matplotlib.animation as animation
import japanize_matplotlib

# CSVファイルからデータを読み込む(f1= right, f2= left)
name = "Arai"
f1 = 'exam/' + name +'/r_1107_1228_0.csv'
f2 = 'exam/' + name +'/l_1107_1228_0.csv'
data1 = pd.read_csv(f1)
data2 = pd.read_csv(f2)
data1 = data1[244:1244] #データのトリミング(システム時刻 rが880㎳遅い → 44のずれ を考慮)
data2 = data2[200:1200]
timestamps = data1['timestamp'].values

acceleration_vectors1 = data1[['accX', 'accY', 'accZ']].values
acceleration_vectors2 = data2[['accX', 'accY', 'accZ']].values
game_rotation_vectors1 = data1[['gamerotX', 'gamerotY', 'gamerotZ', 'gamerotω']].values
game_rotation_vectors2 = data2[['gamerotX', 'gamerotY', 'gamerotZ', 'gamerotω']].values

# タイムスタンプを0から始まる0.02刻みの新しい値に置き換え
new_timestamps = np.arange(0, len(data1['timestamp']) * 0.02, 0.02)
# 配列の長さを調整（生成した配列がtimestampを超える場合）
if len(new_timestamps) > len(data1['timestamp']):
    new_timestamps = new_timestamps[:len(data1['timestamp'])]
timestamps = new_timestamps


# 回転ベクトルを回転行列に変換し、デバイス座標系の加速度を初期姿勢のX座標系に変換
def rotation_vector_to_matrix(rotation_vectors, acceleration_vectors):
    world_accelerations = []
    for rotation_vector, acceleration in zip(rotation_vectors, acceleration_vectors):
        rot = R.from_quat(rotation_vector)
        rotation_matrix = rot.as_matrix()
        
        # 回転行列を使用してデバイス座標系の加速度を世界座標系に変換
        world_acceleration = rotation_matrix @ acceleration
        world_accelerations.append(world_acceleration)   
  
    return np.array(world_accelerations)

#ベクトルの座標から角度を求める関数
def vector_angle(x, y):
    # atan2(y, x) 関数でベクトルの方向角(x軸正方向)をラジアンで取得
    #atan2の返り値は-πからπ （-180度から180度）
    angle_radians_xy = math.atan2(y, x)
    
    # ラジアンを度に変換
    angle_degrees = math.degrees(angle_radians_xy)
    
    return angle_degrees

#~加速度PCA~
# 3軸(x,y,z)のデータを抽出
# データを標準化する関数
def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# n個ずつ抽出して主成分分析を行い、現在のデバイスに対する進行方向を獲得する関数
def pca_and_plot(data, n, pca_data):
    num_samples = data.shape[0]

    for i in range(0, num_samples, 1):
        # 抽出したデータを分割
        subset = data[i:i + n]
        if subset.shape[0] < n:
            break  # 残りがn個に満たない場合は終了

        # PCAのインスタンスを作成し、主成分を計算
        pca = PCA(n_components=2)

        # 平均を引いて標準化
        mean = np.mean(subset, axis=0)
        data_centered = subset - mean
        pca.fit_transform(data_centered)

        # 主成分ベクトルを取得
        principal_components = pca.components_
        x = principal_components[0, 0]
        y = principal_components[0, 1]

        #x軸正方向との角度を求める
        x_angle = vector_angle(x,y)

        pca_data.append(x_angle)   # x軸正方向に対する予測進行方向の角度
 
# 加速度データをX座標系に変換
world_acc_data1 = rotation_vector_to_matrix(game_rotation_vectors1, acceleration_vectors1)
world_acc_data2 = rotation_vector_to_matrix(game_rotation_vectors2, acceleration_vectors2)
acc_data1 = world_acc_data1[:,:2]
acc_data2 = world_acc_data2[:,:2]

# ~進行方向推定~
#timestampsを進行方向のデータ数に合わせる
sample = 40
times =[]

for i in range(sample - 1, acc_data1.shape[0], 1):
    times.append(timestamps[i])

pca_data1 =[]
pca_data2 =[]

pca_and_plot(acc_data1, sample, pca_data1)
pca_and_plot(acc_data2, sample, pca_data2)

# 各要素の平均を取り、pca_data3に保存
pca_data3 = [(x + y) / 2 for x, y in zip(pca_data1, pca_data2)]

# # pca_data3 から各データに対して絶対値を計算
# data_abs_diff = [abs(value - 90.0) for value in pca_data3]

# ～各センサでの予測進行方向のグラフをプロット
plt.figure(figsize=(10, 6))

plt.plot(times, pca_data1, label='右', color='r')
plt.plot(times, pca_data2, label='左', color='b')
plt.plot(times, pca_data3, label='平均', color='g')
plt.axhline(y=90, color='gold', linestyle="--", linewidth=5)

# グラフのタイトルとラベルを設定
plt.title('Time change Predicted direction')
plt.xlabel('時間(秒)',fontsize="20")
plt.ylabel('推定進行法の角度（度）',fontsize="20")

# 凡例を表示
plt.legend()

# グラフを表示
plt.grid(True)
plt.xticks(fontsize=30)  # x軸の目盛り文字サイズ
plt.yticks(fontsize=30)  # y軸の目盛り文字サイズ
plt.show()
# ～各センサでの予測進行方向のグラフをプロット
# ~進行方向推定~

