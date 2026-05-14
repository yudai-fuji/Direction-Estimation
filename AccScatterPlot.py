#---回転された加速度の散布図を出力---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = '2kaime_9.csv'

try:
    df = pd.read_csv(file_path,skiprows=[1,2,3])

except FileNotFoundError:
    print('ファイルが見つかりません')
    exit()

acc_df = df[df['Sensor'] == 'Lacc'].copy()
gameRo_df = df[df['Sensor'] == 'GameRo'].copy()

merged_df = pd.merge_asof(
    acc_df.sort_values('Timestamp'),
    gameRo_df.sort_values('Timestamp'),
    on='Timestamp',direction='backward',suffixes=('_acc', '_ro'))

# 結合後の列名を分かりやすく設定
# 加速度
ax = merged_df['X_acc']
ay = merged_df['Y_acc']
az = merged_df['Z_acc']
# クォータニオン
gx = merged_df['X_ro']
gy = merged_df['Y_ro']
gz = merged_df['Z_ro']
gw = merged_df['W_ro']

gx0, gy0, gz0, gw0 = gameRo_df.iloc[0]['X'], gameRo_df.iloc[0]['Y'], gameRo_df.iloc[0]['Z'], gameRo_df.iloc[0]['W']
    
# 共役な四元数を求める
def kyoyaku(gx, gy, gz, gw):
    return (-gx, -gy, -gz, gw)

# 1行目の四元数の共役を取得
Gx0, Gy0, Gz0, Gw0 = kyoyaku(gx0, gy0, gz0, gw0)

# --- 相対クォータニオンの計算（q_relative = q * q_0_conjugate）---
# 相対化のための四元数積
gwc = gw*Gw0 - gx*Gx0 - gy*Gy0 - gz*Gz0
gxc = gw*Gx0 + gx*Gw0 - gy*Gz0 + gz*Gy0
gyc = gw*Gy0 + gx*Gz0 + gy*Gw0 - gz*Gx0
gzc = gw*Gz0 - gx*Gy0 + gy*Gx0 + gz*Gw0

# --- 回転行列の適用（加速度ベクトルの回転 P' = q_relative * P * q_relative_conjugate）---
# 相対化した回転行列の計算
Ax = (2*gwc*gwc + 2*gxc*gxc - 1)*ax + (2*gxc*gyc - 2*gzc*gwc)*ay + (2*gxc*gzc + 2*gyc*gwc)*az
Ay = (2*gxc*gyc + 2*gzc*gwc)*ax + (2*gwc*gwc + 2*gyc*gyc - 1)*ay + (2*gyc*gzc - 2*gxc*gwc)*az
Az = (2*gxc*gzc - 2*gyc*gwc)*ax + (2*gyc*gzc + 2*gxc*gwc)*ay + (2*gwc*gwc + 2*gzc*gzc - 1)*az

plt.figure(figsize=(10,10))
plt.scatter(Ax, Ay, alpha=0.7, s=5)
plt.grid(True)
plt.xlabel('Ax (m/s^2)')
plt.ylabel('Ay (m/s^2)')
plt.show()
