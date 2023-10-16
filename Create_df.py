#%%
import deepxde as dde
import torch

import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
#Getting data
#Columns 10,11,12,13,14,15 are the DOF data (surge, sway, heave, pitch, roll, yaw)
#72 is obviously the wave elevation data
#There are a few sources for tension. For now 73,74,75 refer to the scalar tension on the fair lines.
no = 561580795 - 561580790

time = []
waves = []
surge = []
sway = []
heave = []
pitch = []
roll = []
yaw = []
T1 = []
T2 = []
T3 = []
for i in range(no):
    data = pd.read_csv('winddata/seed%i.out'%(-561580795 + i), delim_whitespace=True,skiprows=8)
    time.append(data.iloc[:,0].astype(float).to_numpy())
    waves.append(data.iloc[:,87].astype(float).to_numpy())
    surge.append(data.iloc[:,12].astype(float).to_numpy())
    sway.append(data.iloc[:,13].astype(float).to_numpy())
    heave.append(data.iloc[:,14].astype(float).to_numpy())
    pitch.append(data.iloc[:,15].astype(float).to_numpy())
    roll.append(data.iloc[:,16].astype(float).to_numpy())
    yaw.append(data.iloc[:,17].astype(float).to_numpy())
    T1.append(data.iloc[:,88].astype(float).to_numpy())
    T2.append(data.iloc[:,89].astype(float).to_numpy())
    T3.append(data.iloc[:,90].astype(float).to_numpy())

print(surge)
#%%
for i in range(no):
    data = [time[i],waves[i],surge[i],sway[i],heave[i],pitch[i],roll[i],yaw[i],T1[i],T2[i],T3[i]]
    df = pd.DataFrame(data,index=['Time', 'Wave elevation', 'Surge', 'Sway', 'Heave', 'Pitch', "Roll", 'Yaw', 'T1', 'T2', 'T3'])
    df.to_csv('winddata/compressed_data_wind/Compressedata%i.csv'%i)

# %%
df
# %%
print(surge)
# %%

data = pd.read_csv('IEA-15-240-RWT-UMaineSemi.out', delim_whitespace=True,skiprows=8)
# %%
