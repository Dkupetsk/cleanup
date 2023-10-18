#%%
import deepxde as dde
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import random
import copy


no = 561580795 - 561579726 #use with .out files
trainno = 500
testno = 20

percentremove = 0.8
indices = np.random.choice(np.arange(200), replace=False,
                           size=int(200 * percentremove)) #Change percentremove to work with sparse data


wavessparse = []
wavesclean = []
wavessparsetest = []
wavescleantest = []

for i in range(trainno):
    data = pd.read_csv('compressed_data/Compressedata%i.csv'%i)
    wave = data.loc[1,:][1:].to_numpy()
    wavesclean.append(wave)
    wavesparse = wave.copy()
    wavesparse[indices] = 0
    wavessparse.append(wavesparse)

for i in range(testno):
    data = pd.read_csv('compressed_data/Compressedata%i.csv'%(i+trainno))
    wave = data.loc[1,:][1:].to_numpy()
    wavescleantest.append(wave)
    wavetestsparse = wave.copy()
    wavetestsparse[indices] = 0
    wavessparsetest.append(wavetestsparse)

time = data.loc[0,:][1:].to_numpy()
timetest = time

for i in range(trainno):
    plt.plot(time,wavessparse[i],'.')
plt.show()
for i in range(trainno):
    plt.plot(time,wavesclean[i],'.')

#%%

newt = []
for t in time:
    newt.append([t])
newt = np.array(newt).astype(np.float32)

#%%

waveclean = np.array(wavesclean).astype(np.float32)
wavedirty = np.array(wavessparse).astype(np.float32)
wavecleantest = np.array(wavescleantest).astype(np.float32)
wavedirtytest = np.array(wavessparsetest).astype(np.float32)
#%%
x_train = (wavedirty,newt)
y_train = waveclean
x_test = (wavedirtytest,newt)
y_test = wavecleantest
#%%
data = dde.data.TripleCartesianProd(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test)
m = len(time)
net = dde.nn.DeepONetCartesianProd(
    [m, 32, 32],
    [1, 32, 32],
    'sin',
    'Glorot normal'
)

# %%

model = dde.Model(data,net)
model.compile('adam', lr=1e-4)
losshistory, train_state = model.train(iterations=20000,display_every=1000)


# %%
datap = pd.read_csv('compressed_data/Compressedata%i.csv'%601)
wavep = np.array([datap.loc[1,:][1:].to_numpy()])
wavepcompare = wavep.copy()
wavep[0][indices] = 0
surgep = datap.loc[2,:][1:].to_numpy()
timep = datap.loc[0,:][1:].to_numpy()

predtime = []
for t in timep:
    predtime.append([t])
predtime = np.array(predtime).astype(np.float32)

pred2 = model.predict((wavep,predtime))
#%%
plt.plot(timep,wavep[0],'.')
plt.show()
plt.plot(timep,pred2[0],'.')
plt.show()
plt.plot(timep,wavepcompare[0],'.')
print('The MSE error between the corrected data and the original is %f'%(np.square(wavepcompare[0] - pred2[0])).mean(axis=0))
# %%
