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


surgessparse = []
surgesclean = []
surgessparsetest = []
surgescleantest = []

for i in range(trainno):
    data = pd.read_csv('compressed_data/Compressedata%i.csv'%i)
    surge = data.loc[2,:][1:].to_numpy()
    surgesclean.append(surge)
    surgesparse = surge.copy()
    surgesparse[indices] = 0
    surgessparse.append(surgesparse)

for i in range(testno):
    data = pd.read_csv('compressed_data/Compressedata%i.csv'%(i+trainno))
    surge = data.loc[2,:][1:].to_numpy()
    surgescleantest.append(surge)
    surgetestsparse = surge.copy()
    surgetestsparse[indices] = 0
    surgessparsetest.append(surgetestsparse)

time = data.loc[0,:][1:].to_numpy()
timetest = time

for i in range(trainno):
    plt.plot(time,surgessparse[i],'.')
plt.show()
for i in range(trainno):
    plt.plot(time,surgesclean[i],'.')

#%%

newt = []
for t in time:
    newt.append([t])
newt = np.array(newt).astype(np.float32)

#%%

surgeclean = np.array(surgesclean).astype(np.float32)
surgedirty = np.array(surgessparse).astype(np.float32)
surgecleantest = np.array(surgescleantest).astype(np.float32)
surgedirtytest = np.array(surgessparsetest).astype(np.float32)
#%%
x_train = (surgedirty,newt)
y_train = surgeclean
x_test = (surgedirtytest,newt)
y_test = surgecleantest
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
datap = pd.read_csv('compressed_data/Compressedata%i.csv'%701)
surgep = np.array([datap.loc[2,:][1:].to_numpy()])
surgecompare = surgep.copy()
surgep[0][indices] = 0
timep = datap.loc[0,:][1:].to_numpy()
predtime = []
for t in timep:
    predtime.append([t])
predtime = np.array(predtime).astype(np.float32)

pred2 = model.predict((surgep,predtime))
#%%
plt.plot(timep,surgep[0],'.')
plt.show()
plt.plot(timep,pred2[0],'.')
plt.show()
plt.plot(timep,surgecompare[0],'.')
print('The MSE error between the corrected data and the original is %f'%(np.square(surgecompare[0] - pred2[0])).mean(axis=0))
# %%
