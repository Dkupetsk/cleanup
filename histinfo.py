#%%
'''
Please install deepxde from my branch - type pip install -e git+https://github.com/Dkupetsk/deepxde.git@master#egg=deepxde
This is required because of the matrix multiplication after the forward pass, the trunk net output will be 3D instead of 2D

For details, look at DeepXDE -> nn -> PyTorch -> deeponet.py on line 123. The multiplication of matrices subscript wise is bi, ni -> bn (no_funcs, last bnet no_rows), (no_timesteps, last tnet no_rows).
I changed this to bi, bni -> bn for (no_funcs, last bnet no_rows), (no_funcs_in_history_states,no_timesteps,last tnet no_rows)

Finally, this will be much faster on HPC, it is very slow on a personal computer
'''
import numpy as np
import tensorflow as tf
import torch
import deepxde as dde
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import random


no = 561580795 - 561579726 #use with .out files
trainno = 500
testno = 20

percentremove = 0
indices = np.random.choice(np.arange(200), replace=False,
                           size=int(200 * percentremove)) #Change percentremove to work with sparse data


waves = []
surges = []
surgescopy = []
wavestest = []
surgestest = []
for i in range(trainno):
    data = pd.read_csv('compressed_data/Compressedata%i.csv'%i)
    wave = data.loc[1,:][1:].to_numpy()
    surge = data.loc[2,:][1:].to_numpy()
    surgecopy = surge.copy()
    wave[indices] = 0
    surge[indices] = 0
    waves.append(wave)
    surges.append(surge)
    surgescopy.append(surgecopy)
    
for i in range(testno):
    data = pd.read_csv('compressed_data/Compressedata%i.csv'%(i+trainno))
    wavestest.append(data.loc[1,:][1:].to_numpy()) 
    surgestest.append(data.loc[2,:][1:].to_numpy())

time = data.loc[0,:][1:].to_numpy()
timetest = time
#for i in range(trainno):
    #plt.plot(time,waves[i])
#plt.show()
#for i in range(trainno):
    #plt.plot(time,surges[i])

#plt.show()

#lt.plot(time,waves[0],'.')
#plt.show()
#plt.plot(time,surges[0],'.')

#%%
#Make all zeros the average of the element closest in front of and behind it - filtering
#Use this (if you want) with sparse data to "fill in the gaps"
import copy

def takeaverage(funcs,timelim,stretch):
    for wavefunc in funcs:
        for i, timestep in enumerate(wavefunc):
            if i == timelim:
                break
            if timestep == 0:
                j = 1
                while np.abs(j) < stretch:
                    b1, b2 = wavefunc[i - j] != 0, wavefunc[i + j] != 0
                    if b1 == True:
                        val1 = copy.copy(j)
                    if b2 == True:
                        val2 = copy.copy(j)
                    if b1 == True and b2 == True:
                        j = stretch
                    else:
                        j += 1
                timestep = (wavefunc[i - val1] + wavefunc[i + val2])/2
                wavefunc[i] = timestep
    return funcs

#waves = takeaverage(waves,199,10)
#surges = takeaverage(surges,199,10)

#plt.plot(time,waves[0],'.')
#plt.show()
#plt.plot(time,surges[0],'.')

#%%

newt = []
for t in time:
    newt.append([t])
newt = np.array(newt).astype(np.float32)

timetestnew = []
for t in timetest:
    timetestnew.append([t])
timetestnew[-1][0] = timetestnew[-1][0] + .0001
timetestnew = np.array(timetestnew).astype(np.float32) #Change datatype to mark later on

#%%
wavel = np.array(waves).astype(np.float32)
surges = np.array(surges).astype(np.float32)
wavestest = np.array(wavestest)
surgestest = np.array(surgestest)
# %%
y_train = surges.astype(np.float32)

x_train = (wavel.astype(np.float32),newt.astype(np.float32))

x_test = (wavestest.astype(np.float32),timetestnew.astype(np.float32))

y_test = np.array(surgestest.astype(np.float32))
# %%
import random
data = dde.data.TripleCartesianProd(X_train=x_train,y_train=y_train, X_test=x_test, y_test=y_test)

m = len(time)

nh = 5
net = dde.nn.DeepONetCartesianProd(
    [m,100,300],
    [nh + 1,100,300],
    'sin',
    'Glorot normal',
)
#nh = 5 but +1 for timestep


model = dde.Model(data, net)

# %%

#historical deeponet planning
boolmat = np.empty((200,6),dtype=bool)
boolmat.fill("True")

rows = np.where(np.tril(boolmat) == True)[0]
columns = np.where(np.tril(boolmat) == True)[1]

rcvec = []
for i, row in enumerate(rows):
    rcvec.append([row,columns[i]])

#Here, we get an array of all the [rows,columns] indices we want to fill the trunk net input with
#All other elements will be zero.

#%%
import cProfile
#We can use cProfile to track what takes the most time during training.
#Implementing callbacks for later detection if incoming data is train or test
#Also for future predictions

class callbacks(dde.callbacks.Callback):
    def __init__(self):
        super().__init__()
    
    def on_epoch_begin(a1, self):
        return [model.data.train_x, model.data.test_x]

        

callbacks = callbacks()

trainindicator = callbacks.on_epoch_begin([])[0][1][-1][0]
testindicator = callbacks.on_epoch_begin([])[1][1][-1][0]
"""
This is to be cleaned up later, but right now it works.

historicalinternaltrain - takes in the training timepoints and creates corresponding historical surge dataset
historicalinternaltest - same as above but for the test dataset
historicalinternalpredict - same as above but for prediction
(these above three do pretty much the same thing with the only difference being which surge dataset to use and the amount of training data)
(the first two are now just under the "historical" function)

historical - used to create the training and testing dataset, basically creates the objects train_t and test_t depending on if the incoming time data is train or test
*the way that the two can be differentiated is the last element being .0001 more than the other. This hopefully should not affect training too much.

historicalapplyfeature - what is actually used for the feature transform:
if train, sets input to the training feature transform. If test, sets input to testing feature transform.
if predict, set the input as the initial wave elevation data, predict the output, then use that data to make future predictions. Still working on this part
"""
# def historicalinternaltrain(t,notimeseries):
#     tp = np.ravel(t)
#     tp3d = []
#     for i in range(notimeseries):
#         tp3d.append(np.pad(tp[...,np.newaxis], ((0,0),(0,5)),mode='constant',constant_values=0))
#     tp3d = np.array(tp3d)
#     for j in range(notimeseries):
#         for pair in rcvec:
#                 p1,p2 = pair
#                 if p2 != 0:
#                     tp3d[j][p1][p2] = surges[j][p1 - p2]
#     return tp3d

# def historicalinternaltest(t,notimeseries):
#     tp = np.ravel(t)
#     tp3d = []
#     for i in range(notimeseries):
#         tp3d.append(np.pad(tp[...,np.newaxis], ((0,0),(0,5)),mode='constant',constant_values=0))
#     tp3d = np.array(tp3d)
#     for j in range(notimeseries):
#         for pair in rcvec:
#                 p1,p2 = pair
#                 if p2 != 0:
#                     tp3d[j][p1][p2] = surgestest[j][p1 - p2]
#     return tp3d


def historicalinternalpredict(t,notimeseries,predictiondataset):
    tp = t
    tp = tp.detach().numpy()
    tp = np.ravel(tp)
    tp3d = []
    for i in range(notimeseries):
        tp3d.append(np.pad(tp[...,np.newaxis], ((0,0),(0,5)),mode='constant',constant_values=0))
    tp3d = np.array(tp3d)
    for j in range(notimeseries):
        for pair in rcvec:
                p1,p2 = pair
                if p2 != 0:
                    tp3d[j][p1][p2] = predictiondataset[j][p1 - p2]
    return tp3d


def historical(t):
    tp = t
    if tp[-1][0] == trainindicator:
        #Optionally, this whole section could be replaced with historicalinternaltrain(tp,trainno)
        tp = np.ravel(tp)
        tp3d = []
        for i in range(trainno):
            tp3d.append(np.pad(tp[...,np.newaxis], ((0,0),(0,5)),mode='constant',constant_values=0))
        tp3d = np.array(tp3d)
        for j in range(trainno):
            for pair in rcvec:
                    p1,p2 = pair
                    if p2 != 0:
                        tp3d[j][p1][p2] = surges[j][p1 - p2]
        return(tp3d)

    else:
        if tp[-1][0] == testindicator:
            #Optionally, this whole section could be replaced with historicalinternaltest(tp,testno)
            tp = np.ravel(tp)
            tp3d = []
            for i in range(testno):
                tp3d.append(np.pad(tp[...,np.newaxis], ((0,0),(0,5)),mode='constant',constant_values=0))
            tp3d = np.array(tp3d)
            for j in range(testno):
                for pair in rcvec:
                        p1,p2 = pair
                        if p2 != 0:
                            tp3d[j][p1][p2] = surgestest[j][p1 - p2]
            print('successfully detected test time data')
        return(tp3d)

train_t = historical(newt).astype(np.float32)
test_t = historical(timetestnew).astype(np.float32)

inputpredtimesignal = np.array([np.pad(newt,((0,0),(0,5)),mode='constant',constant_values=0)]).astype(np.float32)



def regularpass(t):
    return t


def historicalapplyfeature(t):
    tp = t.detach().numpy()
    if tp[-1][0] == trainindicator:
        tp = train_t
        tp = torch.from_numpy(tp)
        return(tp)
    else:
        if tp[-1][0] == testindicator:
            tp = test_t
            tp = torch.from_numpy(tp)
            return(tp)
        else:
            global wavep
            tp = inputpredtimesignal
            net.apply_feature_transform(regularpass)
            prediction = model.predict((wavep,tp))
            tp = historicalinternalpredict(t,1,prediction)
            tp = torch.from_numpy(tp)
            return(tp)


#%%
net.apply_feature_transform(historicalapplyfeature)
#%%
model.compile('adam', lr=1e-4)
#pr = cProfile.Profile()
#pr.enable()
losshistory, train_state = model.train(iterations=40000,display_every=1)
#pr.disable()
#%%
#import pstats
#sortby = 'cumulative'
#ps = pstats.Stats(pr).sort_stats(sortby)
#ps.print_stats()
#%%
datap = pd.read_csv('compressed_data/Compressedata%i.csv'%1000)
wavep = np.array([datap.loc[1,:][1:].to_numpy()])
surgep = datap.loc[2,:][1:].to_numpy()
timep = datap.loc[0,:][1:].to_numpy()

predtime = []
for t in timep:
    predtime.append([t])
predtime = np.array(predtime).astype(np.float32)

predtime[-1][0] = predtime[-1][0] - .0001

pred2 = model.predict((wavep,predtime))

#%%



plt.plot(timep,pred2[0],'.',color='r') #output is 1x1x200, just need one of these
plt.plot(timep,surgep,'.',color='k')
plt.show()
plt.savefig("comparison.png")
#%%