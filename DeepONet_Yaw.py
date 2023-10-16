#%%
#Please use regular DeepXDE for this case
import numpy as np
import torch

def sigmoid(x):
    return 100/(1 + np.exp(-5*(x - 1)))

def sigmoidprime(x):
    return 500*np.exp(-5*(x-1))/(1 + np.exp(-5*(x-1))**2)

lambdas = torch.from_numpy(np.zeros(200))

num_train = 0
import deepxde as dde


import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

no = 561580795 - 561579726
trainno = 72
testno = 15

waves = []
yaws = []
wavestest = []
yawstest = []
#In order: 0 time, 1 wave elevation, 2 surge, 3 sway, 4 heave, 5 pitch, 6 roll, 7 yaw, 8 T1, 9 T2, 10 T3

for i in range(trainno):
    data = pd.read_csv('compressed_data/Compressedata%i.csv'%i)
    waves.append(data.loc[1,:][1:].to_numpy())
    yaws.append(data.loc[7,:][1:].to_numpy())
for i in range(testno):
    data = pd.read_csv('compressed_data/Compressedata%i.csv'%(i+trainno))
    wavestest.append(data.loc[1,:][1:].to_numpy())
    yawstest.append(data.loc[7,:][1:].to_numpy())

time = data.loc[0,:][1:].to_numpy()


#Data formatting

newt = []
for t in time:
    newt.append([t])
newt = np.array(newt)

wavel = np.array(waves)
yaws = np.array(yaws)
wavestest = np.array(wavestest)
yawstest = np.array(yawstest)


# %%
y_train = np.array(yaws.astype(np.float32))

x_train = (wavel.astype(np.float32),newt.astype(np.float32))

x_test = (wavestest.astype(np.float32),newt.astype(np.float32))

y_test = np.array(yawstest.astype(np.float32))

# %%
data = dde.data.TripleCartesianProd(X_train=x_train,y_train=y_train, X_test=x_test, y_test=y_test)

m = len(time)

net = dde.nn.deeponet.DeepONetCartesianProd(
    [m,100,300],
    [1,100,100,300],
    'sin',
    'Glorot normal',
)


model = dde.Model(data, net)

#%%

model.compile("adam", lr=1e-4)

losshistory, train_state = model.train(iterations=20000)

randseed = np.random.randint(trainno,no)
data2=pd.read_csv('compressed_data/Compressedata%i.csv'%randseed)
wavel2 = data2.loc[1,:][1:].to_numpy()
time2 = data2.loc[0,:][1:].to_numpy()
surge2 = data2.loc[7,:][1:].to_numpy()

wavel2 = wavel2
surge2 = surge2
plt.plot(time2,wavel2) #A different \eta(t)!
plt.plot(time,wavel[0]) #The first \eta(t) from the train set
plt.show()
wavel2 = np.array([wavel2])

pred = np.ravel(model.predict((wavel2,newt)))

plt.plot(time,surge2,'k')
plt.plot(time,pred,'r')

error = np.square(np.subtract(surge2,pred)).mean()
print('Mean squared error of %f'%error)
plt.savefig('comparison.png')

# %%
