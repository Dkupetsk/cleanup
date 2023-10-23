#%%
#Please use deepxde for wind in this case - type  pip install -e git+https://github.com/Dkupetsk/deepxde.git@wind#egg=deepxde

import numpy as np
import torch
import deepxde as dde
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

no = 561580795 - 561579726
trainno = 50
testno = 20

def createquintuplets(a1,a2,a3,a4):
    dpoint = []
    for i, element in enumerate(a1):
        dpoint.append([element,a2[i],a3[i],a4[i]])
    return dpoint

traindata = []
surges = []
testdata = []
surgestest = []
for i in range(trainno):
    data = pd.read_csv('compressed_data_wind/Compressedata%i.csv'%i)
    uhub = data.loc[1,:][1:].to_numpy()
    w1 = data.loc[2,:][1:].to_numpy()
    w2 = data.loc[3,:][1:].to_numpy()
    w3 = data.loc[4,:][1:].to_numpy()
    traindata.append(createquintuplets(uhub,w1,w2,w3))
    surges.append(data.loc[2,:][1:].to_numpy())
time = data.loc[0,:][1:].to_numpy()
for i in range(testno):
    data = pd.read_csv('compressed_data/Compressedata%i.csv'%(i+trainno))
    uhub = data.loc[1,:][1:].to_numpy()
    w1 = data.loc[2,:][1:].to_numpy()
    w2 = data.loc[3,:][1:].to_numpy()
    w3 = data.loc[4,:][1:].to_numpy()
    testdata.append(createquintuplets(uhub,w1,w2,w3))
    surgestest.append(data.loc[2,:][1:].to_numpy())


print(traindata[24])
plt.plot(time,traindata[24],'.')
#%%
#Data formatting

newt = []
for t in time:
    newt.append([t])
newt = np.array(newt)

wavel = np.array(waves)
surges = np.array(surges)
wavestest = np.array(wavestest)
surgestest = np.array(surgestest)


# %%
y_train = np.array(surges.astype(np.float32))

x_train = (wavel.astype(np.float32),newt.astype(np.float32))

x_test = (wavestest.astype(np.float32),newt.astype(np.float32))

y_test = np.array(surgestest.astype(np.float32))

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
 #%%

randseed = np.random.randint(trainno,no)
data2=pd.read_csv('compressed_data/Compressedata%i.csv'%1000)
wavel2 = data2.loc[1,:][1:].to_numpy()
time2 = data2.loc[0,:][1:].to_numpy()
surge2 = data2.loc[2,:][1:].to_numpy()

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
plt.savefig('comparison_vanilla.png')

 # %%

#Testing on wind data (probably wont work well!)
#To be developed
data = pd.read_csv('IEA-15-240-RWT-UMaineSemi.out', delim_whitespace=True,skiprows=8)
time2 = data.iloc[:,0][::40].astype(float).to_numpy()
hub = data.iloc[:,85-12][::40].astype(float).to_numpy()
heave = data.iloc[:,12][::40].astype(float).to_numpy()
plt.plot(time2,hub,'.')
#plt.plot(time2,heave,'.')


#%%
time2 = []
waves2 = []
surge2 = []
time2 = data.iloc[:,0][::40].astype(float).to_numpy()
waves2 = data.iloc[:,85][::40].astype(float).to_numpy()
surge2 = data.iloc[:,10][::40].astype(float).to_numpy()

plt.plot(time2,waves2) #A different \eta(t)!
plt.plot(time,wavel[0]) #The first \eta(t) from the train set
plt.show()
waves2 = np.array([waves2])

pred = np.ravel(model.predict((waves2,newt)))

plt.plot(time,surge2,'k')
plt.plot(time,pred,'r')

error = np.square(np.subtract(surge2,pred)).mean()
print('Mean squared error of %f'%error)
plt.savefig('comparison_vanilla.png')

# %%

# %%
