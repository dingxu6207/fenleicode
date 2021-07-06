# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:06:21 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy.pyTiming import pyPDM
from PyAstronomy.pyasl import foldAt
from scipy import interpolate  
from tensorflow.keras.models import load_model

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\DSCT\\'
file = 'AP3275972_0.0983621.csv'
data = pd.read_csv(path+file, sep = ',' )

hjdmag = data[['hjd', 'mag']]
nphjmag = np.array(hjdmag)

npjd = nphjmag[:,0]
npmag = nphjmag[:,1]



P = 0.0983621
phases = foldAt(npjd, P)
sortIndi = np.argsort(phases)
phases = phases[sortIndi]
resultmag = npmag[sortIndi]
resultmag = resultmag - np.mean(resultmag)

plt.figure(1)
plt.plot(phases, resultmag,'.')

listmag = resultmag.tolist()
listmag.extend(listmag)
    
listphrase = phases.tolist()
listphrase.extend(listphrase+np.max(1)) 
    
    
nplistmag = np.array(listmag)
sortmag = np.sort(nplistmag)
maxindex = np.median(sortmag[-15:])
indexmag = listmag.index(maxindex)
    
nplistphrase = np.array(listphrase)
nplistphrase = nplistphrase-nplistphrase[indexmag]
nplistmag = np.array(listmag)
    
phasemag = np.vstack((nplistphrase, nplistmag)) #纵向合并矩阵
phasemag = phasemag.T
    
phasemag = phasemag[phasemag[:,0]>=0]
phasemag = phasemag[phasemag[:,0]<=1]
    
s=np.diff(phasemag[:,1],2).std()/np.sqrt(6)



#num = len(phasemag[:,1])
#sx1 = np.linspace(0,1,100)
#func1 = interpolate.UnivariateSpline(phasemag[:,0], phasemag[:,1],k=3,s=s*s*num,ext=3)
#sy1 = func1(sx1)
#plt.plot(sx1, sy1,'.')
sx1 = np.linspace(0,1,100)
sy1 = np.interp(sx1, phasemag[:,0], phasemag[:,1])
#plt.plot(sx1, sy1,'.')

model = load_model('resultztfmodule5.hdf5')#eclipseothers,ztfmodule
nparraydata = np.reshape(sy1,(1,100))
prenpdata = model.predict(nparraydata)

index = np.argmax(prenpdata[0])
print(index)

plt.figure(2)
plt.plot(phasemag[:,0], phasemag[:,1],'.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

if index == 0:
    plt.title('Prediction is BYDra')
    
if index == 1:
    plt.title('Prediction is DSCT')

if index == 2:
    plt.title('Prediction is EA')

if index == 3:
    plt.title('Prediction is EW')

if index == 4:
    plt.title('Prediction is RR')