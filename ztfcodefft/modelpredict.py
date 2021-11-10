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
from scipy.fftpack import fft,ifft

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\testdata\\EA\\'
file = '106979_1.0577141.csv'
data = pd.read_csv(path+file, sep = ',' )

hjdmag = data[['hjd', 'mag']]
nphjmag = np.array(hjdmag)

npjd = nphjmag[:,0]
npmag = nphjmag[:,1]



P = 1.0577141
phases = foldAt(npjd, P)
sortIndi = np.argsort(phases)
phases = phases[sortIndi]
resultmag = npmag[sortIndi]
resultmag = resultmag


N = 100
x = np.linspace(0,1,N)
y = np.interp(x, phases, resultmag) 

fft_y = fft(y) 
half_x = x[range(int(N/2))]  #取一半区间
abs_y = np.abs(fft_y) 
angle_y = np.angle(fft_y)            #取复数的角度
normalization_y = abs_y/N            #归一化处理（双边频谱）                              
normalization_half_y = normalization_y[range(int(N/2))] 
normalization_half_y[0] = P/10


plt.figure(0)
plt.plot(half_x, normalization_half_y,'.')


plt.figure(1)
plt.plot(phases, resultmag,'.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

sy1 = np.copy(normalization_half_y)
#model = load_model('modelrot.hdf5')#eclipseothers,ztfmodule
#sy1 = sy1[0:45]
model = load_model('model50.hdf5')#eclipseothers,ztfmodule
nparraydata = np.reshape(sy1,(1,50))
prenpdata = model.predict(nparraydata)

index = np.argmax(prenpdata[0])
print(index)

if index == 0:
    plt.title('Prediction is ROT')
    
if index == 1:
    plt.title('Prediction is DSCT')

if index == 2:
    plt.title('Prediction is EA')

if index == 3:
    plt.title('Prediction is EW')

if index == 4:
    plt.title('Prediction is MIRA')
    
if index == 5:
    plt.title('Prediction is RRAB')
    
if index == 6:
    plt.title('Prediction is RRC')
    
if index == 7:
    plt.title('Prediction is SR')  
    
if index == 8:
    plt.title('Prediction is CEP') 

    
if index == 9:
    plt.title('Prediction is CEP') 
    
''' 
s=np.diff(phasemag[:,1],2).std()/np.sqrt(6)



#num = len(phasemag[:,1])
#sx1 = np.linspace(0,1,100)
#func1 = interpolate.UnivariateSpline(phasemag[:,0], phasemag[:,1],k=3,s=s*s*num,ext=3)
#sy1 = func1(sx1)
#plt.plot(sx1, sy1,'.')
sx1 = np.linspace(0,1,100)
sy1 = np.interp(sx1, phasemag[:,0], phasemag[:,1])
#plt.plot(sx1, sy1,'.')

model = load_model('resultztfmodel.hdf5')#eclipseothers,ztfmodule
nparraydata = np.reshape(sy1,(1,100))
prenpdata = model.predict(nparraydata)

index = np.argmax(prenpdata[0])
print(index)

plt.figure(2)
plt.plot(phasemag[:,0], phasemag[:,1],'.')
plt.plot(sx1, sy1,'.')
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
    
if index == 5:
    plt.title('Prediction is SR')
'''