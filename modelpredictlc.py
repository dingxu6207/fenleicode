# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:32:29 2021

@author: dingxu
"""

from tensorflow.keras.models import load_model
#from keras.models import load_model
from PyAstronomy.pyasl import foldAt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing  
from scipy import interpolate  
from scipy.fftpack import fft,ifft
import pandas as pd

#model = load_model('modelrot.hdf5')
model = load_model('model10N.hdf5')
model.summary()

def classifyfftdata(phases, resultmag, P):
    phases = np.copy(phases)
    resultmag = np.copy(resultmag)
    N = 1000
    x = np.linspace(0,1,N)
    y = np.interp(x, phases, resultmag) 

    fft_y = fft(y) 
    half_x = x[range(int(N/2))]  #取一半区间
    abs_y = np.abs(fft_y) 
    normalization_y = abs_y/N            #归一化处理（双边频谱）                              
    normalization_half_y = normalization_y[range(int(N/2))] 
    normalization_half_y[0] = P/10
    sy1 = np.copy(normalization_half_y)
    #model = load_model('modelall.hdf5')#eclipseothers,ztfmodule
    sy1 = sy1[0:10]
    nparraydata = np.reshape(sy1,(1,10))
    prenpdata = model.predict(nparraydata)

    index = np.argmax(prenpdata[0])
    return index,np.max(prenpdata[0]),x,y

def pholddata(per, times, fluxes):
    mags = -2.5*np.log10(fluxes)
    mags = mags-np.mean(mags)
    phases = foldAt(times, per)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = mags[sortIndi]
    return phases, resultmag

def zerophse(phases, resultmag):
    listmag = resultmag.tolist()
    listmag.extend(listmag)
    listphrase = phases.tolist()
    listphrase.extend(listphrase+np.max(1))
    
    nplistmag = np.array(listmag)
    sortmag = np.sort(nplistmag)
    maxindex = np.median(sortmag[-1:])
    indexmag = listmag.index(maxindex)
    nplistphrase = np.array(listphrase)
    nplistphrase = nplistphrase-nplistphrase[indexmag]
    nplistmag = np.array(listmag)
    
    phasemag = np.vstack((nplistphrase, nplistmag)) #纵向合并矩阵
    phasemag = phasemag.T
    phasemag = phasemag[phasemag[:,0]>=0]
    phasemag = phasemag[phasemag[:,0]<=1]
    
    return phasemag

#path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\testdata\\RRC\\'
file = 'lcdd4_0_1204_1224.9_61.7_341.96_-32.159_c.dat'
nphjmag = np.loadtxt(file)
npjd = nphjmag[:,0]
npmag = nphjmag[:,1]

P = 3.404 

phases, resultmag = pholddata(P, npjd, npmag)
phasemag = zerophse(phases, resultmag)

index,prob,x,y = classifyfftdata(phases, resultmag, P)


plt.figure(2)
plt.plot(x, y,'.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

plt.figure(1)
plt.plot(phases, resultmag,'.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

if index == 0:
    plt.title('Prediction is ROT')
    print('probility is ROT'+str(prob))

    
if index == 1:
    plt.title('Prediction is DSCT')
    print('probility is DSCT'+str(prob))

if index == 2:
    plt.title('Prediction is EA')
    print('probility is EA'+str(prob))

if index == 3:
    plt.title('Prediction is EW')
    print('probility is EW'+str(prob))

if index == 4:
    plt.title('Prediction is MIRA')
    print('probility is MIRA'+str(prob))
    
if index == 5:
    plt.title('Prediction is RRAB')
    print('probility is RRAB'+str(prob))
    
if index == 6:
    plt.title('Prediction is RRC')
    print('probility is RRC'+str(prob))
    
if index == 7:
    plt.title('Prediction is SR')  
    print('probility is SR'+str(prob))
    
if index == 8:
    plt.title('Prediction is CEP') 
    print('probility is CEP'+str(prob))



