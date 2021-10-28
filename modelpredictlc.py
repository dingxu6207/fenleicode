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

model = load_model('modelrot.hdf5')
model.summary()

def classifyfftdata(phases, resultmag, P):
    phases = np.copy(phases)
    resultmag = np.copy(resultmag)
    N = 100
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
    nparraydata = np.reshape(sy1,(1,50))
    prenpdata = model.predict(nparraydata)

    index = np.argmax(prenpdata[0])
    return index

def pholddata(per, times, fluxes):
    mags = -2.5*np.log10(fluxes)
    mags = mags-np.mean(mags)
    phases = foldAt(times, per)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = mags[sortIndi]
    return phases, resultmag


#path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\testdata\\RRC\\'
file = 'lcdd2_3_425_569.1_1013.9_16.947_43.146_b.dat'
nphjmag = np.loadtxt(file)
npjd = nphjmag[:,0]
npmag = nphjmag[:,1]

P = 0.12*2

phases, resultmag = pholddata(P, npjd, npmag)
index = classifyfftdata(phases, resultmag, P)


plt.figure(1)
plt.plot(phases, resultmag,'.')
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


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

