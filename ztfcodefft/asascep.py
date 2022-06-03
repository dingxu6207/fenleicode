# -*- coding: utf-8 -*-
"""
Created on Sat May 28 22:01:24 2022

@author: dingxu
"""

import os
import pandas as pd
from astropy.timeseries import LombScargle
import numpy as np
from PyAstronomy.pyasl import foldAt
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

def computeperiod(JDtime, targetflux):
   
    ls = LombScargle(JDtime, targetflux, normalization='model')
    frequency, power = ls.autopower(minimum_frequency=0.01,maximum_frequency=40)
    index = np.argmax(power)
    maxpower = np.max(power)
    period = 1/frequency[index]
    wrongP = ls.false_alarm_probability(power.max())
    return period, wrongP, maxpower

def pholddata(per, times, fluxes):
    
    phases = foldAt(times, per)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = fluxes[sortIndi]
    return phases, resultmag

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

    return sy1

temp = []
path = 'J:\\CEP\\'
for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-4:] == '.csv'):
           data = pd.read_csv(strfile)
           print('it is ok!')
           HJD = data['hjd']
           mag = data['mag']
           try:
               period, wrongP, maxpower = computeperiod(HJD, mag)
               print(period)
               phases, resultmag = pholddata(period, HJD, mag)
               
               snr = np.std(resultmag)/(np.diff(resultmag, 2).std()/np.sqrt(6))
               print('snr = ', snr)
               
               if snr>1.5:
                   sxdata = classifyfftdata(phases, resultmag, period)
                   temp.append(sxdata)
               
               plt.clf()
               plt.figure(0)
               plt.plot(phases, resultmag, '.')
               plt.xlabel('phase',fontsize=14)
               plt.ylabel('mag',fontsize=14)
               plt.title(str(snr))
               ax1 = plt.gca()
               ax1.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
               ax1.invert_yaxis() #y轴反向
               plt.pause(10)
               
               
           except:
               print('it is error!')
               
               
nptemp = np.array(temp)
np.savetxt('nptemp.txt', nptemp)