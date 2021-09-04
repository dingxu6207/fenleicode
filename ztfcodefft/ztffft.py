# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 21:33:38 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate
import os,pickle,time
from scipy import interpolate
from scipy.fftpack import fft,ifft

P = 0.5591434
CSV_FILE_PATH = '7.csv'

dfdata = pd.read_csv(CSV_FILE_PATH)
hjd = dfdata['HJD']
mag = dfdata['mag']
rg = dfdata['band'].value_counts()
lenr = rg['r']

nphjd = np.array(hjd)
npmag = np.array(mag)

hang = rg['g']
nphjd = nphjd[hang:]
npmag = npmag[hang:]
#npmag = npmag[hang:]-np.mean(npmag[hang:])
    
phases = foldAt(nphjd, P)
sortIndi = np.argsort(phases)
phases = phases[sortIndi]
resultmag = npmag[sortIndi]


N = 100
x = np.linspace(0,1,N)
y = np.interp(x, phases, resultmag) 

fft_y = fft(y) 
half_x = x[range(int(N/2))]  #取一半区间
abs_y = np.abs(fft_y) 
angle_y = np.angle(fft_y)            #取复数的角度
normalization_y = abs_y/N            #归一化处理（双边频谱）                              
normalization_half_y = normalization_y[range(int(N/2))] 
normalization_half_y[0] = P/20
plt.figure(0)
plt.plot(phases, resultmag, '.')

plt.figure(1)
plt.plot(half_x, normalization_half_y)