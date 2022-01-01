# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 09:09:33 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy.pyTiming import pyPDM
from PyAstronomy.pyasl import foldAt
from astropy.timeseries import LombScargle

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\testdata\\EW\\'
file = '189085_0.4665728.csv'
data = pd.read_csv(path+file, sep = ',' )

hjdmag = data[['hjd', 'mag']]
nphjmag = np.array(hjdmag)

t = nphjmag[:,0]
y = nphjmag[:,1]


def computeperiod(npjdmag):
    JDtime = npjdmag[:,0]
    targetflux = npjdmag[:,1]
    ls = LombScargle(JDtime, targetflux, normalization='model')
    frequency, power = ls.autopower(minimum_frequency=0.001,maximum_frequency=5)
    index = np.argmax(power)
    maxpower = np.max(power)
    period = 1/frequency[index]
    wrongP = ls.false_alarm_probability(power.max())
    return period, wrongP, maxpower, frequency, power


def pholdata(npjdmag, P):
    phases = foldAt(npjdmag[:,0], P)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = npjdmag[:,1][sortIndi]
    
    return phases, resultmag

def stddata(npjdmag, P):
    yuandata = np.copy(npjdmag[:,1])
    phases, resultmag = pholdata(npjdmag, P)
    datamag = np.copy(resultmag)
    datanoise = np.diff(datamag,2).std()/np.sqrt(6)
    stddata = np.std(yuandata)
    print(datanoise, stddata)
    return stddata/datanoise

period, wrongP, maxpower, frequency, power = computeperiod(nphjmag)


stdata1 = stddata(nphjmag, period)
stdata2 = stddata(nphjmag, 2*period)

if (stdata2/stdata1)>1.05:
    phases, resultmag = pholdata(nphjmag, 2*period)
else:
    phases, resultmag = pholdata(nphjmag, period)
print(stdata2/stdata1)    
print('wrongP= '+str(wrongP))
plt.figure(0)
ax = plt.gca()
ax.plot(t,y,'.')
plt.xlabel('hjd',fontsize=14)
plt.ylabel('mag', fontsize=14)

ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

plt.figure(1)
ax = plt.gca()
ax.plot(phases, resultmag, '.')
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('phase',fontsize=18)
plt.ylabel('mag',fontsize=18)
plt.title('EW')


plt.figure(2)
plt.plot(frequency, power)
plt.xlabel('frequency',fontsize=14)
plt.ylabel('power', fontsize=14)
               