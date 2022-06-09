# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 21:46:13 2022

@author: dingxu
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ztfquery import lightcurve
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
from astropy.timeseries import LombScargle
from tensorflow.keras.models import load_model
from scipy.fftpack import fft,ifft

model = load_model('model50p10cep.hdf5')

def computeperiod(npjdmag):
    JDtime = npjdmag[:,0]
    targetflux = npjdmag[:,1]
    ls = LombScargle(JDtime, targetflux, normalization='model')
    frequency, power = ls.autopower(minimum_frequency=0.025,maximum_frequency=20)
    index = np.argmax(power)
    maxpower = np.max(power)
    period = 1/frequency[index]
    wrongP = ls.false_alarm_probability(power.max())
    return period, wrongP, maxpower


def pholdata(npjdmag, P):
    phases = foldAt(npjdmag[:,0], P)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = npjdmag[:,1][sortIndi]
    
    return phases, resultmag

def computePDM(npjdmag,P):
    timedata = npjdmag[:,0]
    magdata = npjdmag[:,1]
    f0 =1/(2*P) 
    S = pyPDM.Scanner(minVal=f0-0.01, maxVal=f0+0.01, dVal=0.001, mode="frequency")
    P = pyPDM.PyPDM(timedata, magdata)
    bindata = int(len(magdata)/4)
    #bindata = 10
    f2, t2 = P.pdmEquiBin(bindata, S)
    delta = np.min(t2)
    pdmp = 1/f2[np.argmin(t2)]
    return pdmp, delta

def showfig(phases, resultmag, index):
    plt.figure(1)
    plt.plot(phases, resultmag, '.')
    plt.xlabel('phase',fontsize=14)
    plt.ylabel('mag',fontsize=14)
    plt.title(str(index))
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
    ax.invert_yaxis() #y轴反向
    plt.pause(1)
    plt.clf()
    
    
def showmjdmag(mjdmag):
    plt.figure(2)
    plt.clf()
    mjd = mjdmag[:,0]
    mag = mjdmag[:,1]
    plt.plot(mjd, mag, '.')
    plt.xlabel('MJD',fontsize=14)
    plt.ylabel('mag',fontsize=14)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
    ax.invert_yaxis() #y轴反向
    plt.pause(0.1)
    
    
def showcmd(highdata,RA,DEC):
    plt.figure(3)
    plt.clf()
    plt.scatter(highdata.iloc[:,0], highdata.iloc[:,1], c ='r', marker='o', s=1)
    plt.plot(RA,DEC,'o',c='g')
    plt.xlabel('RA',fontsize=14)
    plt.ylabel('DEC',fontsize=14)
    plt.pause(1)
    
def computePDMA(npjdmag):
    timedata = npjdmag[:,0]
    magdata = npjdmag[:,1]
    S = pyPDM.Scanner(minVal=0.005, maxVal=20, dVal=0.0001, mode="frequency")
    P = pyPDM.PyPDM(timedata, magdata)
    bindata = int(len(magdata)/4)
    #bindata = 10
    f2, t2 = P.pdmEquiBin(bindata, S)
    delta = np.min(t2)
    pdmp = 1/f2[np.argmin(t2)]
    return pdmp, delta   

def computeperiodbs(JDtime, targetflux):
    from astropy.timeseries import BoxLeastSquares
    model = BoxLeastSquares(JDtime, targetflux)
    results = model.autopower(0.16)
    period = results.period[np.argmax(results.power)]
    return period, 0, 0

def stddata(npjdmag, P):
    yuandata = np.copy(npjdmag[:,1])
    phases, resultmag = pholdata(npjdmag, P)
    datamag = np.copy(resultmag)
    datanoise = np.diff(datamag,2).std()/np.sqrt(6)
    stdndata = np.std(yuandata)
    return stdndata/datanoise

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
    normalization_half_y[0] = P
    sy1 = np.copy(normalization_half_y)
    #model = load_model('modelall.hdf5')#eclipseothers,ztfmodule
    #nparraydata = np.reshape(sy1,(1,50,1)) #cnnmodel
    sy1 = sy1[0:50]
    nparraydata = np.reshape(sy1,(1,50)) #mlpmodel
    prenpdata = model.predict(nparraydata)

    index = np.argmax(prenpdata[0])
    return index, np.max(prenpdata[0])

alltem= []
path = 'I:\\CG20variable\\'
for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-4:] == '.csv'):
           print(root[16:])
           
           try:
               dfdata = pd.read_csv(strfile)
               dfdata = dfdata[dfdata["filtercode"]=='zr']
               dfdata = dfdata[dfdata["catflags"]!= 32768]
               mjdmag = dfdata[['mjd', 'mag']]
               npmjdmag = np.array(mjdmag)
               period, wrongP, maxpower = computeperiod(npmjdmag)
               P1 = period
               P2 = period*2
               stddata1 = stddata(npmjdmag, P1)
               stddata2 = stddata(npmjdmag, P2)
               
               print('stddata = ', (stddata2/stddata1))
           
               if (stddata2/stddata1)>1.03:
                   per = P2
               else:
                   per = P1
               
               phases, resultmag = pholdata(npmjdmag, per)
               index,prob = classifyfftdata(phases, resultmag, per)
               
               showfig(phases, resultmag, index)
               temp = [root[16:], file, per, index, np.max(resultmag)- np.min(resultmag), np.log10(wrongP), stddata2]
               alltem.append(temp)
           except:
               continue
         
name=['clustername','RADEC','per','INDEX','MAXMIN','wrongp', 'SNR']      
test = pd.DataFrame(columns=name,data=alltem)#数据有三列，列名分别为one,two,three
test.to_csv('testcsv.csv',encoding='gbk')          
           
           