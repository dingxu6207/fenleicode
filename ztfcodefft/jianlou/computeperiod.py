# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:27:50 2021

@author: dingxu
"""

import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
from astropy.timeseries import LombScargle
import shutil
from tensorflow.keras.models import load_model
from scipy.fftpack import fft,ifft
import shutil, os

#model = load_model('modelrot.hdf5')
def classfiydata(phasemag):
    sx1 = np.linspace(0,1,100)
    sy1 = np.interp(sx1, phasemag[:,0], phasemag[:,1])
    nparraydata = np.reshape(sy1,(1,100))
    prenpdata = model.predict(nparraydata)

    index = np.argmax(prenpdata[0])
    print(index)
    return index

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

def readfits(fits_file):
    with fits.open(fits_file, mode="readonly") as hdulist:
        tess_bjds = hdulist[1].data['TIME']
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
        print(hdulist[0].header['OBJECT'])
        print(hdulist[0].header['RA_OBJ'], hdulist[0].header['DEC_OBJ'])
        
        indexflux = np.argwhere(pdcsap_fluxes > 0)
#        print(sap_fluxes)
        time = tess_bjds[indexflux]
        time = time.flatten()
        flux = pdcsap_fluxes[indexflux]
        flux =  flux.flatten()
        
        return time, flux

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

def computeperiod(JDtime, targetflux):
   
    ls = LombScargle(JDtime, targetflux, normalization='model')
    frequency, power = ls.autopower(minimum_frequency=0.01,maximum_frequency=40)
    index = np.argmax(power)
    maxpower = np.max(power)
    period = 1/frequency[index]
    wrongP = ls.false_alarm_probability(power.max())
    return period, wrongP, maxpower

def computeperiodbs(JDtime, targetflux):
    from astropy.timeseries import BoxLeastSquares
    model = BoxLeastSquares(JDtime, targetflux)
    results = model.autopower(0.16)
    period = results.period[np.argmax(results.power)]
    return period, 0, 0


def computebindata1(lendata):
    
    if lendata>5000:
        bindata = int(lendata/100)
    elif lendata>3000:
        bindata = int(lendata/10)
    elif lendata>400:
        bindata = int(lendata/6)
    elif lendata>200:
        bindata = int(lendata/3)
    else:
        bindata = int(lendata/2)
    return bindata

def computebindata2(lendata):
    
    if lendata>5000:
        bindata = int(lendata/100)
    elif lendata>3000:
        bindata = int(lendata/10)
    elif lendata>400:
        bindata = int(lendata/6)
    elif lendata>200:
        bindata = int(lendata/3)
    else:
        bindata = int(lendata/2)
    return bindata*2

def computePDM(f0, time, fluxes, flag):
    period = 1/f0
    lendata =  int((period/13)*len(time))
    fluxes = fluxes[0:lendata]
    time = time[0:lendata]
    mag = -2.5*np.log10(fluxes)
    mag = mag-np.mean(mag)
    S = pyPDM.Scanner(minVal=f0-0.01, maxVal=f0+0.01, dVal=0.001, mode="frequency")
    P = pyPDM.PyPDM(time, mag)
    #bindata = int(len(mag)/20)
    #bindata = 100
    lenmag = len(mag)
    if flag == 1:
        bindata = computebindata1(lenmag)
    elif flag == 2:
        bindata = computebindata2(lenmag/2)
        
    f2, t2 = P.pdmEquiBin(bindata, S)
    delta = np.min(t2)
    pdmp = 1/f2[np.argmin(t2)]
    return pdmp, delta

def pholddata(per, times, fluxes):
    mags = -2.5*np.log10(fluxes)
    mags = mags-np.mean(mags)
    
    lendata =  int((per/13)*len(times))
    
    time = times[0:lendata]
    mag = mags[0:lendata]
    phases = foldAt(time, per)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = mag[sortIndi]
    return phases, resultmag

def stddata(timedata, fluxdata, P):
    yuanflux = np.copy(fluxdata)
    yuanmag = -2.5*np.log10(yuanflux)
    
    phases, resultmag = pholddata(P, timedata, fluxdata)
    datamag = np.copy(resultmag)
    datanoise = np.diff(datamag,2).std()/np.sqrt(6)
    stddata = np.std(yuanmag)
    return stddata/datanoise

#TIC 259864042
#path = 'J:\\TESSDATA\\section1variable\\EA\\'
pathdelete = 'Z:\\DingXu\\TESSDATA\\jianlou\\ERROR\\'
#path = 'Z:\\DingXu\\TESSDATA\\jianlou\\section3\\EW\\' #捡漏中清除
path = 'Z:\\DingXu\\TESSDATA\\2rt\\TESSDATAVARIABLE\\section3\\EW\\' #EA EW脏数据清除
temp = []
for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-5:] == '.fits'):
           print(strfile)
           tbjd, fluxes = readfits(strfile)
           
#           plt.figure(1)
#           plt.plot(tbjd, fluxes,'.')
#           plt.pause(1)
#           plt.clf()
           
           period, wrongP, maxpower = computeperiod(tbjd, fluxes)
           stdodata1 = stddata(tbjd, fluxes, period)
           stdodata2 = stddata(tbjd, fluxes, period*2)
           
           if stdodata1>2 or stdodata2>2:
               print('it is a variable star!')
               
           if (stdodata2/stdodata1)>1.5:
               phases, resultmag = pholddata(period*2, tbjd, fluxes)
           else:
               phases, resultmag = pholddata(period, tbjd, fluxes)
               
           print('stdodata1= '+str(stdodata1))
           print('stdodata2= '+str(stdodata2))
           
           plt.figure(0)
           plt.plot(tbjd, fluxes,'.')
           plt.pause(1)
           plt.clf()
           
           
           plt.figure(1)
           plt.plot(phases, resultmag,'.')
           plt.title(str(stdodata2/stdodata1))
           ax = plt.gca()
           ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
           ax.invert_yaxis() #y轴反向
           plt.pause(1)
           plt.clf()
           
           #