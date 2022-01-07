# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:46:46 2021

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
    frequency, power = ls.autopower(minimum_frequency=0.01,maximum_frequency=10)
    index = np.argmax(power)
    maxpower = np.max(power)
    period = 1/frequency[index]
    wrongP = ls.false_alarm_probability(power.max())#0.0439717
    return period, wrongP, maxpower, frequency, power

def computeperiodbs(JDtime, targetflux):
    from astropy.timeseries import BoxLeastSquares
    model = BoxLeastSquares(JDtime, targetflux)
    results = model.autopower(0.16)
    period = results.period[np.argmax(results.power)]
    return period, 0, 0

def computebindata(lendata, fg):
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
    
    if fg==1:
        return bindata
    if fg==2:
        return (bindata*2)


def computePDM(f0, time, fluxes, flag):
    period = 1/f0
    lendata =  int((period/26)*26*len(time))
    fluxes = fluxes[0:lendata]
    time = time[0:lendata]
    mag = -2.5*np.log10(fluxes)
    mag = mag-np.mean(mag)
    S = pyPDM.Scanner(minVal=f0/10, maxVal=5*f0, dVal=0.01, mode="frequency")
    P = pyPDM.PyPDM(time, mag)

    lenmag = len(mag)
    if flag == 1:
        #bindata = computebindata(lenmag, 1)
        bindata = int(lendata/10)
    elif flag == 2:
        bindata = computebindata(lenmag/2, 2)
        
    f2, t2 = P.pdmEquiBin(bindata, S)
    delta = np.min(t2)
    pdmp = 1/f2[np.argmin(t2)]
    return pdmp, delta, f2, t2

def pholddata(per, times, fluxes):
    mags = -2.5*np.log10(fluxes)
    mags = mags-np.mean(mags)
    
    lendata =  int((per/26)*26*len(times))
     
    time = times[0:lendata]
    mag = mags[0:lendata]
    phases = foldAt(time, per)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = mag[sortIndi]
    return phases, resultmag

path = 'J:\\TESSDATA\\section1\\' 
file = 'tess2018206045859-s0001-0000000126602778-0120-s_lc.fits'

tbjd, fluxes = readfits(path+file)


period, wrongP, maxpower,  frequency, power = computeperiod(tbjd, fluxes)
#pdmp, delta, f2, t2 = computePDM(1/period, tbjd, fluxes, flag=1)
phases, resultmag = pholddata(period, tbjd, fluxes)

phases1, resultmag1 = pholddata(period*2, tbjd, fluxes)

print(wrongP)

#print('period: ', period, 'pdmp: ', pdmp)
s = np.diff(resultmag,2).std()/np.sqrt(6)
print(np.std(resultmag)/s)

s = np.diff(resultmag1,2).std()/np.sqrt(6)
print(np.std(resultmag1)/s)

plt.figure(0)
plt.plot(tbjd, fluxes, '.')
plt.xlabel('tbjd',fontsize=18)
plt.ylabel('flux',fontsize=18)
plt.savefig('TIC 38586438.png')

plt.figure(1)
plt.plot(tbjd[0:2000], fluxes[0:2000], '.')
plt.xlabel('tbjd',fontsize=18)
plt.ylabel('flux',fontsize=18)
plt.savefig('TIC 38586438.png')

plt.figure(2)
ax1 = plt.gca()
ax1.plot(phases, resultmag, '.')
plt.xlabel('phase',fontsize=18)
plt.ylabel('mag',fontsize=18)
plt.text(0.5,0.3,'Period='+str(np.round(1*period,3)),fontsize=18)
ax1.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax1.invert_yaxis() #y轴反向

plt.figure(3)
plt.plot(frequency, power)
plt.xlabel('frequency',fontsize=18)
plt.ylabel('power',fontsize=18)

plt.figure(4)
ax2 = plt.gca()
ax2.plot(phases1, resultmag1, '.')
plt.xlabel('phase',fontsize=18)
plt.ylabel('mag',fontsize=18)
plt.text(0.5,0.3,'Period='+str(np.round(2*period,3)),fontsize=18)
ax2.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax2.invert_yaxis() #y轴反向

'''
plt.figure(3)
plt.plot(f2, t2)
plt.xlabel('frequency',fontsize=18)
plt.ylabel(r"$\Theta$",fontsize=18)
'''
