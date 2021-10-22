# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:56:50 2021

@author: dingxu
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
from astropy.timeseries import LombScargle
from tensorflow.keras.models import load_model
from scipy.fftpack import fft,ifft

model = load_model('modelalls.hdf5')
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
    
def computeperiod(JDtime, targetflux):
   
    ls = LombScargle(JDtime, targetflux, normalization='model')
    frequency, power = ls.autopower(minimum_frequency=0.01,maximum_frequency=40)
    index = np.argmax(power)
    maxpower = np.max(power)
    period = 1/frequency[index]
    wrongP = ls.false_alarm_probability(power.max())
    return period, wrongP, maxpower    
  
def pholddata(per, times, fluxes):
    mags = -2.5*np.log10(fluxes)
    mags = mags-np.mean(mags)
    
    lendata =  int((per/12)*len(times))
     
    time = times[0:lendata]
    mag = mags[0:lendata]
    phases = foldAt(time, per)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = mag[sortIndi]
    return phases, resultmag

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
    if fg == 1:
        return bindata
    if fg == 2:
        return (bindata*2)
    

def computePDM(f0, time, fluxes, flag):
    period = 1/f0
    lendata =  int((period/12)*len(time))
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
        bindata = computebindata(lenmag, 1)
    elif flag == 2:
        bindata = computebindata(lenmag/2, 2)
        
    f2, t2 = P.pdmEquiBin(bindata, S)
    delta = np.min(t2)
    pdmp = 1/f2[np.argmin(t2)]
    return pdmp, delta

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

path = 'I:\\TESSDATA\\section1\\' 
file = 'tess2018206045859-s0001-0000000308851582-0120-s_lc.fits'

tbjd, fluxes = readfits(path+file)
mag = -2.5*np.log10(fluxes)
mag = mag-np.mean(mag)



comper, wrongP, maxpower = computeperiod(tbjd, fluxes)
pdmp, delta  = computePDM(1/comper, tbjd, fluxes, 1)
if delta <0.5 and pdmp < 12:
    pdmp2, delta2  = computePDM(1/(comper*2), tbjd, fluxes, 2)
    print(delta/delta2)       
    if (delta/delta2)<1.5:
        p = comper
        phases, resultmag = pholddata(comper, tbjd, fluxes)
    else:
        p = comper*2 
        phases, resultmag = pholddata(comper*2, tbjd, fluxes)


phases, resultmag = pholddata(p, tbjd, fluxes)
index = classifyfftdata(phases, resultmag, p)
#from astropy.timeseries import BoxLeastSquares
#model = BoxLeastSquares(tbjd, mag)
#results = model.autopower(0.2)
#print(results.period[np.argmax(results.power)])  

plt.figure(0)
plt.plot(tbjd, fluxes, '.')
plt.xlabel('JD',fontsize=14)
plt.ylabel('FLUX',fontsize=14) 

plt.figure(1)
plt.plot(phases, resultmag, '.')
plt.xlabel('phase',fontsize=14)
plt.ylabel('mag',fontsize=14)
ax1 = plt.gca()
ax1.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax1.invert_yaxis() #y轴反向



