# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 22:40:05 2022

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
from scipy.signal import find_peaks

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
    print(maxpower)
    period = 1/frequency[index]
    wrongP = ls.false_alarm_probability(power.max())
    return period, wrongP, maxpower, frequency, power  


def findpeak(x, y, maxpeak):
    peak_id,peak_property = find_peaks(y, height=maxpeak/10, distance=1)
    peak_freq = x[peak_id]
    peak_height = peak_property['peak_heights']
    print('peak_freq',peak_freq)
    print('peak_height',peak_height)


path = 'J:\\EADATA\\' 

#file = 'tess2018206045859-s0001-0000000419744996-0120-s_lc.fits'
file = 'tess2021039152502-s0035-0000000383055703-0205-s_lc.fits'
tbjd, fluxes = readfits(path+file)

plt.figure(0)
plt.plot(tbjd, fluxes, '.')
plt.xlabel('JD',fontsize=14)
plt.ylabel('FLUX',fontsize=14) 

comper, wrongP, maxpower, frequency, power = computeperiod(tbjd, fluxes)

plt.figure(1)
plt.plot(frequency, power)
plt.xlabel('frequentcy',fontsize=18)
plt.ylabel('power',fontsize=18) 

findpeak(frequency, power, maxpower)