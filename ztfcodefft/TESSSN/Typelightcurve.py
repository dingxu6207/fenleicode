# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 21:12:15 2021

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
import winsound
from scipy import interpolate

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
        target = hdulist[0].header['OBJECT']
        return time, flux, target
    
def zerophse(phases, resultmag):
    listmag = resultmag.tolist()
    listmag.extend(listmag)
    listphrase = phases.tolist()
    listphrase.extend(listphrase+np.max(1))
    
    nplistmag = np.array(listmag)
    sortmag = np.sort(nplistmag)
    maxindex = np.median(sortmag[-10:])
    indexmag = listmag.index(maxindex)
    nplistphrase = np.array(listphrase)
    nplistphrase = nplistphrase-nplistphrase[indexmag]
    nplistmag = np.array(listmag)
    
    phasemag = np.vstack((nplistphrase, nplistmag)) #纵向合并矩阵
    phasemag = phasemag.T
    phasemag = phasemag[phasemag[:,0]>=0]
    phasemag = phasemag[phasemag[:,0]<=1]
    return phasemag

def NEWzerophse(phases, resultmag):
    listmag = resultmag.tolist()
    listmag.extend(listmag)
    listphrase = phases.tolist()
    listphrase.extend(listphrase+np.max(1))
    
    nplistmag = np.array(listmag)
    nplistphase = np.array(listphrase)

    s = np.diff(nplistmag,2).std()/np.sqrt(6)
    num = len(nplistmag)
    #lvalue = np.max(nplistphase)
    sx1 = np.linspace(0,1,2000)
    nplistphase = np.sort(nplistphase)
    func1 = interpolate.UnivariateSpline(nplistphase, nplistmag,s=s*s*num)#强制通过所有点
    sy1 = func1(sx1)
    indexmag = np.argmax(sy1)
    nplistphase = nplistphase-sx1[indexmag]

    phasemag = np.vstack((nplistphase, nplistmag)) #纵向合并矩阵
    phasemag = phasemag.T
    phasemag = phasemag[phasemag[:,0]>=0]
    phasemag = phasemag[phasemag[:,0]<=1]
    return phasemag


def pholddata(per, times, fluxes):
    mags = -2.5*np.log10(fluxes)
    mags = mags-np.mean(mags)
    
    lendata =  int((per/26)*1.9*len(times))
     
    time = times[0:lendata]
    mag = mags[0:lendata]
    phases = foldAt(time, per)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = mag[sortIndi]
    return phases, resultmag

path = 'J:\\TESSDATA\\section26\\' 
file = 'tess2020160202036-s0026-0000000237204998-0188-s_lc.fits'

tbjd, fluxes, target = readfits(path+file)
plt.figure(3)
plt.plot(tbjd, fluxes, '.')
plt.xlabel('BJD',fontsize=18)
plt.ylabel('FLUX',fontsize=18) 
plt.title(target)  

period = 0.701971630361437
phases, resultmag = pholddata(period, tbjd, fluxes)
phasemag = NEWzerophse(phases, resultmag)

plt.figure(0)
plt.plot(phasemag[:,0], phasemag[:,1], '.')
plt.xlabel('phase',fontsize=18)
plt.ylabel('mag',fontsize=18) 
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.title(target) 