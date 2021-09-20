# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:37:20 2021

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

model = load_model('modelalls.hdf5')
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


def computebindata(lendata):
    
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
        bindata = computebindata(lenmag)
    elif flag == 2:
        bindata = computebindata(lenmag/2)
        
    f2, t2 = P.pdmEquiBin(bindata, S)
    delta = np.min(t2)
    pdmp = 1/f2[np.argmin(t2)]
    return pdmp, delta

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

path = 'I:\\TESSDATA\\section1\\'
bydrapath = 'I:\\TESSDATA\\section3variable\\BYDRA\\'
dsctpath = 'I:\\TESSDATA\\section3variable\\DSCT\\'
eapath = 'I:\\TESSDATA\\section3variable\\EA\\'
ewpath = 'I:\\TESSDATA\\section3variable\\EW\\'
mirapath = 'I:\\TESSDATA\\section3variable\\MIRA\\'
rrabpath = 'I:\\TESSDATA\\section3variable\\RRAB\\'
rrcpath = 'I:\\TESSDATA\\section3variable\\RRC\\'
rscvnpath = 'I:\\TESSDATA\\section3variable\\RSCVN\\'
srpath = 'I:\\TESSDATA\\section3variable\\SR\\'
ceppath = 'I:\\TESSDATA\\section3variable\\CEP\\'
unkownpath = 'I:\\TESSDATA\\section3variable\\UNKNOWN\\'
count = 0
for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-5:] == '.fits'):
           print(strfile)
       try:
               tbjd, fluxes = readfits(strfile)
               count = count+1
               print('it is time'+str(count))
           
               comper, wrongP, maxpower = computeperiod(tbjd, fluxes)
               pdmp, delta  = computePDM(1/comper, tbjd, fluxes, 1)
               if delta <0.5 and pdmp < 12:
                   pdmp2, delta2  = computePDM(1/(comper*2), tbjd, fluxes, 2)
           
                   if (delta < delta2):
                       p = comper
                       phases, resultmag = pholddata(comper, tbjd, fluxes)
                   else:
                       p = comper*2 
                       phases, resultmag = pholddata(comper*2, tbjd, fluxes)
           

                   
                   index = classifyfftdata(phases, resultmag, p)
           
                   if index == 0:
                       shutil.copy(strfile,bydrapath)
    
                   if index == 1:
                       shutil.copy(strfile,dsctpath)

                   if index == 2:
                       shutil.copy(strfile,eapath)

                   if index == 3:
                       shutil.copy(strfile,ewpath)

                   if index == 4:
                       shutil.copy(strfile,mirapath)
    
                   if index == 5 :
                       shutil.copy(strfile,rrabpath)
                       
                   if index == 6 :
                       shutil.copy(strfile,rrcpath)
                       
                   if index == 7 :
                       shutil.copy(strfile,rscvnpath) 
                       
                   if index == 8 :
                       shutil.copy(strfile,srpath) 
                       
                   if index == 9 :
                       shutil.copy(strfile,ceppath)
                   
                   print(str(index)+'is ok!')
               elif delta < 0.5 and pdmp > 12:
                   shutil.copy(strfile,unkownpath)
                    
       except:
              continue

               
               
               