# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 13:46:23 2022

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
        
        return time, flux
    
    
 
path = 'I:\\TESSDATA\\section4\\EA\\' 

#plt.figure(1)
#plt.plot(tbjd, fluxes, '.')
#plt.xlabel('tbjd',fontsize=14)
#plt.ylabel('flux',fontsize=14)


for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-5:] == '.fits'):
           tbjd, fluxes = readfits(path+file)
           magdata = -2.5*np.log10(fluxes)
           s = np.diff(magdata,2).std()/np.sqrt(6)
           print('noise =', s)
           
           plt.clf()
           plt.figure(0)
           ax = plt.gca()
           ax.plot(tbjd, magdata)
           plt.xlabel('tbjd',fontsize=14)
           plt.ylabel('mag',fontsize=14) 
           plt.title(str(s))
           ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
           ax.invert_yaxis() #y轴反向
           plt.pause(4)
           