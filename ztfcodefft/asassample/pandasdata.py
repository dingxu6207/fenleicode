# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:18:06 2022

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
import pandas as pd


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
    
def showfigtimeflux(phases, resultmag):
    plt.figure(1)
    plt.plot(phases, resultmag, '.')
    plt.xlabel('time',fontsize=14)
    plt.ylabel('flux',fontsize=14)
    plt.pause(0.1)
    plt.clf()

def showfig(phases, resultmag):
    plt.figure(1)
    plt.plot(phases[0:2000], resultmag[0:2000], '.')
    plt.xlabel('phase',fontsize=14)
    plt.ylabel('mag',fontsize=14)
    
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
    ax.invert_yaxis() #y轴反向
    plt.pause(1)
    plt.clf()
path = 'J:\\TESSDATA\\CSV\\'
#path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\CSVDATA\\'
file = 'testcsv1.csv'

readpath = 'J:\\TESSDATA\\section1\\'

data = pd.read_csv(path+file)

dataindex = data[data['index']==0]

#r1 = pd.Series(dataindex).value_counts()
#print(r1)

hang,lie = dataindex.shape

for i in range(0, hang):
    filename = dataindex.iloc[i,1]
    print(filename)
    
    time, flux = readfits(readpath+filename)
    
    showfigtimeflux(time, flux)
    
    