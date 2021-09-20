# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 23:45:55 2021

@author: dingxu
"""
import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
import matplotlib.pylab as plt
from scipy import interpolate
import os,pickle,time,shutil
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

def ztfdata(CSV_FILE_PATH, period):
    dfdata = pd.read_csv(CSV_FILE_PATH)
    hjd = dfdata['HJD']
    mag = dfdata['mag']
    rg = dfdata['band'].value_counts()
    try:
        lenr = rg['r']
    except:
        return np.array([0,0,0]),np.array([0,0,0])
     
    nphjd = np.array(hjd)
    npmag = np.array(mag)
    
    try:
        hang = rg['g']
    except:
        return np.array([0,0,0]),np.array([0,0,0])
    
    nphjd = nphjd[hang:]
    npmag = npmag[hang:]
     
    phases = foldAt(nphjd, period)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = npmag[sortIndi]

    s1 = np.diff(resultmag,2).std()/np.sqrt(6)
    s2 = np.std(resultmag)
    if (s2/s1)<2:
        return np.array([0,0,0]),np.array([0,0,0])
    if len(resultmag)<50: 
        return np.array([0,0,0]),np.array([0,0,0])

    N = 100
    x = np.linspace(0,1,N)
    y = np.interp(x, phases, resultmag) 

    fft_y = fft(y) 
    mx = np.arange(N)
    half_x = mx[range(int(N/2))]  #取一半区间
    abs_y = np.abs(fft_y) 
    normalization_y = abs_y/N            #归一化处理（双边频谱）                              
    normalization_half_y = normalization_y[range(int(N/2))] 
    normalization_half_y[0] = period/10
    
    return half_x,normalization_half_y


#tot=781602
tot = 781602
w = 10000
t1w = tot//w
dat = np.genfromtxt('Table2data.txt',dtype=str)
datemp = []
tot=dat.shape[0]
ID=0

for j in range(t1w+1):
    print(str(j)+' is ok!')
    for i in range(w):
        if ID>(tot-1):
            break
        sourceid=dat[ID,1]
        P = float(dat[ID][4])
        gmag=float(dat[ID,8])
        dirnm='Z:/DingXu/ZTF_jkf/alldata/'+str(int(sourceid)//w).zfill(4)
        filename = dirnm+'/'+str(sourceid).zfill(7)+'.csv'
        #print(filename)
        
        if os.path.getsize(filename)>100:
            if (dat[ID,24].upper()=='EA'):
                sx1,sy1 = ztfdata(filename, P)
                sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                sx1sy1 = sx1sy1.T
                if len(sx1) > 10:
                    np.savetxt('I:\\ZTFDATA\\FFTDATA1\\EA\\'+sourceid+'.txt', sx1sy1)
            
            if (dat[ID,24].upper()=='EW' and ID<100000):
                sx1,sy1 = ztfdata(filename, P)
                sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                sx1sy1 = sx1sy1.T
                if len(sx1) > 10:
                    np.savetxt('I:\\ZTFDATA\\FFTDATA1\\EW\\'+sourceid+'.txt', sx1sy1)
        
            if (dat[ID,24].upper()=='DSCT'):
                sx1,sy1 = ztfdata(filename, P)
                sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                sx1sy1 = sx1sy1.T
                if len(sx1) > 10:
                    np.savetxt('I:\\ZTFDATA\\FFTDATA1\\DSCT\\'+sourceid+'.txt', sx1sy1)
            
            if (dat[ID,24].upper()=='BYDRA' and ID<800000):
                sx1,sy1 = ztfdata(filename, P)
                sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                sx1sy1 = sx1sy1.T
                if len(sx1) > 10:
                    np.savetxt('I:\\ZTFDATA\\FFTDATA1\\BYDRA\\'+sourceid+'.txt', sx1sy1)
            
            if (dat[ID,24].upper()=='RR'):
                sx1,sy1 = ztfdata(filename, P)
                sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                sx1sy1 = sx1sy1.T
                if len(sx1) > 10:
                    np.savetxt('I:\\ZTFDATA\\FFTDATA1\\RR\\'+sourceid+'.txt', sx1sy1)
                
            if (dat[ID,24].upper()=='SR' and ID<800000):
                sx1,sy1 = ztfdata(filename, P)
                sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                sx1sy1 = sx1sy1.T
                if len(sx1) > 10:
                    np.savetxt('I:\\ZTFDATA\\FFTDATA1\\SR\\'+sourceid+'.txt', sx1sy1)

            if (dat[ID,24].upper()=='RRC'):
                sx1,sy1 = ztfdata(filename, P)
                sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                sx1sy1 = sx1sy1.T
                if len(sx1) > 10:
                    np.savetxt('I:\\ZTFDATA\\FFTDATA1\\RRC\\'+sourceid+'.txt', sx1sy1)
                
            if (dat[ID,24].upper()=='RSCVN' and ID<800000):
                sx1,sy1 = ztfdata(filename, P)
                sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                sx1sy1 = sx1sy1.T
                if len(sx1) > 10:
                    np.savetxt('I:\\ZTFDATA\\FFTDATA1\\RSCVN\\'+sourceid+'.txt', sx1sy1)

            if (dat[ID,24].upper()=='MIRA'):
                sx1,sy1 = ztfdata(filename, P)
                sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                sx1sy1 = sx1sy1.T
                if len(sx1) > 10:
                    np.savetxt('I:\\ZTFDATA\\FFTDATA1\\MIRA\\'+sourceid+'.txt', sx1sy1)
            
            if (dat[ID,24].upper()=='CEP' or dat[ID,24].upper()=='CEPII'):
                sx1,sy1 = ztfdata(filename, P)
                sx1sy1 = np.vstack((sx1, sy1)) #纵向合并矩阵
                sx1sy1 = sx1sy1.T
                if len(sx1) > 10:
                    np.savetxt('I:\\ZTFDATA\\FFTDATA1\\CEP\\'+sourceid+'.txt', sx1sy1)
        ID+=1






