# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 16:47:03 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle

PATH = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\'
ROT = np.loadtxt(PATH+'ROT.txt') #0
DSCT = np.loadtxt(PATH+'DSCT.txt') #1
EA = np.loadtxt(PATH+'EA.txt')     #2
EW = np.loadtxt(PATH+'EW.txt')     #3
MIRA = np.loadtxt(PATH+'MIRA.txt') #4
RRAB = np.loadtxt(PATH+'RRAB.txt') #5
RRC = np.loadtxt(PATH+'RRC.txt')   #6
#RSCVN = np.loadtxt(PATH+'RSCVN.txt')
SR = np.loadtxt(PATH+'SR.txt')     #7
SR[:,54] = 7
CEP = np.loadtxt(PATH+'CEP1.txt')   #8
CEP[:,54] = 8

dfdata1 = pd.DataFrame(ROT)
dfdata1 = dfdata1.sample(n=6800)
BYDradata = np.array(dfdata1)
#
dfdata2 = pd.DataFrame(DSCT)
dfdata2 = dfdata2.sample(n=6800)
DSCTdata = np.array(dfdata2)
#
dfdata3 = pd.DataFrame(EA)
dfdata3 = dfdata3.sample(n=6800)
EAdata = np.array(dfdata3)
#
dfdata4 = pd.DataFrame(EW)
dfdata4 = dfdata4.sample(n=6800)
EWdata = np.array(dfdata4)
#
#
dfdata5 = pd.DataFrame(RRAB)
dfdata5 = dfdata5.sample(n=6800)
RRdata = np.array(dfdata5)
#
#
dfdata6 = pd.DataFrame(RRC)
dfdata6 = dfdata6.sample(n=6800)
RRCdata = np.array(dfdata6)
#
#
#dfdata7 = pd.DataFrame(RSCVN)
#dfdata7 = dfdata7.sample(n=6800)
#RSCVNdata = np.array(dfdata7)
#
#
dfdata7 = pd.DataFrame(SR)
dfdata7 = dfdata7.sample(n=6800)
SRdata = np.array(dfdata7)

dfdata8 = pd.DataFrame(MIRA)
dfdata8 = dfdata8.sample(n=6800)
MIRAdata = np.array(dfdata8)
#
dfdata9 = pd.DataFrame(CEP)
dfdata9 = dfdata9.sample(n=6800)
CEPdata = np.array(dfdata9)

data1 = np.vstack((BYDradata, DSCTdata))
data2 = np.vstack((data1, EAdata))
data3 = np.vstack((data2, EWdata))
data4 = np.vstack((data3, RRdata))
data5 = np.vstack((data4, RRCdata))
#data6 = np.vstack((data5, RSCVNdata))
data6 = np.vstack((data5, SRdata))
#data7 = np.loadtxt('ZTFDATAFFT.txt')
data7 = np.vstack((data6, MIRAdata))
data8 = np.vstack((data7, CEPdata))

np.savetxt('ZTFDATAFFTALLSrotN3.txt', data8)

