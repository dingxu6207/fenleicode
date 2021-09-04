# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 16:47:03 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle

PATH = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\ztfcodefft\\'
BYDra = np.loadtxt(PATH+'BYDra.txt')
DSCT = np.loadtxt(PATH+'DSCT.txt')
EA = np.loadtxt(PATH+'EA.txt')
EW = np.loadtxt(PATH+'EW.txt')
RRAB = np.loadtxt(PATH+'RRAB.txt')
RRC = np.loadtxt(PATH+'RRC.txt')
RSCVN = np.loadtxt(PATH+'RSCVN.txt')
SR = np.loadtxt(PATH+'SR.txt')

dfdata1 = pd.DataFrame(BYDra)
dfdata1 = dfdata1.sample(n=8000)
BYDradata = np.array(dfdata1)

dfdata2 = pd.DataFrame(DSCT)
dfdata2 = dfdata2.sample(n=8000)
DSCTdata = np.array(dfdata2)

dfdata3 = pd.DataFrame(EA)
dfdata3 = dfdata3.sample(n=8000)
EAdata = np.array(dfdata3)

dfdata4 = pd.DataFrame(EW)
dfdata4 = dfdata4.sample(n=8000)
EWdata = np.array(dfdata4)


dfdata5 = pd.DataFrame(RRAB)
dfdata5 = dfdata5.sample(n=8000)
RRdata = np.array(dfdata5)


dfdata6 = pd.DataFrame(RRC)
dfdata6 = dfdata6.sample(n=8000)
RRCdata = np.array(dfdata6)


dfdata7 = pd.DataFrame(RSCVN)
dfdata7 = dfdata7.sample(n=8000)
RSCVNdata = np.array(dfdata7)
#
#
dfdata8 = pd.DataFrame(SR)
dfdata8 = dfdata8.sample(n=8000)
SRdata = np.array(dfdata8)

data1 = np.vstack((BYDradata, DSCTdata))
data2 = np.vstack((data1, EAdata))
data3 = np.vstack((data2, EWdata))
data4 = np.vstack((data3, RRdata))
data5 = np.vstack((data4, RRCdata))
data6 = np.vstack((data5, RSCVNdata))
data7 = np.vstack((data6, SRdata))

np.savetxt('ZTFDATAFFT.txt', data7)

