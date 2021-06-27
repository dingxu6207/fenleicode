# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:29:03 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy.pyTiming import pyPDM
from PyAstronomy.pyasl import foldAt

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\EA\\'
file = '145041.csv'
data = pd.read_csv(path+file, sep = ',' )

hjdmag = data[['hjd', 'mag']]
nphjmag = np.array(hjdmag)

t = nphjmag[:,0]
y = nphjmag[:,1]


S = pyPDM.Scanner(minVal=0.1, maxVal=10, dVal=0.0001, mode="frequency")
P = pyPDM.PyPDM(t, y)

#f1, t1 = P.pdmEquiBinCover(10, 3, S)
f2, t2 = P.pdmEquiBin(10, S)
plt.figure(0)
plt.plot(f2, t2, 'gp-')
#plt.plot(f1, t1, 'rp-')
plt.xlabel('frequency',fontsize=14)
plt.ylabel('Theta', fontsize=14)
print(f2[np.argmin(t2)])

valuet = np.sort(t2[t2<0.5])
print(valuet)


    
#for i in range (0, len(valuet)):
for i in range (0, 10):
    try:
        itemindex = np.argwhere(t2 == valuet[i])
        itemindex = itemindex[0][0]

   # P=1/(f2[np.argmin(t2)])
        P=1/(f2[itemindex])

        phases = foldAt(t, P)
        sortIndi = np.argsort(phases)
        phases = phases[sortIndi]
        resultmag = y[sortIndi]

        plt.figure(i+1)
        plt.plot(phases, resultmag, '.')
        plt.title('P ='+str(np.round(P,4))+'day')
    
        ax = plt.gca()
        ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
        ax.invert_yaxis() #y轴反向
    
        plt.xlabel('phase',fontsize=14)
        plt.ylabel('mag', fontsize=14)
    except:
        print('it is ok!')