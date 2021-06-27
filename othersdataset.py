# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 17:25:20 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy.pyTiming import pyPDM
from PyAstronomy.pyasl import foldAt
from scipy import interpolate  
#from tensorflow.keras.models import load_model

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\other\\'
file = 'AP3433419_2.0051253.csv'
data = pd.read_csv(path+file, sep = ',' )

hjdmag = data[['hjd', 'mag']]
nphjmag = np.array(hjdmag)

npjd = nphjmag[:,0]
npmag = nphjmag[:,1]



P = 2.0051253
phases = foldAt(npjd, P)
sortIndi = np.argsort(phases)
phases = phases[sortIndi]
resultmag = npmag[sortIndi]
resultmag = resultmag - np.mean(resultmag)

plt.figure(0)
plt.plot(phases, resultmag,'.')

listmag = resultmag.tolist()
listmag.extend(listmag)
    
listphrase = phases.tolist()
listphrase.extend(listphrase+np.max(1)) 
    
    
nplistmag = np.array(listmag)
sortmag = np.sort(nplistmag)
maxindex = np.median(sortmag[-9:])
indexmag = listmag.index(maxindex)
    
nplistphrase = np.array(listphrase)
nplistphrase = nplistphrase-nplistphrase[indexmag]
nplistmag = np.array(listmag)
    
phasemag = np.vstack((nplistphrase, nplistmag)) #纵向合并矩阵
phasemag = phasemag.T
    
phasemag = phasemag[phasemag[:,0]>=0]
phasemag = phasemag[phasemag[:,0]<=1]
    
s=np.diff(phasemag[:,1],2).std()/np.sqrt(6)

plt.figure(1)
plt.plot(phasemag[:,0], phasemag[:,1],'.')


sx1 = np.linspace(0,1,100)
sy1 = np.interp(sx1, phasemag[:,0], phasemag[:,1])
plt.plot(sx1, sy1,'.')

datatemp = []
for i in range(0,1333):
    data1 = np.random.uniform(low=0.8, high=1.2)*sy1
    data1 = data1 - np.mean(data1)
    listdata = data1.tolist()
    listdata.append(0)
    listdata.append(0)
    listdata.append(0)
    listdata.append(0)
    listdata.append(4)
    datatemp.append(listdata)
#    plt.figure(2)
#    plt.plot(data1,'.')
#    plt.pause(1)
#    plt.clf()

    
npdata = np.array(datatemp)
np.savetxt('data6.txt', npdata)

#model = load_model('weights-improvement-00001-0.9327.hdf5')
#nparraydata = np.reshape(sy1,(1,100))
#prenpdata = model.predict(nparraydata)
#
#index = np.argmax(prenpdata[0])
#print(index)