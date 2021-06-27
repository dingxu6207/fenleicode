# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 23:24:33 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy.pyTiming import pyPDM
from PyAstronomy.pyasl import foldAt
from scipy import interpolate  
from tensorflow.keras.models import load_model

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\EB\\'
file = 'AP36475859_4.3578061.csv'
data = pd.read_csv(path+file, sep = ',' )

hjdmag = data[['hjd', 'mag']]
nphjmag = np.array(hjdmag)

npjd = nphjmag[:,0]
npmag = nphjmag[:,1]



P = 4.3578061
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

#num = len(phasemag[:,1])
#sx1 = np.linspace(0,1,100)
#func1 = interpolate.UnivariateSpline(phasemag[:,0], phasemag[:,1],k=3,s=s*s*num,ext=3)
#sy1 = func1(sx1)
#plt.plot(sx1, sy1,'.')
sx1 = np.linspace(0,1,100)
sy1 = np.interp(sx1, phasemag[:,0], phasemag[:,1])
plt.plot(sx1, sy1,'.')

model = load_model('eclipemodel.hdf5')
nparraydata = np.reshape(sy1,(1,100))
prenpdata = model.predict(nparraydata)

index = np.argmax(prenpdata[0])
print(index)