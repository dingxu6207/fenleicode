# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:32:29 2021

@author: dingxu
"""

from tensorflow.keras.models import load_model
#from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing  
from scipy import interpolate  

model = load_model('phmodsample2.h5')
model.summary()

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\testdata\\'
file = '5.lc'


data = np.loadtxt(path+file)
datam = np.copy(data)
data = data[0:100,1]

data = -2.5*np.log10(data)

datay = data-np.mean(data)

sx1 = np.linspace(0,1,100)
func1 = interpolate.UnivariateSpline(datam[0:100,0], datay,s=0)#强制通过所有点
sy1 = func1(sx1)

nparraydata = np.reshape(sy1,(1,100))
prenpdata = model.predict(nparraydata)

print(prenpdata)

index = np.argmax(prenpdata[0])
print(index)

plt.figure(0)
plt.plot(sx1, sy1, '.')


