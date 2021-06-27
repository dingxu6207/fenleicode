# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 09:56:50 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle

contactdata = np.loadtxt('contact.txt')
semicontact = np.loadtxt('semicontact.txt')
fencontact = np.loadtxt('fencontact.txt')
#others = np.loadtxt('others.txt')

dfdata1 = pd.DataFrame(contactdata)
dfdata1 = dfdata1.sample(n=8000)
contactdata = np.array(dfdata1)

dfdata2 = pd.DataFrame(semicontact)
dfdata2 = dfdata2.sample(n=8000)
semidata = np.array(dfdata2)

dfdata3 = pd.DataFrame(fencontact)
dfdata3 = dfdata3.sample(n=8000)
fendata = np.array(dfdata3)

#dfdata = pd.DataFrame(others)
#dfdata = dfdata.sample(n=9676)
#othersdata = np.array(dfdata)

data = np.vstack((contactdata, semidata))
data = np.vstack((data, fendata))


for i in range(len(data)):
    data[i,0:100] = -2.5*np.log10(data[i,0:100]) 
    data[i,0:100] = (data[i,0:100])+0.03*np.random.normal(0,1,100)
    data[i,0:100] = data[i,0:100]-np.mean(data[i,0:100])
    
#alldata = np.vstack((others, data))

plt.plot(data[:,104])
np.savetxt('alldata.txt', np.round(data,3))

for i in range(10000,len(data)):
#    plt.figure(0)
#    plt.plot(fencontact[i,0:100], '.')
#    plt.pause(0.1)
#    plt.clf()
    
    plt.figure(1)
    plt.plot(data[i,0:100], '.')
    plt.pause(0.01)
    plt.clf()