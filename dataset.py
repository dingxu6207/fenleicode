# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 21:21:44 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle

eclipsedata = np.loadtxt('alldata.txt')
others = np.loadtxt('others.txt')

dfdata1 = pd.DataFrame(others)
dfdata1 = dfdata1.sample(n=8000)
othersdata = np.array(dfdata1)


data = np.vstack((eclipsedata, othersdata))

np.savetxt('eclipeothers.txt', data)


for i in range(28890,32000):
#    plt.figure(0)
#    plt.plot(fencontact[i,0:100], '.')
#    plt.pause(0.1)
#    plt.clf()
    
    plt.figure(1)
    plt.plot(data[i,0:100], '.')
    plt.pause(0.01)
    plt.clf()
