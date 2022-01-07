# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 21:51:40 2022

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('datamag.txt')

phase = data[:,0]
mag = data[:,1]

#plt.plot(phase, mag, '.')

plt.figure(0)
ax1 = plt.gca()
ax1.plot(phase, mag, '.')
plt.xlabel('phase',fontsize=18)
plt.ylabel('mag',fontsize=18)
#plt.text(0.5,0.3,'Period='+str(np.round(1*period,3)),fontsize=18)
ax1.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax1.invert_yaxis() #y轴反向

noise = np.random.normal(0.0, 0.06, 200)
noisy = mag + noise


plt.figure(1)
ax2 = plt.gca()
ax2.plot(phase, noisy, '.')
plt.xlabel('phase',fontsize=18)
plt.ylabel('mag',fontsize=18)
#plt.text(0.5,0.3,'Period='+str(np.round(1*period,3)),fontsize=18)
ax2.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax2.invert_yaxis() #y轴反向

s = np.diff(noisy,2).std()/np.sqrt(6)
print(s)
print(np.std(noisy)/s)