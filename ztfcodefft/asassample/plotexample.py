# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 13:42:14 2022

@author: dingxu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\ALLDATA\SR\\'
path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\example\\EA\\'
temp = []
i = 0
for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-4:] == '.csv'):
           phasemag = np.loadtxt(strfile)
           i = i+1
           plt.plot(phasemag[:,0], phasemag[:,1]+i, c='b')
           plt.text(0.1, np.min(phasemag[:,1])+i-0.08,file[:-4])
plt.xlim(-0.25,1.25)           
plt.xlabel('phase',fontsize=18)
plt.ylabel('mag(offset)',fontsize=18) 
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向           +