# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:39:20 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import shutil

model = load_model('ztfmodule5.hdf5')#eclipseothers,ztfmodule


path = 'H:\\ZTFDATA\\BYDra\\'  
mypath = []
count = 0
for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-4:] == '.txt'):
           mypath.append(strfile)
           print(strfile)
           count = count+1
           print(count)
           
dirname1 = 'H:\\ZTFDATA\\ztfvarable\\BYDra\\'
lenpath = len(mypath)
testdata = []
for i in range(0, lenpath):
    if i==20000:
        break
    try:
        data = np.loadtxt(mypath[i])
        # print(data.shape)
        hangdata = data[:,0][0:100]
        liedata = data[:,1][0:100]

        nparraydata = np.reshape(liedata,(1,100))
        prenpdata = model.predict(nparraydata)
        
        index = np.argmax(prenpdata[0])
        print(index)
        
        print('it is ok'+str(i))
        if index == 0:
#            if(os.path.exists(dirname1)):
#                os.remove(mypath[i])
#            else:
            shutil.copy(mypath[i], dirname1)
            #os.system('xcopy %s %s /s' % (mypath[i], dirname1))
    except:
        print('it is error!')
    


