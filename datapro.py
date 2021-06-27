#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 20:39:45 2021

@author: dingxu
"""
import numpy as np
import matplotlib.pyplot as plt
import os

path = 'H:\\ZTFDATA\\'
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
           

lenpath = len(mypath)
testdata = []
for i in range(lenpath):
    #print(mypath[i])
    try:
        data = np.loadtxt(mypath[i])
        # print(data.shape)
        hangdata = data[:,0][0:100]
        liedata = data[:,1][0:100]

        listliedata = list(liedata)
        
        listliedata.append(0)
        listliedata.append(0)
        listliedata.append(0)
        listliedata.append(0)
        listliedata.append(4)
    
        lightydata = np.array(listliedata)
    #print(lightydata.shape)
    
        testdata.append(lightydata)
        
        print('it is ok'+str(i))
    except:
        print('it is error!')
    
lightdata = np.array(testdata)

savedata = np.savetxt('others.txt', lightdata)

