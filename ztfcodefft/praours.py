# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 02:40:07 2022

@author: dingxu
"""

import numpy as np
import pandas as pd

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\starid\\'

IDdata = np.zeros(10,dtype = np.uint)
for i in range(1,47):
    filename = path+str(i)+'.txt'
    data = np.genfromtxt(filename, dtype=str)
    iddata = data[:,1][1:]
    f32id = np.int64(iddata)
    IDdata = np.hstack((f32id, IDdata))

zedata = np.zeros(len(IDdata),dtype = np.uint)
allpindata =  np.vstack((IDdata, zedata)) 
allpindata = allpindata.T

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\'
file = 'savevariableLAST0.csv'
sectiondata = pd.read_csv(path+file)
sectiondata = sectiondata[sectiondata['Type'] == 'EA']
sectiondata = sectiondata[sectiondata['prob']>0.9]
dfname = sectiondata['objectname']

temdata = []
for i in range(0, len(sectiondata)):
    f32data = np.int64(dfname.iloc[i][4:])
    databin = [f32data, 0]
    temdata.append(databin)
    
nptemdata = np.array(temdata)


from scipy.spatial import cKDTree
kdt = cKDTree(allpindata)
dist, indices = kdt.query(nptemdata)
sectiondata['distabce'] = dist 

sectiondata = sectiondata[sectiondata['distabce'] != 0]

TYPE = sectiondata['Type'].value_counts()
print(TYPE)