# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 22:04:14 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

filename = 'TESSVARIABLE.csv'
data = pd.read_csv(filename)
radec = data[['RA','DEC','index','period','prob']]
name = data['name']
npradec = np.array(radec)
radecname = np.array(name)
listname = name.tolist()
npname = np.array(listname)
radecname = np.column_stack((radec, npname))

###################################################################
VSXdata = pd.read_csv('VSX.tsv', sep = ';', encoding='gbk')
VSXdata = VSXdata.drop(labels=0)
VSXdata.index = range(len(VSXdata))
datara = VSXdata['RAJ2000']
datadec = VSXdata['DEJ2000']
#print(VSXdata.iloc[1,1])
 
listdatara = datara.tolist()
listdatadec = datadec.tolist()

radectemp = []
for i in range(0,len(listdatara)):
    RA = np.float32(listdatara[i])
    DEC = np.float32(listdatadec[i])
    
    radectemp.append(RA)
    radectemp.append(DEC)
    
VSXradec = np.float32(radectemp).reshape(-1,2)

kdt = cKDTree(radecname[:,0:2])
dist, indices = kdt.query(VSXradec)

temp = []
for i in range (len(indices)):
    index = indices[i] 
    temp.append(radecname[index])
nptemp = np.array(temp)  

VSXradec = np.column_stack((dist, VSXradec))
allradec = np.column_stack((VSXradec, nptemp))

lista = ['distance','VSXRA', 'VSXDEC', 'TESSRA', 'TESSDEC','TESSindex', 'TESSPeriod','prob', 'TESSname']
dfallradec = pd.DataFrame(allradec, columns= lista)

df4 = [VSXdata, dfallradec]
resultsort = pd.concat(df4, axis=1)
#resultsort = result.sort_values(by='distance')
#resultsort = result.sort_values('cg20name')
print(str(len(resultsort.iloc[0,1]))+resultsort.iloc[0,1])
DataEW = resultsort[resultsort.iloc[:,1] == 'EW'+' '.join(' 'for i in range(14)) +' '] #14=(30-2)/2

DataEW = DataEW[DataEW.iloc[:,7].astype(np.float)<0.01]
resultsort = resultsort[resultsort.iloc[:,7].astype(np.float)<0.0013]
#resultsort.drop_duplicates(subset="cg20name",inplace=True,keep="first")
resultsort.to_csv('allvsx.csv', index=0)
#DataEW.to_csv('EW.csv', index=0)