# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:58:20 2022

@author: dingxu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VSXdata = pd.read_csv('VSX.tsv', sep = ';', encoding='gbk')
VSXdata = VSXdata.drop(labels=0)

drpROTtype = VSXdata['Type'].str.strip();
VSXdata['Type'] = drpROTtype

#drpptype = VSXdata['Period'].str.strip();
#VSXdata['Period'] = drpptype

VSXEWdata = VSXdata[VSXdata['Type'] == 'EW']
dataperiod = VSXEWdata['Period']

listperiod = dataperiod.tolist()

periodtemp = []
for i in range(0,len(listperiod)):
    try:
        perioddata = np.float32(listperiod[i])
    except:
        perioddata = 0
    periodtemp.append(perioddata)
    
VSXEWdata['P'] = periodtemp

VSXEWdata = VSXEWdata[VSXEWdata.iloc[:,7]>0]

#plt.hist(np.array(listperiod))
VSXEWdata.to_csv('VSXEW.csv',index=0)


