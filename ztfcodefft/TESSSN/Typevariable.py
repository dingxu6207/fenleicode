# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 23:00:50 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

filename = 'allvsx.csv'
data = pd.read_csv(filename)

'''
data1 = data[data['Type']=='RS'+' '.join(' 'for i in range(14)) +' ']
data2 = data[data['Type']=='DCEP'+' '.join(' 'for i in range(13)) +' ']
data3 = data[data['Type']=='CEP'+' '.join(' 'for i in range(13)) +' '+' ']
data4 = data[data['Type']=='DSCTC'+' '.join(' 'for i in range(12)) +' '+' ']
'''

daraROT = data[data['TESSindex']==5]
drpROTtype = daraROT['Type'].str.strip();
daraROT['Type'] = drpROTtype

daraROT = daraROT[daraROT['prob']>0.9]

Vsxperiod = daraROT['Period']

gri = daraROT['Type'].value_counts()
print(gri)

