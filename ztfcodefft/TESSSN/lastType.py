# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 09:39:56 2022

@author: dingxu
"""

import pandas as pd

file = 'TESSVARIABLE.csv'

data = pd.read_csv(file)
hang,lie = data.shape

gri = data['index'].value_counts()
print(gri)

temp = []
for i in range(0, hang):
    index = data.iloc[i, 4]
    print(index)
    if index == 0:
        temp.append('ROT')
        
    if index == 1:
        temp.append('DSCT')
        
    if index == 2:
        temp.append('EA')
        
    if index == 3:
        temp.append('EW')
        
    if index == 4:
        temp.append('MIRA')
    
    if index == 5:
        temp.append('RRAB')
    
    if index == 6:
        temp.append('RRC')
        
    if index == 7:
        temp.append('SR')
        
    if index == 8:
        temp.append('CEP')
    
data['Type'] = temp

savedata = data[['objectname', 'RA', 'DEC', 'prob', 'period', 'Type']]
savedata.to_csv('savevariable.csv',index=0)

sectiondata = pd.read_csv('savevariable.csv')
EAEW = pd.read_csv('EAEW.csv')

#sectiondata = sectiondata[sectiondata['prob']>0.9]
gri = sectiondata['Type'].value_counts()
print(gri)
