# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:50:42 2022

@author: dingxu
"""
import pandas as pd
import numpy as np


file = 'tessvarible.csv'
data = pd.read_csv(file)



data = data[data['index'] != 4]
data = data[data['index'] != 7]

data = data[data['prob']>0.6]

r1 = data['index'].value_counts()
print(r1)


data.to_csv('retessvarible.csv',index=0) 


print(data.iloc[1,11])

data['sector'] = data['sector'].astype('object')

data.at[1,'sector'] = [89,100]