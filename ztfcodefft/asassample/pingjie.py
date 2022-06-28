# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:07:59 2021

@author: dingxu
"""

import pandas as pd
import os

path = 'J:\\TESSDATA\\CSV\\'
temp = []

for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-4:] == '.csv'):
           data = pd.read_csv(strfile)
           if (strfile[-6:-5] == 'v'):
               data['sector'] = strfile[-5:-4]
           else:
               data['sector'] = strfile[-6:-4]
           
           temp.append(data)
           
           
result = pd.concat(temp, axis=0)

result = result.sort_values(result.columns[3], ascending=False)

result.drop_duplicates(subset="objectname",inplace=True,keep="first")

result.to_csv('tessvarible.csv',index=0) 

result = result[result['period']<13]

r1 = result['index'].value_counts()
print(r1)

#
#EAEW = pd.read_csv('EAEW.csv')
#others = pd.read_csv('others.csv')
#
#tempdata = [EAEW, others]
#resultall = pd.concat(tempdata, axis=0)
#resultall.drop_duplicates(subset="objectname",inplace=True,keep="first")
#resultall.to_csv('TESSVARIABLE.csv',index=0) 
#
#gri = resultall['index'].value_counts()
#print(gri)
