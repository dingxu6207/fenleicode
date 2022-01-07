# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:07:59 2021

@author: dingxu
"""

import pandas as pd
import os

#path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\ALLDATA\SR\\'
path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\Typedata\\process\\others'
temp = []
for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-4:] == '.csv'):
           data = pd.read_csv(strfile)
           temp.append(data)
           
result = pd.concat(temp, axis=0)

result = result.sort_values(result.columns[3], ascending=False)

result.drop_duplicates(subset="objectname",inplace=True,keep="first")

result.to_csv('others.csv',index=0) 

EAEW = pd.read_csv('EAEW.csv')
others = pd.read_csv('others.csv')

tempdata = [EAEW, others]
resultall = pd.concat(tempdata, axis=0)
resultall.drop_duplicates(subset="objectname",inplace=True,keep="first")
resultall.to_csv('TESSVARIABLE.csv',index=0) 

gri = resultall['index'].value_counts()
print(gri)
