# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:01:53 2022

@author: dingxu
"""

import pandas as pd
import shutil 

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\'
file = 'TESSVARIABLE.csv'


data = pd.read_csv(path+file)

hang,lie = data.shape

index = data.iloc[0,4]
section = data.iloc[0,8]

alldatapath = 'J:\\TESSDATA\\'
dstpath = 'J:\\EADATA\\'
for i in range(0,hang):
    index = data.iloc[i,4]
    section = data.iloc[i,8]
    name = data.iloc[i,1]
    
    
    if index == 2:
        print(alldatapath+'section'+str(section)+'\\'+name)
        filename = alldatapath+'section'+str(section)+'\\'+name
        shutil.copy(filename, dstpath)

    