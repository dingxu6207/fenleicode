# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 20:39:32 2022

@author: dingxu
"""

import pandas as pd

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\Typedata\\process\\EAEW\\'
EA = 'EA.csv'
EW = 'EW.csv'

EADATA = pd.read_csv(path+EA)
EWDATA = pd.read_csv(path+EW)

EAEW = pd.concat([EADATA, EWDATA], axis=0)

EAEW = EAEW.sort_values(EAEW.columns[3], ascending=False)

EAEW.drop_duplicates(subset="objectname",inplace=True,keep="first")

EAEW.to_csv('EAEW.csv',index=0) 

gri = EAEW['index'].value_counts()
print(gri)