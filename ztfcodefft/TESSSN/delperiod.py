# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 15:50:34 2021

@author: dingxu
"""

import pandas as pd

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\Typedata\\noprocess\\'
SAVEPATH = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\Typedata\\process\\'
file = 'SR.csv'

data = pd.read_csv(path+file)

pDATA = data[data['period']<13]

pDATA.to_csv(SAVEPATH+file,index=0) #不保存行索引