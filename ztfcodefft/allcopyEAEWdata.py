# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:56:32 2022

@author: dingxu
"""
import os
import pandas as pd
import shutil 

DSTDATA = 'J:\\chendata\\EW\\'
SRDATA = 'J:\\TESSDATA\\'

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\ALLDATA\\EW\\'
for root, dirs, files in os.walk(path):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-4:] == '.csv'):
           print(strfile)
           data = pd.read_csv(strfile)
           hang,lie = data.shape
           
           for i in range(0, hang):
               index = data.iloc[i,4]
               section = data.iloc[i,8]
               
               if section < 0.99999999999999999999999999:
                   section = data.iloc[i,9]
                   
               name = data.iloc[i,1]
               
               if index == 3:
                   print(SRDATA+'section'+str(section)+'\\'+name)
                   filename = SRDATA+'section'+str(section)+'\\'+name
                   
                   try:
                       shutil.copy(filename, DSTDATA)
                       print('it is ok!'+str(i))
                   except:
                       continue