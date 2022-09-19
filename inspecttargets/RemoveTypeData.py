# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:41:48 2022

@author: dingxu
"""

import os
import pandas as pd
import shutil 

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\ALLCODE\\classify\\'
file = 'tessvarible.csv'


DSTDSCT = 'J:\\TESSDATAType\\DSCT\\'
DSTEA = 'J:\\TESSDATAType\\EA\\'
DSTEW = 'J:\\TESSDATAType\\EW\\'
DSTRRAB = 'J:\\TESSDATAType\\RRAB\\'
DSTRRC = 'J:\\TESSDATAType\\RRC\\'
DSTCEP = 'J:\\TESSDATAType\\CEP\\'
DSTROT = 'J:\\TESSDATAType\\ROT\\'

SRDATA = 'J:\\TESSDATA\\section'

data = pd.read_csv(path+file)

hang,lie = data.shape
for i in range(0, hang):
    name = data.iloc[i, 1]
    sector = data.iloc[i, 11]
    index = data.iloc[i, 4]
        
    filename = SRDATA+str(sector)+'\\'+name
    print(filename)
    
    if index == 0:
        shutil.copy(filename, DSTROT)
    
#    if index == 1:
#        shutil.copy(filename, DSTDSCT)
#        
#    if index == 2:
#        shutil.copy(filename, DSTEA)
#        
#    if index == 3:
#        shutil.copy(filename, DSTEW)
#        
#    if index == 5:
#        shutil.copy(filename, DSTRRAB)
#    
#    if index == 6:
#        shutil.copy(filename, DSTRRC)
#        
#    if index == 8:
#        shutil.copy(filename, DSTCEP)