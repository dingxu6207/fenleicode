# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:15:55 2022

@author: dingxu
"""

import os
import shutil 
import time

DSTDATA = 'I:\\CG20variable\\King_5\\'
SRDATA = 'I:\\ngcdata\\'


t1 = time.time()
for root, dirs, files in os.walk(DSTDATA):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-4:] == '.csv'):
           print(strfile)
           shutil.copy(strfile, SRDATA)
           
           
print('time=', time.time()-t1) #MCMC运行时间