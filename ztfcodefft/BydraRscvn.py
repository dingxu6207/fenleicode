# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 21:15:32 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle

PATH = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\'
BYDra = np.loadtxt(PATH+'BYDra.txt')
RSCVN = np.loadtxt(PATH+'RSCVN.txt')
RSCVN[:,54] = 0

dfdata1 = pd.DataFrame(BYDra)
dfdata1 = dfdata1.sample(n=6800)
BYDradata = np.array(dfdata1)

dfdata2 = pd.DataFrame(RSCVN)
dfdata2 = dfdata2.sample(n=6800)
RSCVNdata = np.array(dfdata2)

data1 = np.vstack((BYDradata, RSCVNdata))

np.savetxt('ROT.txt', data1)