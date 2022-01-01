# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 14:30:59 2021

@author: dingxu
"""

import pandas as pd

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\CSVDATA\\'

indexcount = '43'
file = 'testcsv' + indexcount +'.csv'
SAVEPATH = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\ALLDATA\\'

data = pd.read_csv(path+file)

dataROT = data[data['index']==0]
dataROT['section'] = [int(indexcount) for i in range(len(dataROT))]
dataROT = dataROT.sort_values(dataROT.columns[3])
dataROT.to_csv(SAVEPATH+'ROT\\dataROT'+indexcount+'.csv',index=0) #不保存行索引

dataDSCT = data[data['index']==1]
dataDSCT['section'] = [int(indexcount) for i in range(len(dataDSCT))]
dataDSCT = dataDSCT.sort_values(dataDSCT.columns[3])
dataDSCT.to_csv(SAVEPATH+'DSCT\\dataDSCT'+indexcount+'.csv',index=0) #不保存行索引

dataEA = data[data['index']==2]
dataEA['section'] = [int(indexcount) for i in range(len(dataEA))]
dataEA = dataEA.sort_values(dataEA.columns[3])
dataEA.to_csv(SAVEPATH+'EA\\dataEA'+indexcount+'.csv',index=0) #不保存行索引

dataEW = data[data['index']==3]
dataEW['section'] = [int(indexcount) for i in range(len(dataEW))]
dataEW = dataEW.sort_values(dataEW.columns[3])
dataEW.to_csv(SAVEPATH+'EW\\dataEW'+indexcount+'.csv',index=0) #不保存行索引

dataMIRA = data[data['index']==4]
dataMIRA['section'] = [int(indexcount) for i in range(len(dataMIRA))]
dataMIRA = dataMIRA.sort_values(dataMIRA.columns[3])
dataMIRA.to_csv(SAVEPATH+'MIRA\\dataMIRA'+indexcount+'.csv',index=0) #不保存行索引

dataRRAB = data[data['index']==5]
dataRRAB['section'] = [int(indexcount) for i in range(len(dataRRAB))]
dataRRAB = dataRRAB.sort_values(dataRRAB.columns[3])
dataRRAB.to_csv(SAVEPATH+'RRAB\\dataRRAB'+indexcount+'.csv',index=0) #不保存行索引

dataRRC = data[data['index']==6]
dataRRC['section'] = [int(indexcount) for i in range(len(dataRRC))]
dataRRC = dataRRC.sort_values(dataRRC.columns[3])
dataRRC.to_csv(SAVEPATH+'RRC\\dataRRC'+indexcount+'.csv',index=0) #不保存行索引

dataSR = data[data['index']==7]
dataSR['section'] = [int(indexcount) for i in range(len(dataSR))]
dataSR = dataSR.sort_values(dataSR.columns[3])
dataSR.to_csv(SAVEPATH+'SR\\dataSR'+indexcount+'.csv',index=0) #不保存行索引

dataCEP = data[data['index']==8]
dataCEP['section'] = [int(indexcount) for i in range(len(dataCEP))]
dataCEP = dataCEP.sort_values(dataCEP.columns[3])
dataCEP.to_csv(SAVEPATH+'CEP\\dataCEP'+indexcount+'.csv',index=0) #不保存行索引