# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 17:36:56 2021

@author: dingxu
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\Typedata\\noprocess\\'
SAVEPATH = 'E:\\shunbianyuan\\phometry\\pipelinecode\\fenlei\\fenleicode\\ztfcodefft\\TESSSN\\Typedata\\process\\'
cepfile = 'CEP.csv'
rotfile = 'ROT.csv'
eafile = 'EA.csv'
ewfile = 'EW.csv'
rrcfile = 'RRC.csv'
rrabfile = 'RRAB.csv'
dsctfile = 'DSCT.csv'

rotdata = pd.read_csv(SAVEPATH+rotfile)
rotp = rotdata['prob']

cepdata = pd.read_csv(SAVEPATH+cepfile)
cepp = cepdata['prob']

eapdata = pd.read_csv(SAVEPATH+eafile)
eap = eapdata['prob']

ewpdata = pd.read_csv(SAVEPATH+ewfile)
ewp = ewpdata['prob']

rrcpdata = pd.read_csv(SAVEPATH+rrcfile)
rrcp = rrcpdata['prob']

rrabpdata = pd.read_csv(SAVEPATH+rrabfile)
rrabp = rrabpdata['prob']

dsctpdata = pd.read_csv(SAVEPATH+dsctfile)
dsctp = dsctpdata['prob']

plt.figure(0)
#sns.kdeplot(probilitydata, shade=True, color="g", label="CEP", alpha=.7)
#plt.xlim([0.2, 1])
sns.kdeplot(cepp, label="CEP", alpha=.9)
#sns.kdeplot(rotp, label="ROT", alpha=.7)
sns.kdeplot(eap, label="EA", alpha=.9)
sns.kdeplot(ewp, label="EW", alpha=.9)
sns.kdeplot(rrcp, label="RRC", alpha=.9)
sns.kdeplot(rrabp, label="RRAB", alpha=.9)

#plt.figure(1)
sns.kdeplot(rotp, label="ROT", alpha=.9)
sns.kdeplot(dsctp, label="DSCT", alpha=.9)

plt.xlabel('Classification probability',fontsize=18)
plt.ylabel('Normalised density',fontsize=18)