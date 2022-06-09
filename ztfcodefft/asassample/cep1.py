# -*- coding: utf-8 -*-
"""
Created on Sun May 29 09:21:01 2022

@author: dingxu
"""

import numpy as np

cepdata = np.loadtxt('nptemp.txt')

ping = np.zeros((1847, 5))
ping[:,4] = 8

ar8 = np.hstack((cepdata, ping))

cep = np.loadtxt('CEP.txt')

cep1 = cep[:5055,:]

savecep = np.vstack((cep1, ar8))

np.savetxt('CEP1.txt', savecep)