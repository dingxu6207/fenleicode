# -*- coding: utf-8 -*-
"""
Created on Sun May 29 09:21:01 2022

@author: dingxu
"""

import numpy as np

cepdata = np.loadtxt('npEA.txt')

ping = np.zeros((14618, 5))
ping[:,4] = 2

ar8 = np.hstack((cepdata, ping))

#cep = np.loadtxt('CEP.txt')
#
#cep1 = cep[:5055,:]
#
#savecep = np.vstack((cep1, ar8))

np.savetxt('EA1.txt', ar8)