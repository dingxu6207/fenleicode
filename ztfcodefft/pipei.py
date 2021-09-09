# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 23:20:47 2021

@author: dingxu
"""
from scipy.spatial import cKDTree
import numpy as np

data1 = np.array([2,3])
data2 = np.array([[2,3],[4,5],[7,8]])

kdt = cKDTree(data2)
dist, indices = kdt.query(data1)