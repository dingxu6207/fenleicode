# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 20:58:32 2022

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

matrix = np.array([[629, 17,   2,    2,   0,     6,     35,     14,     30],
                   [20,  624,  0,    0,   0,     0,     0,      0,      0],
                   [0,   0,   628,   28,  0,     0,     0,      0,      0],
                   [0,   0,    8,    674, 0,     0,     0,      0,      0],
                   [0,   0,    0,    0,   676,   0,     0,      6,      0],
                   [2,   0,    3,    0,   0,     679,   3,      0,      0],
                   [12,  0,    0,    0,   0,     2,     654,    0,      0],
                   [7,   0,    0,    0,   12,    0,     0,      630,    2],
                   [46,  0,    2,    0,   6,     6,     2,      1,      658]], dtype = np.int64)

plt.figure(0)
plt.imshow(matrix, cmap=plt.cm.cool)


indices = range(len(matrix))
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
#plt.xticks(indices, [0, 1, 2])
#plt.yticks(indices, [0, 1, 2])
plt.xticks(indices, ['ROT','DSCT','EA','EW','MIRA','RRAB','RRC','SR','CEP'])
plt.yticks(indices, ['ROT','DSCT','EA','EW','MIRA','RRAB','RRC','SR','CEP'])

plt.colorbar()

plt.xlabel('predict label',fontsize=20)
plt.ylabel('actual label',fontsize=20)
plt.title('confusion matrix',fontsize=20)


# 显示数据
for first_index in range(len(matrix)):    #第几行
    for second_index in range(len(matrix[first_index])):    #第几列
        plt.text(first_index, second_index, matrix[first_index][second_index])
# 在matlab里面可以对矩阵直接imagesc(confusion)
# 显示
#plt.show()

plt.savefig('matrix.png')

