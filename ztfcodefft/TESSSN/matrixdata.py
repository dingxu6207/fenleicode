#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 09:04:23 2021

@author: dingxu
"""
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


testxydata = np.loadtxt('testxydata.txt')
testxdata = testxydata[:,0:10]

testydata = testxydata[:,54]

path = './model/'
file = 'weights-improvement-00204-0.9616.hdf5'
models = load_model(path+file)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#testxdata = np.reshape(testxdata, [len(testxdata),50,1])
#predicted = models.predict_classes(testxdata)
predict = models.predict(testxdata)
predicted = np.argmax(predict,axis=1)

#matrix = confusion_matrix(testydata, predicted)
matrix = confusion_matrix(predicted, testydata)
#report = classification_report(testydata,predicted,output_dict=True)
report = classification_report(testydata,predicted)
print(matrix)
print(report)

# df1 = pd.DataFrame(report).transpose()
# df1.to_csv('report.csv', index=True)

# for i in range(0,9):
#     for j in range(0,9):
#       matrix[i,j] = matrix[i,j]/np.sum(matrix[:,j])

plt.figure(2)
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
plt.show()

plt.savefig('matrix.png')