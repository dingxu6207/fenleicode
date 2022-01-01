# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 14:19:50 2022

@author: dingxu
"""

"""
    默认的是竖值条形图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 将全局的字体设置为黑体
#matplotlib.rcParams['font.family'] = 'SimHei'
## 数据
#N = 5
#y = [20, 10, 30, 25, 15]
#x = np.arange(N)
## 添加地名坐标
#str1 = ("北京", "上海", "武汉", "深圳", "重庆")
## 绘图 x x轴， height 高度, 默认：color="blue", width=0.8
#p1 = plt.bar(x, height=y, width=0.5, label="城市指标", tick_label=str1)
## 添加数据标签
#for a, b in zip(x, y):
#    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
## 添加图例
#plt.legend()
## 展示图形
#plt.show()

bar_width = 0.3
N = 7
x = [1,4,7,10,13,16,19]
x = np.array(x)
targets = [4236, 1263, 1152, 515, 210, 113, 199]
cha = [3628, 1187, 1037, 443, 177, 71, 123]
cn = [0,1,2,3,4,5,6]
str1 = ('ROT', 'EA', 'EW', 'DSCT', 'RRAB', 'RRC', 'CEP')
plt.figure(1)
plt.bar(x, height=targets, width=1, tick_label=str1)
plt.bar(x+1, height=cha, width=1, align="center")
for a, b in zip(x, targets):
    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
# 添加图例
for a, b in zip(x, cha):
    plt.text(a+1, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)

for i in range(0,7):
    plt.text(x[i]+1, cha[i]+200 , str(np.round(cha[i]/targets[i],3)), c='r')   

plt.xlabel('Type',fontsize=18)
plt.ylabel('Numbers',fontsize=18)