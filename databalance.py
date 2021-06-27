# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 22:11:47 2021

@author: dingxu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle
from keras.utils import np_utils
from keras.layers import Dense, LSTM, Activation
from keras.models import Sequential

data = np.loadtxt('alldara.txt')

    
    
shuffle(data)


P = 0.8
duan = int(len(data)*P)

dataX = data[:duan,0:100]
dataY = data[:duan,104]
dataY = np_utils.to_categorical(dataY)
#dataY[:,0] = dataY[:,0]/90

testX = data[duan:,0:100]
testY = data[duan:,104]
testY = np_utils.to_categorical(testY)

models = Sequential()
models.add(Dense(500,activation='relu' ,input_dim=100))
models.add(Dense(500, activation='relu'))
models.add(Dense(500, activation='relu'))
models.add(Dense(100, activation='relu'))
models.add(Dense(80, activation='relu'))
models.add(Dense(40, activation='relu'))
models.add(Dense(5, activation='softmax'))

models.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = models.fit(dataX, dataY, batch_size=10, epochs=100, verbose=1, validation_data=(testX, testY))   #fit将模型训练epochs轮

models.save('phmodsample2.h5')

plt.figure(6)
history_dict=history.history
loss_value=history_dict['loss']
val_loss_value=history_dict['val_loss']
epochs=range(1,len(loss_value)+1)
plt.plot(epochs,loss_value,'r',label='Training loss')
plt.plot(epochs,val_loss_value,'b',label='Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

#for i in range(0,hang):
#    plt.figure(0)
#    plt.plot(data[i,0:100], '.')
#    plt.pause(1)
#    plt.clf()
#    
#    plt.figure(1)
#    plt.plot(semicontact[i,0:100], '.')
#    plt.pause(1)
#    plt.clf()

