#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 22:49:34 2020

@author: dingxu
"""


#tensorboard --logdir=./log

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
import os
import shutil
import tensorflow as tf
import imageio
#from keras.optimizers import adam, rmsprop, adadelta

from random import shuffle
from keras.callbacks import TensorBoard

#data = np.loadtxt('savedatasample3.txt')
data = np.loadtxt('savedatasample3.txt')
 
data = data[data[:,100]>20]
#data = data[data[:,101]<1.0]
#data = data[data[:,103]<1.2]
print(len(data))


data[:,100] = data[:,100]
data[:,101] = data[:,101]*100
data[:,102] = data[:,102]*100
data[:,103] = data[:,103]*100

for i in range(len(data)):
    data[i,0:100] = -2.5*np.log10(data[i,0:100]) 
    data[i,0:100] = (data[i,0:100])-np.mean(data[i,0:100])
    
    
shuffle(data)

P = 0.8
duan = int(len(data)*P)

dataX = data[:duan,0:100]
dataY = data[:duan,100]
#dataY[:,0] = dataY[:,0]/90

testX = data[duan:,0:100]
testY = data[duan:,100]
#testY[:,0] = testY[:,0]/90aaa

models = Sequential()
models.add(Dense(500,activation='relu' ,input_dim=100))
models.add(Dense(500, activation='relu'))
models.add(Dense(500, activation='relu'))
models.add(Dense(100, activation='relu'))
models.add(Dense(80, activation='relu'))
models.add(Dense(40, activation='relu'))
models.add(Dense(1))
#models.add(Dense(3,activation='tanh'))

#adamoptimizer = adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00001)
models.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

modeldir = './model/'
logdir = './log/'

if os.path.exists(modeldir):
    shutil.rmtree(modeldir)
os.mkdir(modeldir)

if os.path.exists(logdir):
    shutil.rmtree(logdir)
os.mkdir(logdir)

#gif_images = []
def displayimg(testY, predictY):
    plt.figure(4)
    plt.clf()
    plt.subplot(221)
    plt.plot(testY[:,0], predictY[:,0],'.')
    plt.title('incl')
    n = np.vstack((testY[:,0], predictY[:,0]))
    np.savetxt('incl.txt', n)

    plt.subplot(222)
    plt.plot(testY[:,1], predictY[:,1],'.')
    plt.title('q')
    n = np.vstack((testY[:,1], predictY[:,1]))
    np.savetxt('q.txt', n)

    plt.subplot(223)
    plt.plot(testY[:,2], predictY[:,2],'.')
    plt.title('r')
    n = np.vstack((testY[:,2], predictY[:,2]))
    np.savetxt('r.txt', n)


    plt.subplot(224)
    plt.plot(testY[:,3], predictY[:,3],'.')
    plt.title('divT')
    n = np.vstack((testY[:,3], predictY[:,3]))
    np.savetxt('divT.txt', n)

    plt.pause(0.1)
    plt.savefig('tu.jpg')

    #gif_images.append(imageio.imread('tu.jpg'))

def displayoneimg(testY, predictY):
    plt.figure(4)
    plt.clf()
    plt.plot(testY, predictY,'.')
    plt.title('q', fontsize = 20)

    plt.pause(0.1)
    plt.savefig('tu.jpg')
    
    #gif_images.append(imageio.imread('tu.jpg'))

class PredictionCallback(tf.keras.callbacks.Callback):    
    def on_epoch_end(self, epoch, logs={}):
        #print(self.validation_data[0])
        if (epoch%1 == 0):
            
            y_pred = self.model.predict(testX)
            
            displayoneimg(testY, y_pred)
        #print('prediction: {} at epoch: {}'.format(y_pred, epoch))
            #plt.figure(5)        
            #plt.clf()
            #plt.plot(testY[:,0], y_pred[:,0],'.')
            #plt.pause(0.1)
            #plt.ioff()
    
    
# checkpoint
filepath = modeldir+'weights-improvement-{epoch:05d}-{val_loss:.4f}.hdf5'
# 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,mode='min')
#callbacks_list = [checkpoint]
tensorboard = [TensorBoard(log_dir=logdir)]
callback_lists = [tensorboard, checkpoint, PredictionCallback()]
history = models.fit(dataX, dataY, epochs=20000, batch_size=300, validation_data=(testX, testY),shuffle=True,callbacks=callback_lists)

predictY = models.predict(testX)
predictdatay = models.predict(dataX)

score = models.evaluate(dataX, dataY, batch_size=10)

models.save('phmodsample2.h5')
print(score)



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




#imageio.mimsave('test.gif', gif_images, fps =10)
'''
plt.figure(4)
plt.plot(dataY[:,0], predictdatay[:,0],'.')
plt.title('incly')
n = np.vstack((dataY[:,0], predictdatay[:,0]))
np.savetxt('incly.txt', n)

plt.figure(5)
plt.plot(dataY[:,1], predictdatay[:,1],'.')
plt.title('qy')
n = np.vstack((dataY[:,1], predictdatay[:,1]))
np.savetxt('qy.txt', n)

plt.figure(6)
plt.plot(dataY[:,2], predictdatay[:,2],'.')
plt.title('ry')
n = np.vstack((dataY[:,2], predictdatay[:,2]))
np.savetxt('ry.txt', n)


plt.figure(7)
plt.plot(dataY[:,3], predictdatay[:,3],'.')
plt.title('divTy')
n = np.vstack((dataY[:,3], predictdatay[:,3]))
np.savetxt('divTy.txt', n)
'''