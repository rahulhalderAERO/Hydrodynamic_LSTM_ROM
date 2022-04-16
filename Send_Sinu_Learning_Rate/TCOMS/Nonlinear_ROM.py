#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 02:27:35 2019

@author: rahul
"""
from keras.models import Sequential 
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras import regularizers
import pandas as pd

import time
print(time.time())
data = pd.read_csv("train_1500_mod.csv",skiprows = None , header = None )
data1 = pd.read_csv("train_2000_mod.csv",skiprows = None , header = None )

x_train,y_train = [],[]
x_train1,y_train1 = [],[]

step = 75
data = data.values
for i in range(step,len(data)):
    x_train.append(data[i-step:i,0].reshape(-1,1))
    y_train.append(data[i,1])
#    x_train,y_train = np.array(x_train), np.array(y_train)
x_train , y_train = np.array(x_train) , np.array(y_train)

data1 = data1.values
for i in range(step,len(data1)):
    x_train1.append(data1[i-step:i,0].reshape(-1,1))
    y_train1.append(data1[i,1])
#    x_train,y_train = np.array(x_train), np.array(y_train)
x_train1 , y_train1 = np.array(x_train1) , np.array(y_train1)

###############################################################################

def max( data_Rahul ):
	max_value = data_Rahul[0]
	for x in data_Rahul:
		if max_value < x:
			max_value = x
	return max_value

def scheduler(epoch, lr):
    exp = np.floor((1 + epoch) / 25)
    alpha = 0.001 * (0.5 ** exp)
    return float(alpha)

###############################################################################

Traindata_align = (abs(y_train)).reshape(-1,1)
max_train = max(Traindata_align)
y_train_original = y_train
y_train = (1/max_train)*y_train
#y_train = (1/max_train)*y_train1

###############################################################################

#create model
model = Sequential()

#Initialise the RNN
#regressor = Sequential()

# Adding the first LSTM layer
model.add(LSTM(units =50, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))    
#model.add(Dropout(0.1))
# Adding the Second LSTM layer
model.add(LSTM(units =50,return_sequences = True))
#model.add(Dropout(0.1))
#
## Adding the Third LSTM layer
model.add(LSTM(units =50,return_sequences = True))
#odel.add(Dropout(0.1))

# Adding the Fourth LSTM layer
model.add(LSTM(units =50,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5)))
#model.add(Dropout(0.1))

# Adding the output layer
model.add(Dense(units = 1,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4),activity_regularizer=regularizers.l2(1e-5)))


#Compile the RNN
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer = opt, loss = 'mean_squared_error')

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(x_train,y_train, epochs = 200, callbacks=[callback], batch_size = 10 )
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#
plt.plot(np.arange(0,len(y_train1)),y_train1,label='Actual')
predicted_y = max_train*(model.predict(x_train1))
plt.plot(np.arange(0,len(y_train1)),predicted_y , label ='Predicted')
#plt.ylabel('Mode1')
#plt.xlabel('Time step')
print(time.time())

    
  

