#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 02:27:35 2019

@author: rahul
"""
from keras.models import Sequential 
from keras.layers import Dense
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import LSTM
#from keras.layers import Dropout
import pandas as pd

import time
print(time.time())
data = pd.read_csv("train_mod.csv",skiprows = None , header = None )
data1 = pd.read_csv("test_mod.csv",skiprows = None , header = None )

x_train,y_train = [],[]
x_train1,y_train1 = [],[]

data = data.values
for i in range(75,len(data)):
    x_train.append(data[i-75:i,0].reshape(-1,1))
    y_train.append(data[i,1])
#    x_train,y_train = np.array(x_train), np.array(y_train)
x_train , y_train = np.array(x_train) , np.array(y_train)

data1 = data1.values
for i in range(75,len(data1)):
    x_train1.append(data1[i-75:i,0].reshape(-1,1))
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
model.add(LSTM(units =80,return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))    
#model.add(Dropout(0.2))
# Adding the Second LSTM layer
model.add(LSTM(units =80,return_sequences = True ))
##model.add(Dropout(0.2))
#
## Adding the Third LSTM layer
model.add(LSTM(units =80,return_sequences = True ))
#model.add(Dropout(0.2))

# Adding the Fourth LSTM layer
model.add(LSTM(units =80))
#model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1))


#Compile the RNN
model.compile(optimizer = 'adam' , loss = 'mean_squared_error')

history = model.fit(x_train,y_train, epochs = 250, batch_size = 20 )
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

    
  

