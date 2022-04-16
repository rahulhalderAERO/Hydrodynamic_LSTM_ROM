# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 05:33:25 2022

@author: rahalder
"""

import tensorflow as tf
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow_probability as tfp

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

x_train_tf=tf.Variable(x_train)  

def get_model():
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
  
  return model

model=get_model()

def interior_loss():
 
  with tf.GradientTape() as tape:
    tape.watch(x_train_tf)
    with tf.GradientTape() as tape2:
      u_predicted=model(x_train_tf)
      grad=tape2.gradient(u_predicted,x_train_tf)
      
  return grad
    
interior_loss = interior_loss()


























