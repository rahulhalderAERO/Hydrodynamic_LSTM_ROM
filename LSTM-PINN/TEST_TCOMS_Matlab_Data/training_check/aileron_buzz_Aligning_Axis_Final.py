from keras.models import Sequential 
from keras.layers import Dense
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import LSTM
#from keras.layers import Dropout
import pandas as pd
from keras import backend as K
import tensorflow as tf
import random
#import tensorflow as tf
###############################################################################
output_data = pd.read_csv("Udot_Data_extended.csv",skiprows = None , header = None )
input_data= pd.read_csv("F_Data.csv",skiprows = None , header = None )
output_data1 = pd.read_csv("Udot_Data_extended.csv",skiprows = None , header = None )
input_data1 = pd.read_csv("F_Data.csv",skiprows = None , header = None )

test_step = 25
x_train = []
y_train = []
x_train1 = []
y_train1 = []
x_train_local = []

output_data = output_data.values
input_data = input_data.values
output_data1 = output_data1.values
input_data1 = input_data1.values
###############################################################################
def max( data_Rahul ):
	max_value = data_Rahul[0]
	for x in data_Rahul:
		if max_value < x:
			max_value = x
	return max_value
#############################################################################
for i in range(test_step,input_data.shape[0]):
    x_train.append(input_data[i-test_step:i,0].reshape(-1,1))
x_train_mid = np.array(x_train)
x_train = []

for j in range(1,2):
    for i in range(test_step,input_data.shape[0]):
        x_train.append(input_data[i-test_step:i,j].reshape(-1,1))
    x_train_mid1 = np.array(x_train)
    x_train = []
    x_train_final = np.concatenate((x_train_mid,x_train_mid1),axis =2) 
    x_train_mid = x_train_final
    
x_train = x_train_final
###############################################################################
for i in range(test_step,input_data1.shape[0]):
    x_train1.append(input_data1[i-test_step:i,0].reshape(-1,1))
x_train1_mid = np.array(x_train1)
x_train1 = []

for j in range(1,2):
    for i in range(test_step,input_data1.shape[0]):
        x_train1.append(input_data1[i-test_step:i,j].reshape(-1,1))
    x_train1_mid1 = np.array(x_train1)
    x_train1 = []
    x_train1_final = np.concatenate((x_train1_mid,x_train1_mid1),axis =2) 
    x_train1_mid = x_train1_final
    
x_train1 = x_train1_final
###############################################################################
y_train = output_data[test_step:len(output_data)]
y_train1 = output_data1[test_step:len(output_data1)]
###############################################################################


list=[]
for i in range(10):
    r=random.randint(1,y_train.shape[0]-1)
    if r not in list: list.append(r)
    list_array = np.array(list)

for k in range(len(list_array)):
    val_row = list_array[k]
    y_train[val_row,0:5] = 0

Traindata_align = (abs(y_train[:,:])).reshape(-1,1)
max_train = max(Traindata_align)
y_train_original = y_train
y_train = (1/max_train)*y_train

###############################################################################
model = Sequential()

#Initialise the RNN
#regressor = Sequential()

# Adding the first LSTM layer
model.add(LSTM(units = 50,return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))    
#model.add(Dropout(0.2))
#Adding the Second LSTM layer
model.add(LSTM(units =50,return_sequences = True ))
#model.add(Dropout(0.2))
#
## Adding the Third LSTM layer
model.add(LSTM(units =50,return_sequences = True ))
#model.add(Dropout(0.2))
## Adding the Fourth LSTM layer
model.add(LSTM(units = 50))
#model.add(Dropout(0.2))
## Adding the output layer
model.add(Dense(units = 8))
#pred = model(x_train)
dt = 0.005

def PINN_LOSS(y_true, y_pred):
    
    def DIFF_FILTERED(x, y):
        y_true_new = x+0.00001
        y_true_div = tf.math.divide(y, y_true_new)
        difference = 1-y_true_div
        y_true_mult = tf.math.multiply(x, difference)
        return y_true_mult
    
    
    
    loss1 = (K.square(DIFF_FILTERED(y_true[:,0], y_pred[:,0]))) 
    loss2 = (K.square(DIFF_FILTERED(y_true[:,1], y_pred[:,1])))
    loss3 = (K.square(DIFF_FILTERED(y_true[:,2], y_pred[:,2])))
    loss4 = (K.square(DIFF_FILTERED(y_true[:,3], y_pred[:,3]))) 
    loss5 = (K.square(DIFF_FILTERED(y_true[:,4], y_pred[:,4])))
    loss6 = (K.square(DIFF_FILTERED(y_true[:,5], y_pred[:,5])))
    h_double_dot = (3*y_pred[:,0]-4*y_pred[:,1]+y_pred[:,2])
    a_double_dot = (3*y_pred[:,3]-4*y_pred[:,4]+y_pred[:,5])
    loss7 = (K.square(h_double_dot-(2*dt)*y_pred[:,6]))
    loss8 = (K.square(a_double_dot-(2*dt)*y_pred[:,7]))

    loss_mid = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8
    
    loss = K.mean((loss_mid),axis=-1)
    return loss
#
##Compile the RNN
model.compile(optimizer = 'adam' , loss = PINN_LOSS)
#
history = model.fit(x_train,y_train, epochs = 100 , batch_size = 10)
#predicted_y = model.predict(x_train1)
#predicted_y = max_train*predicted_y
predicted_y = max_train*(model.predict(x_train1))

plt.plot(np.arange(0,len(y_train1)),y_train1[:,0],label='Actual')
plt.plot(np.arange(0,len(y_train1)),predicted_y[:,0] , label ='Predicted')