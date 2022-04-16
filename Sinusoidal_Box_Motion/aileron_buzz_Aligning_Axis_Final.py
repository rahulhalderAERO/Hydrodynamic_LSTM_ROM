from keras.models import Sequential 
from keras.layers import Dense
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import LSTM
#from keras.layers import Dropout
import pandas as pd
from keras import backend as K
#import tensorflow as tf
###############################################################################
output_data = pd.read_csv("box_training.csv",skiprows = None , header = None )
input_data= pd.read_csv("Height_training_1.csv",skiprows = None , header = None )
output_data1 = pd.read_csv("box_training.csv",skiprows = None , header = None )
input_data1 = pd.read_csv("Height_training_1.csv",skiprows = None , header = None )

test_step = 60
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

for j in range(1,16):
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

for j in range(1,16):
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

Traindata_align = (abs(y_train)).reshape(-1,1)
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
##
model.add(LSTM(units =50,return_sequences = True ))

model.add(LSTM(units =50,return_sequences = True ))

## Adding the Fourth LSTM layer
model.add(LSTM(units = 50))
#model.add(Dropout(0.2))
#
## Adding the output layer
model.add(Dense(units = 3))
#pred = model(x_train)

def PINN_LOSS(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
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
