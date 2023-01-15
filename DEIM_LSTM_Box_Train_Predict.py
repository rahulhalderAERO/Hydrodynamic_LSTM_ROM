# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 00:14:04 2022

@author: rahalder
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.models import load_model 
import time
from scipy.io import savemat


start_time = time.time()

###############################################################################
output_data = pd.read_csv("box_training_new_T102.csv",skiprows = None , header = None )
input_data= pd.read_csv("Height_training_New_T102_1.csv",skiprows = None , header = None )
output_data1 = pd.read_csv("box_training_new_T103.csv",skiprows = None , header = None )
input_data1 = pd.read_csv("Height_training_New_T103_1.csv",skiprows = None , header = None )

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

output_scaler = MinMaxScaler(feature_range=(0, 1))
output_data_Scaled = output_scaler.fit_transform(output_data)
output_data_Scaled1 = output_scaler.fit_transform(output_data1)

###############################################################################

for i in range(test_step,input_data.shape[0]):
    x_train.append(input_data[i-test_step:i,0].reshape(-1,1))
x_train_mid = np.array(x_train)
x_train = []

for j in range(1,31):
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

for j in range(1,31):
    for i in range(test_step,input_data1.shape[0]):
        x_train1.append(input_data1[i-test_step:i,j].reshape(-1,1))
    x_train1_mid1 = np.array(x_train1)
    x_train1 = []
    x_train1_final = np.concatenate((x_train1_mid,x_train1_mid1),axis =2) 
    x_train1_mid = x_train1_final
    
x_train1 = x_train1_final
###############################################################################
y_train = output_data_Scaled[test_step:len(output_data_Scaled)]
y_train1 = output_data_Scaled1[test_step:len(output_data_Scaled1)]
###############################################################################


Model = load_model("Teststep_25epochs_250variable_units_100_time_1637.1116094589233.h5")

predicted_y = (Model.predict(x_train1))

y_pred_trainset_inv = output_scaler.inverse_transform(predicted_y)


# plt.plot(np.arange(0,len(y_train1)),output_data1[test_step:,0],label='Actual')
# plt.plot(np.arange(0,len(y_train1)),y_pred_trainset_inv[:,0] , label ='Predicted')

database = {"predict": y_pred_trainset_inv, "Actual": output_data1}
savemat('Teststep_25epochs_250variable_units_100_time_1637.1116094589233.mat',database,appendmat="False")
