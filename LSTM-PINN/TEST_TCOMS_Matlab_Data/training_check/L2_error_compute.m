load 'LSTM_PINN_WO_PINN.mat'
Sum_Num = 0;
Sum_deno = 0;
for i = 1:868
    Sum_Num =  Sum_Num + (Actual_LSTM_PINN(i,7)-Predicted_LSTM(i,7))^2;
    Sum_deno = Sum_deno + (Actual_LSTM_PINN(i,7))^2;
end
L2_Error = (Sum_Num/Sum_deno)*100;