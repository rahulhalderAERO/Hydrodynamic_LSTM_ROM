[n,~] = size(Actual);
num_tot = 0;
den_tot = 0;
for i =1:n
    num_tot = num_tot + (LSTM_PINN_965_of_1000(i,1)-Actual(i,1))^2;
    den_tot = den_tot + (Actual(i,1))^2;
end
mse = (sqrt(num_tot)/sqrt(den_tot))*100;