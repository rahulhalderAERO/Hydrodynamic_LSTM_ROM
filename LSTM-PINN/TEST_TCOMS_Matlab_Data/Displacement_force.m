load 'Forces.mat';
iter = 1000;
h_Mat = zeros(2,iter);
a_Mat = zeros(2,iter);
s_Mat = zeros(2,iter);

x_np1_h = zeros(2,1);
x_n_h = zeros(2,1);
x_n1_h = zeros(2,1);
x_np1_s = zeros(2,1);
x_n_s = zeros(2,1);
x_n1_s = zeros(2,1);
x_np1_a = zeros(2,1);
x_n_a = zeros(2,1);
x_n1_a = zeros(2,1);
dt = 0.005;
% t = 0:dt:(iter-1)*dt;
A_Mat = [3/(2*dt) -1 ;0  3/(2*dt)];
Fh_div_M = F_a;
Fs_div_M = 2*F_a;
Fa_div_M = 3*F_a;

for i = 1:iter
x_np1_h = A_Mat\((4/(2*dt))*x_n_h -(1/(2*dt))*x_n1_h +[0;Fh_div_M(i)]);
x_n1_h = x_n_h;
x_n_h = x_np1_h;
h_Mat(:,i) = x_np1_h;

x_np1_a = A_Mat\((4/(2*dt))*x_n_a -(1/(2*dt))*x_n1_a + [0;Fa_div_M(i)]);
x_n_a = x_np1_a;
x_n1_a = x_n_a;
a_Mat(:,i) = x_np1_a;

x_np1_a = A_Mat\((4/(2*dt))*x_n_a -(1/(2*dt))*x_n1_a + [0;Fa_div_M(i)]);
x_n_a = x_np1_a;
x_n1_a = x_n_a;
a_Mat(:,i) = x_np1_a;

x_np1_s = A_Mat\((4/(2*dt))*x_n_s -(1/(2*dt))*x_n1_s + [0;Fs_div_M(i)]);
x_n_s = x_np1_s;
x_n1_s = x_n_s;
s_Mat(:,i) = x_np1_s;
end

h_Mat = h_Mat';
a_Mat = a_Mat';
s_Mat = s_Mat';
Fh_div_M = Fh_div_M';
Fa_div_M = Fa_div_M';
Fs_div_M = Fs_div_M';
clearvars -except h_Mat a_Mat s_Mat Fh_div_M Fa_div_M Fs_div_M
plot(h_Mat(:,1));