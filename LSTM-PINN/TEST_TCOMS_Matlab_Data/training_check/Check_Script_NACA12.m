clear variables;
clc;
xn_h = zeros(2,1);
xn1_h = zeros(2,1);
xn_a = zeros(2,1);
xn1_a = zeros(2,1);
h_int = 0;
alpha_int = 0;
h_dot_int = 0;
alpha_dot_int = 0;
iter = 1000;
F1_Mat = zeros(iter,1);
h = zeros(iter,1);
h_dot_Mat = zeros(iter,1);
alpha = zeros(iter,1);
alpha_dot_Mat = zeros(iter,1);
load 'CL_CM_NACA12_Mach_0.7.mat';

for k = 1:iter
    F1 = 10*cl(k,1);
    F2 = 10*cm(k,1);
    [dh,dalpha,h_dot,alpha_dot,xnp1_h,xnp1_a] = struc_equn_2dof1(F1,F2,xn_h,xn1_h,xn_a,xn1_a);
    xn1_h = xn_h;
    xn1_a = xn_a;
    xn_h = xnp1_h;
    xn_a = xnp1_a;
    if k ==1
        h(k,1) = h_int;
        alpha(k,1) = alpha_int;
        h_dot_Mat(k,1) = h_dot;
        alpha_dot_Mat(k,1) = alpha_dot;
    else
    h(k,1) = dh;
    alpha(k,1) = dalpha;
    h_dot_Mat(k,1) = h_dot;
    alpha_dot_Mat(k,1) = alpha_dot;
    end
end
clearvars -except h alpha  h_dot_Mat alpha_dot_Mat;
t = 0:0.005:999*0.005;
plot(t,alpha,'r');



