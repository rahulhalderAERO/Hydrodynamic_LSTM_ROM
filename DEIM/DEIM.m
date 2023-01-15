clc;
clear variables;
%% Get the DEIM codes with the period of 1.2 
load 'Wave_Elv_t102.mat';

%% Substract the initial height distribution
Del_Elv =  Wave_Elv_t102 - ones(2001,1)*(Wave_Elv_t102(1,:));
Del_Elv = Del_Elv';
Data = Del_Elv;
[s1,s2] = size(Data);
nxf = s1;

%% Apply SVD

[phin,Svn,Vn] = svd(Data);
svn_max = max(diag(Svn));
Svn_normalized = diag(Svn)./svn_max;
%% Number of DEIM modes = 30
count = 30;

%% DEIM algorithm 

%Chaturantabut, S. and Sorensen, D.C.,” Nonlinear Reduced Order Modelling 
%Via Discrete Empirical Interpolation”.

phin = phin(:,1:count);
IM = eye(nxf,nxf);
count_d = zeros(count,1);
r = zeros(nxf,1);
P = zeros(nxf,1);
Psi_diem = phin;
U_psi = zeros(nxf,1);
k =0;
j =1;
for i = 1:nxf
r(i) = Psi_diem(i,1);
if abs(r(i)) > k
    k=abs(r(i));
    count_d(j) = i;
end
end
U_psi(:,1)=Psi_diem(:,1);
P(:,1)=IM(:,count_d(j));
for j = 2:count
k=0;
 A_psi = transpose(P)*U_psi;
 B_psi =  transpose(P)*Psi_diem(:,j);
 c = A_psi\B_psi;
 r = abs(Psi_diem(:,j)-U_psi*c);
 for i =1:nxf
 if abs(r(i)) > k
    k=abs(r(i));
    count_d(j) =i;
end
end
U_psi(:,1:j)=Psi_diem(:,1:j);
P(:,1:j) = [P(:,1:j-1) IM(:,count_d(j))]; 
j
end

Mat =inv(transpose(P)*U_psi);
DEIM_Modes = U_psi*(Mat);
count_d_Test = count_d;
Data_ML= Data(count_d_Test,:);

clearvars -except DEIM_Modes count_d_Test Data_ML Del_Elv

% count_d_Test is the DEIM control point 
% Data_ML is the redued matrix which will be used for the ML application
% DEIM_Modes are DEIM modes used for the reconstruction from the reduced
% state.

%% Compare reconstructed vs actual data at 250th time step

Test = DEIM_Modes*Data_ML;
plot(Test(:,250));
hold on;
plot(Del_Elv(:,250));













