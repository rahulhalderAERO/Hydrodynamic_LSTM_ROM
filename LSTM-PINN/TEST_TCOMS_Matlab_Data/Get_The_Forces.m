load 'Displacement.mat';

%% Now get the acceleration and forces using finite differencing: 
Vel_x = zeros(2001,1);
Vel_y = zeros(2001,1);
Vel_pitch =  zeros(2001,1);
Fx = zeros(2001,1);
Fy = zeros(2001,1);
T =  zeros(2001,1);
dt = 0.005;
for kk =1:2001
if kk ==1
Vel_x(kk,1) = 0;  
Vel_y(kk,1) = 0;  

elseif kk ==2
Vel_x(kk,1) = (U_x(kk,1)-U_x(kk-1,1))/(dt);
Vel_y(kk,1) = (U_y(kk,1)-U_y(kk-1,1))/(dt);

else
Vel_x(kk,1) = (3*U_x(kk,1)-4*U_x(kk-1,1)+U_x(kk-2,1))/(2*dt);
Vel_y(kk,1) = (3*U_y(kk,1)-4*U_y(kk-1,1)+U_y(kk-2,1))/(2*dt);
% Vel_x(kk,1) = (U_x(kk,1)-U_x(kk-1,1))/(dt);
end
end

for kk =1:2001
if kk ==1
Fx(kk,1) = 0;  
Fy(kk,1) = 0;  
elseif kk ==2
Fx(kk,1) = 30*(Vel_x(kk,1)-Vel_x(kk-1,1))/(dt);
Fy(kk,1) = 30*(Vel_y(kk,1)-Vel_y(kk-1,1))/(dt);
else
Fx(kk,1) = 30*(3*Vel_x(kk,1)-4*Vel_x(kk-1,1)+Vel_x(kk-2,1))/(2*dt);
Fy(kk,1) = 30*(3*Vel_y(kk,1)-4*Vel_y(kk-1,1)+Vel_y(kk-2,1))/(2*dt);
% Fx(kk,1) = (Vel_x(kk,1)-Vel_x(kk-1,1))/(dt);
end
end
t = 0:0.005:2000*0.005;
plot(t,Fx);
