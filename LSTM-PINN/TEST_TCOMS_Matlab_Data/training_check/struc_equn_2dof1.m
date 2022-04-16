function [dh,dalpha,h_dot,alpha_dot,xnp1_h,xnp1_a] = struc_equn_2dof1(F1,F2,xn_h,xn1_h,xn_a,xn1_a)
f = zeros(2,1);
detA = zeros(2,1);
rhs_a = zeros(2,1);
rhs_h = zeros(2,1);
eta = zeros(2,1);
eta_dot = zeros(2,1);

% %%%%%%%% Force decleration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f(1) = (F1);
f(2) = (F2);
f_tilde = f;
dt_dim = 0.005;
dt = dt_dim;
detA(1) = (9/(4*dt^2));
detA(2) = (9/(4*dt^2));
A_inv_h = (1/detA(1)).*[(3/(2*dt)) 1 ; 0 (3/(2*dt))];
A_inv_a = (1/detA(2)).*[(3/(2*dt)) 1 ; 0 (3/(2*dt))];
%%%%%%%%%First Mode Computation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s_h = (-4*xn_h + xn1_h)./(2*dt);
rhs_h(1) = -s_h(1);
rhs_h(2) = f_tilde(1)- s_h(2);
xnp1_h = A_inv_h*rhs_h;
%%%%%%%%%Second Mode Computation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s_a = (-4*xn_a + xn1_a)./(2*dt);
rhs_a(1) = -s_a(1);
rhs_a(2) = f_tilde(2)- s_a(2);
xnp1_a = A_inv_a*rhs_a;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% eta(1) = xnp1_h(1)- xnp1old_h(1);
eta(1) = xnp1_h(1);
eta_dot(1) = xnp1_h(2); 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% eta(2) = xnp1_a(1) - xnp1old_a(1);
eta(2) = xnp1_a(1);
eta_dot(2)=  xnp1_a(2); 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
q_aero = eta;
qdot_aero = eta_dot;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dh =     q_aero(1);
dalpha = q_aero(2);
h_dot =  qdot_aero(1);
alpha_dot = qdot_aero(2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




