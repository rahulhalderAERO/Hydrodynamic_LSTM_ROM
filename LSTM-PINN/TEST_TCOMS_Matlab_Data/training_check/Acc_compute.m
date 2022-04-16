load 'h_alpha_dot_t_var.mat';
dt = 0.005;
hdot_time_variation = hdot_time_variation/10;
alphadot_time_variation = alphadot_time_variation/10;
h_ddot = (3*hdot_time_variation(:,1)-4*hdot_time_variation(:,2)+hdot_time_variation(:,3))/(2*dt);
alpha_ddot = (3*alphadot_time_variation(:,1)-4*alphadot_time_variation(:,2)+alphadot_time_variation(:,3))/(2*dt);
clearvars -except h_ddot alpha_ddot