function [s_ekf, cov] = cr3bp_ekf(observer_indices, state_interp, time_interp, s_lg, t_lg, P0, Q, R, mu)

% Initalize EKF Parameters
num_steps = length(t_lg);
x_est = s_lg(1,1:6);
P_est = P0;

% initialize history
s_ekf= zeros(num_steps, 6);
cov = zeros(num_steps, 6, 6);

s_ekf(1,:) = x_est;
cov(1,:,:) = P_est;

% filter loop
for k=2:num_steps
    dt = t_lg(k) - t_lg(k-1);
    t = t_lg(k);

    % prediction phase 
end