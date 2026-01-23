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

    % --- PREDICT --- %
    Phi_0 = eye(6);
    s0 = [x_est; Phi_0(:)];

    % propagate state and STM
    [~, s_prop] = ode45(@(t,s) cr3bp_dynamics(t, s,mu), [0 dt], s0);
    
    % extract prediction
    s_final = s_prop(end,:)';
    x_pred = s_final(1:6);
    Phi_k = reshape(s_final(7:42), 6, 6);

    % predict covariance
    P_pred = Phi_k * P_est * Phi_k' + Q;

    % --- UPDATE --- %
    
end