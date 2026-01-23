function [s_ekf, cov] = cr3bp_ekf(observer_indices, state_interp, time_interp, s_lg, t_lg, P0, Q, R, mu)

% Initalize EKF Parameters
num_steps = length(t_lg);
x_est = s_lg(1,1:6)';
P_est = P0;

% initialize history
s_ekf= zeros(num_steps, 6);
cov = zeros(num_steps, 6, 6);

s_ekf(1,:) = x_est';
cov(1,:,:) = P_est;

% ode45 options
options = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);

% filter loop
for k=2:num_steps
    dt = t_lg(k) - t_lg(k-1);
    t = t_lg(k);

    % --- PREDICT --- %
    Phi_0 = eye(6);
    s0 = [x_est; Phi_0(:)];

    % propagate state and STM
    [~, s_prop] = ode45(@(t,s) cr3bp_dynamics(t, s,mu), [0 dt], s0, options);
    
    % extract prediction
    s_final = s_prop(end,:)';
    Phi_k = reshape(s_final(7:42), 6, 6);

    % predict covariance
    P_pred = Phi_k * P_est * Phi_k' + Q;
    P_upd = P_pred;
    x_upd = s_final(1:6); % Initialize updated state estimate
    % --- UPDATE --- %
    for i = 1:length(observer_indices)

        % get observer state through interpolation
        obs_idx = observer_indices(i);
        t_ref = time_interp{obs_idx};
        period = t_ref(end);           
        t_local = mod(t, period);
        
        s_ref = state_interp{obs_idx}; %
        r_obs_full = interp1(t_ref, s_ref, t_local, 'linear', 'extrap');
        r_obs = r_obs_full(1:3)';

        % generate noisy measurements
        r_target_truth = s_lg(k,1:3)';
        z_clean = measurement_model(r_target_truth, r_obs);
        noise = mvnrnd([0;0], R)';
        z_meas = z_clean + noise;

        % update state estimate
        z_pred = measurement_model(x_upd(1:3), r_obs);
        y_tilde = z_meas - z_pred; % measurement residual
        y_tilde(1) = atan2(sin(y_tilde(1)),cos(y_tilde(1))); % wrap right ascension
        H = measurement_jacobian(x_upd(1:3), r_obs); % measurement jacobian
        S = H * P_upd * H' + R; % innovation covariance
        K = P_upd * H' / S; % Kalman gain
        x_upd = x_upd + K * y_tilde; % updated state estimate
        P_upd = (eye(6) - K * H) * P_upd; % updated covariance
        P_upd = (P_upd + P_upd')/2; % symmetrize 
    end
    % update state and covariance estimates
    x_est = x_upd; 
    P_est = P_upd;

    % store results
    s_ekf(k,:) = x_est';
    cov(k,:,:) = P_est;
end
end