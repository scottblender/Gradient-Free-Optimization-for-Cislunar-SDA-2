function [s_ekf, cov] = cr3bp_ekf(observer_ICs, s_lg, t_lg, P0, Q, R, mu, LU, ...
                                 sunFcn, sun_min, moon_min, useScreening)

if nargin < 13 || isempty(useScreening)
    useScreening = true;   % default ON
end

num_steps = length(t_lg);
num_obs   = size(observer_ICs, 1);
x_est     = s_lg(1,1:6)';
P_est     = P0;
current_obs_states = observer_ICs;

s_ekf = zeros(num_steps, 6);
cov   = zeros(num_steps, 6, 6);

s_ekf(1,:) = x_est';
cov(1,:,:) = P_est;

options = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);
I6 = eye(6);

for k=2:num_steps
    dt = t_lg(k) - t_lg(k-1);
    t  = t_lg(k);

    % --- PREDICT ---
    Phi_0 = eye(6);
    s0 = [x_est; Phi_0(:)];
    [~, s_prop] = ode45(@(t,s) cr3bp_dynamics(t, s, mu), [0 dt], s0, options);

    s_final = s_prop(end,:)';
    Phi_k   = reshape(s_final(7:42), 6, 6);

    P_pred = Phi_k * P_est * Phi_k' + Q;
    P_upd  = P_pred;
    x_upd  = s_final(1:6);

    next_obs_states = zeros(num_obs, 6);
    for i = 1:num_obs
        s0_obs = current_obs_states(i, :)';
        [~, s_prop_obs] = ode45(@(t,s) cr3bp_dynamics(t, s, mu), [0 dt], s0_obs, options);
        next_obs_states(i, :) = s_prop_obs(end, :);
    end
    current_obs_states = next_obs_states;

    % --- UPDATE ---
    for i = 1:num_obs
        r_obs = current_obs_states(i, 1:3)';

        r_target_truth = s_lg(k,1:3)';
        z_clean = measurement_model(r_target_truth, r_obs);
        noise   = mvnrnd([0;0], R)';     % (assuming R is 2x2 covariance)
        z_meas  = z_clean + noise;

        if useScreening
            r_sun = sunFcn(t_lg(k));

            [occE, occM] = calc_occlusion(r_obs, r_target_truth, mu, LU);
            [ok_excl, ~, ~] = calc_exclusion(r_target_truth, r_obs, r_sun, mu, sun_min, moon_min);

            ok = ok_excl && ~occE && ~occM;
            if ~ok
                continue;
            end
        end

        z_pred  = measurement_model(x_upd(1:3), r_obs);
        y_tilde = z_meas - z_pred;
        y_tilde(1) = atan2(sin(y_tilde(1)), cos(y_tilde(1)));

        H = measurement_jacobian(x_upd(1:3), r_obs);
        S = H * P_upd * H' + R;
        S = (S + S')/2;

        if rcond(S) < 1e-12
            S = S + 1e-12 * eye(size(S));
        end

        [Rchol,p] = chol(S);
        PHt = P_upd * H';

        if p == 0
            K = (PHt / Rchol) / Rchol';
        else
            K = PHt / S;
        end

        x_upd = x_upd + K * y_tilde;
        P_upd = (I6 - K*H) * P_upd * (I6 - K*H)' + K*R*K';
        P_upd = (P_upd + P_upd')/2;
    end

    x_est = x_upd;
    P_est = P_upd;

    s_ekf(k,:) = x_est';
    cov(k,:,:) = P_est;
end
end
