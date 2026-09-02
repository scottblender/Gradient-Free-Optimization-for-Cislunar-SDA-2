function [s_ekf, cov, screeningCount, availableObsCount] = cr3bp_ekf( ...
    observer_ICs, s_target, t_target, P0, Q, R, mu, LU, ...
    sunFcn, sun_exclusion, moon_exclusion, earth_exclusion, useScreening, measCfg)

if nargin < 13 || isempty(useScreening)
    useScreening = true;   % default ON
end

if nargin < 14 || isempty(measCfg)
    measCfg = struct();
    measCfg.type = "ANGLES_ONLY";
end

if ~isfield(measCfg, 'type') || isempty(measCfg.type)
    measCfg.type = "ANGLES_ONLY";
end

measCfg.type = upper(string(measCfg.type));

num_steps = length(t_target);
num_obs   = size(observer_ICs, 1);

% Fixed noise is optional; omitting the seed preserves fresh-noise behavior.
useFixedNoise = isfield(measCfg, 'noiseSeed') && ...
    ~isempty(measCfg.noiseSeed);

if useFixedNoise
    validateattributes(measCfg.noiseSeed, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', '>=', 0, '<=', 2^32-1});

    noiseStream = RandStream('mt19937ar', ...
        'Seed', measCfg.noiseSeed, ...
        'NormalTransform', 'Inversion');

    num_meas = size(R, 1);
    L_noise = chol(R, 'lower');

    % Generate noise for every observer and epoch before screening.
    measurementNoise = L_noise * ...
        randn(noiseStream, num_meas, num_steps*num_obs);

    % Dimensions: measurement component, epoch, observer.
    measurementNoise = reshape( ...
        measurementNoise, num_meas, num_steps, num_obs);
end

x_est     = s_target(1,1:6)';
P_est     = P0;
current_obs_states = observer_ICs;

s_ekf = zeros(num_steps, 6);
cov   = zeros(num_steps, 6, 6);

% NEW
availableObsCount = zeros(num_steps, 1);

s_ekf(1,:) = x_est';
cov(1,:,:) = P_est;

% At initial step, treat all observers as available by default
availableObsCount(1) = num_obs;

options = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);
I6 = eye(6);
screeningCount = 0;

for k = 2:num_steps
    dt = t_target(k) - t_target(k-1);
    t  = t_target(k);

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

    % count available observers this step
    nAvailThisStep = 0;

    % --- UPDATE ---
    for i = 1:num_obs
        r_obs = current_obs_states(i, 1:3)';
        r_target_truth = s_target(k,1:3)';

        r_sun = sunFcn(t);

        ok = calc_visibility( ...
            r_target_truth, r_obs, r_sun, ...
            mu, LU, sun_exclusion, moon_exclusion, earth_exclusion);

        % count valid observers for plotting
        if ok
            nAvailThisStep = nAvailThisStep + 1;
        end

        % always count fails, regardless of useScreening
        if ~ok
            screeningCount = screeningCount + 1;

            % only skip the update if screening is enabled
            if useScreening
                continue;
            end
        end

        % build measurement using selected measurement model
        z_clean = measurement_model(r_target_truth, r_obs, measCfg);
        z_clean = z_clean(:);
        if useFixedNoise
            noise = measurementNoise(:, k, i);
        else
            noise = mvnrnd(zeros(1, size(R,1)), R, 1);
            noise = noise(:);
        end
        z_meas  = z_clean + noise;

        z_pred  = measurement_model(x_upd(1:3), r_obs, measCfg);
        y_tilde = z_meas - z_pred;

        % wrap right ascension residual
        y_tilde(1) = atan2(sin(y_tilde(1)), cos(y_tilde(1)));

        H = measurement_jacobian(x_upd(1:3), r_obs, measCfg);

        S = H * P_upd * H' + R;
        S = (S + S')/2;

        if rcond(S) < 1e-12
            S = S + 1e-12 * eye(size(S));
        end

        [Rchol, p] = chol(S);
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

    % count the number of available observers
    availableObsCount(k) = nAvailThisStep;

    x_est = x_upd;
    P_est = P_upd;

    s_ekf(k,:) = x_est';
    cov(k,:,:) = P_est;
end
end