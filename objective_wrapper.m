function J = objective_wrapper(inputs, orbit_database_in, stabilities_in, s_lg, t_lg, P0, Q, R, mu, opt_flag, solverName, dq)
PENALTY = 1e6;

try
    % unwrap constants
    if isa(orbit_database_in, 'parallel.pool.Constant')
        orbit_database = orbit_database_in.Value;
    else
        orbit_database = orbit_database_in;
    end

    if isa(stabilities_in, 'parallel.pool.Constant')
        stabilities_all = stabilities_in.Value;
    else
        stabilities_all = stabilities_in;
    end

    % unwrap inputs
    if istable(inputs)
        x = table2array(inputs);
    else
        x = inputs;
    end

    x = round(x(:)'); % row, integers

    % orbit/slot pairs
    orbit_indices = x(1:2:end);
    slot_indices  = x(2:2:end);

    % hard clamp indices BEFORE any indexing
    nOrbits = numel(orbit_database);
    if nOrbits < 1
        error('objective_wrapper:EmptyOrbitDB', 'orbit_database is empty.');
    end

    nSlots = size(orbit_database{1}, 1); % assumes consistent
    orbit_indices = max(1, min(orbit_indices, nOrbits));
    slot_indices  = max(1, min(slot_indices,  nSlots));

    % stabilities (now safe)
    stabilities_vec = stabilities_all(orbit_indices);

    % build observer ICs
    num_obs = numel(orbit_indices);
    observer_ICs = zeros(num_obs, 6);
    for k = 1:num_obs
        observer_ICs(k, :) = orbit_database{orbit_indices(k)}(slot_indices(k), :);
    end

    % run EKF
    [s_ekf, cov] = cr3bp_ekf(observer_ICs, s_lg, t_lg, P0, Q, R, mu);

    % compute cost
    Jraw = compute_cost(s_lg, s_ekf, cov, stabilities_vec, opt_flag, solverName, dq);

    % ---------------- guarantee finite output ----------------
    if strcmpi(opt_flag, 'MOO')
        % must be 3x1 (or 1x3) finite vector
        Jraw = Jraw(:);
        if numel(Jraw) ~= 3 || any(~isfinite(Jraw))
            J = [PENALTY; PENALTY; PENALTY];
        else
            J = Jraw;
        end
    else
        % must be finite scalar
        if ~isscalar(Jraw) || ~isfinite(Jraw)
            J = PENALTY;
        else
            J = Jraw;
        end
    end

catch ME
    % send a short diagnostic to the client log 
    try
        if exist('dq','var') && ~isempty(dq)
            send(dq, sprintf('[objective_wrapper] %s: %s', ME.identifier, ME.message));
        end
    catch
    end

    if strcmpi(opt_flag, 'MOO')
        J = [PENALTY; PENALTY; PENALTY];
    else
        J = PENALTY;
    end
end
end