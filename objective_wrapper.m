function J_total = objective_wrapper(inputs, orbit_database_in, stabilities_in, ...
    s_lg, t_lg, P0, Q, R, mu, LU, sunFcn, sun_min, moon_min, ...
    opt_flag, solverName, dq, useScreening, costFlags)

    % ---------------- defaults ----------------
    if nargin < 17 || isempty(useScreening)
        useScreening = true;
    end

    % Cost component toggles (default: all ON)
    if nargin < 18 || isempty(costFlags)
        costFlags = struct('J1', true, 'J2', true, 'J3', true);
    end

    % Make robust to missing fields
    if ~isfield(costFlags,'J1'), costFlags.J1 = true; end
    if ~isfield(costFlags,'J2'), costFlags.J2 = true; end
    if ~isfield(costFlags,'J3'), costFlags.J3 = true; end

    % Predefine for logging
    J_1 = NaN; J_2 = NaN; J_3 = NaN;
    screeningCount = NaN;

    try
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

        if istable(inputs)
            x = table2array(inputs);
        else
            x = inputs;
        end

        x = round(x);

        orbit_indices = x(1:2:end);
        slot_indices  = x(2:2:end);

        stabilities_vec = stabilities_all(orbit_indices);

        num_obs = length(orbit_indices);
        observer_ICs = zeros(num_obs,6);

        for k=1:num_obs
            o_idx = orbit_indices(k);
            s_idx = slot_indices(k);

            o_idx = max(1, min(o_idx, length(orbit_database)));
            s_idx = max(1, min(s_idx, size(orbit_database{o_idx}, 1)));

            observer_ICs(k, :) = orbit_database{o_idx}(s_idx, :);
        end

        % run EKF (pass screening flag)
        [s_ekf, cov, screeningCount] = cr3bp_ekf(observer_ICs, s_lg, t_lg, ...
            P0, Q, R, mu, LU, sunFcn, sun_min, moon_min, useScreening);

        % compute cost with component toggles
        [J_total, J_1, J_2, J_3] = compute_cost(s_lg, s_ekf, cov, stabilities_vec, opt_flag, costFlags);

    catch ME 
        if strcmpi(opt_flag, 'MOO')
            J_total = [1e6; 1e6; 1e6];
        else
            J_total = 1e6;
        end
    end

    % ---------------- logging ----------------
    if nargin >= 15 && ~isempty(dq)
        entry = struct();
        entry.t = char(datetime("now","Format","yyyy-MM-dd HH:mm:ss.SSS"));
        entry.solver = char(solverName);
        entry.opt_flag = char(opt_flag);

        % Components (these are post-toggle values: inactive -> 0)
        entry.J1_rmse = J_1;
        entry.J2_det  = J_2;
        entry.J3_stab = J_3;

        % Store flags too
        entry.useJ1 = logical(costFlags.J1);
        entry.useJ2 = logical(costFlags.J2);
        entry.useJ3 = logical(costFlags.J3);

        if strcmpi(opt_flag,"SOO")
            entry.J_total = J_total;
        else
            entry.J_total1 = J_total(1);
            entry.J_total2 = J_total(2);
            entry.J_total3 = J_total(3);
        end

        entry.screeningCount = screeningCount;
        entry.x = x(:).';
        entry.orbit_indices = orbit_indices(:).';
        entry.slot_indices  = slot_indices(:).';
        send(dq, entry);
    end
end