% ---- test_visibility_trajectories.m ---- %
clear; close all; clc;

% ---------------- Settings ----------------
EKF_DT = 0.01;
gateway_periods = 1;
slots_per_orbit = 50;

sun_min   = deg2rad(20);
moon_min  = deg2rad(10);
earth_min = deg2rad(10);    % Test setting only

% Observer selection: each row is [orbit_index, slot_index].
% Leave empty to test the first orbit from each family at five slots.
% These are observers, separate from the transfer endpoints below.
observer_pairs = [];

% ---------------- Project path ----------------
projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

% ---------------- Constants ----------------
mu = 1.215058560962404E-2;
LU = 384400;               % km
TU = 375695;               % seconds

ode_opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);

sunFcn = @(t) sun_pos_bc4bp(t, LU, TU, 0, 0);
R_sun = 695700 / LU;

% ---------------- Load existing orbit catalog ----------------
catalogFile = projectPaths.catalog;

assert(isfile(catalogFile), ...
    'Place JPL_CR3BP_OrbitCatalog.mat in data/ or the project root.');

C = load(catalogFile, 'T');

assert(isfield(C, 'T') && istable(C.T), ...
    'The catalog must contain the orbit table named T.');

T1 = C.T;

periods = T1.("Period (TU) ");
times   = T1.("time");
states  = T1.("state");

num_orbits = height(T1);

assert(num_orbits >= 400, ...
    'The catalog must contain departure orbit 52 and arrival orbit 400.');

% ---------------- Build orbit database ----------------
% Same interpolation and corrected slot definition as run_opt.
% Do not load an orbit cache that might use the previous slot definition.

fprintf('\nBuilding orbit database...\n');

orbit_database = cell(num_orbits, 1);

for i = 1:num_orbits
    t_raw  = times{i};
    s_raw  = states{i};
    period = periods(i);

    t_slots = (0:slots_per_orbit-1)' * period / slots_per_orbit;

    [t_unique, idx_u] = unique(t_raw);
    s_unique = s_raw(idx_u, :);

    F = griddedInterpolant(t_unique, s_unique, 'spline');
    orbit_database{i} = F(t_slots);
end

% ---------------- Select observers ----------------
if isempty(observer_pairs)
    [~, family_indices] = unique(string(T1.orbitFamily), 'stable');

    slot_indices = [1, 13, 26, 38, 50];

    [orbit_grid, slot_grid] = ndgrid(family_indices, slot_indices);
    observer_pairs = [orbit_grid(:), slot_grid(:)];
end

assert(size(observer_pairs, 2) == 2, ...
    'observer_pairs must contain [orbit_index, slot_index] rows.');

assert(all(isfinite(observer_pairs), 'all') && ...
       all(observer_pairs == round(observer_pairs), 'all'), ...
    'Observer orbit and slot indices must be finite integers.');

assert(all(observer_pairs(:,1) >= 1 & ...
           observer_pairs(:,1) <= num_orbits), ...
    'An observer orbit index is outside the catalog.');

assert(all(observer_pairs(:,2) >= 1 & ...
           observer_pairs(:,2) <= slots_per_orbit), ...
    'An observer slot index is outside the slot range.');

num_obs = size(observer_pairs, 1);
observer_ICs = zeros(num_obs, 6);

for j = 1:num_obs
    i_orbit = observer_pairs(j, 1);
    i_slot  = observer_pairs(j, 2);

    observer_ICs(j,:) = orbit_database{i_orbit}(i_slot,:);
end

fprintf('Selected %d observer orbit/slot pairs.\n', num_obs);

% ---------------- Generate Gateway truth ----------------
% Same initial state, period, and truth cadence as run_opt.

missionCfg = struct();
missionCfg.type = "LUNAR_GATEWAY";

missionCfg.gateway.s0 = [
     1.02202108343387
     0
    -0.182096487798513
     0
    -0.103255420206012
     0
];

missionCfg.gateway.period   = 1.51110546287394;
missionCfg.gateway.dt       = 0.001;
missionCfg.gateway.Nperiods = gateway_periods;

fprintf('\nGenerating Lunar Gateway truth...\n');

[t_gateway, s_gateway, gatewayInfo] = build_target_truth( ...
    missionCfg, T1, orbit_database, times, states, mu, ode_opts);

% ---------------- Generate low-thrust truth ----------------
% Same endpoints and solver settings as run_opt.

missionCfg = struct();
missionCfg.type = "LOW_THRUST_TRANSFER";

missionCfg.transfer.depOrbitIndex = 51;
missionCfg.transfer.depSlot       = 10;
missionCfg.transfer.arrOrbitIndex = 400;
missionCfg.transfer.arrSlot       = 1;
missionCfg.transfer.dt            = 0.001;
missionCfg.transfer.solverMode    = "LOW_THRUST_CLASS";

missionCfg.transfer.lowthrust.sigma    = 1.0;
missionCfg.transfer.lowthrust.m0       = 1.0;
missionCfg.transfer.lowthrust.Tmax     = 0.3672;
missionCfg.transfer.lowthrust.ve       = 39.8;
missionCfg.transfer.lowthrust.tf_guess = 2.0;
missionCfg.transfer.lowthrust.tf_lb    = 0.1;
missionCfg.transfer.lowthrust.tf_ub    = 12.0;

missionCfg.transfer.lowth.lowthrust.lambda_guess = [
   -0.25
    0.75
    0.35
   -0.20
    0.40
    0.10
    0.05
];

missionCfg.transfer.lowthrust.lambda_lb = -20 * ones(7,1);
missionCfg.transfer.lowthrust.lambda_ub =  20 * ones(7,1);

missionCfg.transfer.lowthrust.w_pos_indirect  = 1;
missionCfg.transfer.lowthrust.w_vel_indirect  = 1;
missionCfg.transfer.lowthrust.w_norm_indirect = 1;
missionCfg.transfer.lowthrust.w_mass_indirect = 1;

fprintf('\nGenerating low-thrust transfer:\n');
fprintf('  Departure: orbit 52, slot 10\n');
fprintf('  Arrival:   orbit 400, slot 1\n');

[t_transfer, s_transfer, transferInfo] = build_target_truth( ...
    missionCfg, T1, orbit_database, times, states, mu, ode_opts);

fprintf('Solver exit flag:    %d\n', transferInfo.exitflag);
fprintf('Final residual norm: %.6e\n', transferInfo.finalResidualNorm);

if transferInfo.exitflag <= 0
    warning(['Low-thrust solver did not report convergence. ', ...
        'Visibility agreement alone does not validate the transfer.']);
end

% ---------------- Moon impact check ----------------
% Same sampled 100 km altitude check as run_opt.

r_moon = [1 - mu, 0, 0];
R_moon = 1737.1 / LU;
h_min  = 100 / LU;

d_moon = vecnorm(s_transfer(:,1:3) - r_moon, 2, 2);
min_d_moon = min(d_moon);

if min_d_moon <= R_moon + h_min
    error(['Low-thrust trajectory violates Moon keep-out zone. ', ...
        'Min distance = %.6e LU.'], min_d_moon);
end

% ---------------- Test scenarios ----------------
case_names  = ["LUNAR_GATEWAY", "LOW_THRUST_TRANSFER"];
case_times  = {t_gateway, t_transfer};
case_states = {s_gateway, s_transfer};
case_info   = {gatewayInfo, transferInfo};

results = struct([]);
summary_data = zeros(2, 9);

for m = 1:numel(case_names)

    fprintf('\n--- %s ---\n', char(case_names(m)));

    % ---------------- Resample target truth ----------------
    % Same spline interpolation and EKF cadence as run_opt.

    t_raw = case_times{m}(:);
    s_raw = case_states{m};

    assert(size(s_raw,1) == numel(t_raw) && size(s_raw,2) >= 6, ...
        'Target truth must have matching time rows and six state columns.');

    assert(all(isfinite(t_raw)) && all(isfinite(s_raw), 'all'), ...
        'Target truth contains nonfinite values.');

    [t_unique, idx_u] = unique(t_raw);
    s_unique = s_raw(idx_u, 1:6);

    assert(numel(t_unique) >= 2, ...
        'At least two distinct target epochs are required.');

    t = (t_unique(1):EKF_DT:t_unique(end)).';

    if t(end) < t_unique(end)
        t = [t; t_unique(end)];
    end

    F_truth = griddedInterpolant(t_unique, s_unique, 'spline');
    s_target = F_truth(t);

    % Skip the initialization epoch, matching the EKF measurement loop.
    t_eval = t(2:end);
    num_eval = numel(t_eval);

    sun_positions = zeros(num_eval, 3);

    for k = 1:num_eval
        sun_positions(k,:) = sunFcn(t_eval(k)).';
    end

    % ---------------- Allocate results ----------------
    old_geometry       = false(num_eval, num_obs);
    reference_geometry = false(num_eval, num_obs);
    new_geometry       = false(num_eval, num_obs);
    sun_occlusion      = false(num_eval, num_obs);

    old_visibility   = false(num_eval, num_obs);
    new_visibility   = false(num_eval, num_obs);
    earth_visibility = false(num_eval, num_obs);

    observer_positions = zeros(num_eval, 3, num_obs);

    % ---------------- Propagate and screen observers ----------------
    for j = 1:num_obs

        observer_state = observer_ICs(j,:).';

        for k = 1:num_eval

            dt = t(k+1) - t(k);

            % Same interval-by-interval CR3BP observer propagation.
            [t_obs, s_obs] = ode45( ...
                @(t,s) cr3bp_dynamics(t, s, mu), ...
                [0, dt], observer_state, ode_opts);

            assert(abs(t_obs(end) - dt) <= 1e-12, ...
                'Observer propagation stopped before the requested epoch.');

            observer_state = s_obs(end,:).';

            r_observer = observer_state(1:3);
            r_target   = s_target(k+1, 1:3).';
            r_sun      = sun_positions(k,:).';

            assert(all(isfinite(observer_state)), ...
                'Observer propagation produced a nonfinite state.');

            assert(norm(r_target - r_observer) > 0, ...
                'Observer and target coincide; the sightline is undefined.');

            observer_positions(k,:,j) = r_observer.';

            % Existing Earth/Moon physical occlusion.
            [occE, occM] = calc_occlusion( ...
                r_target, r_observer, mu, LU);

            old_geometry(k,j) = ~occE && ~occM;

            % Independent Sun segment/sphere check.
            % The old physical-occlusion function does not include the Sun.
            occS = test_segment_sphere( ...
                r_observer, r_target, r_sun, R_sun);

            sun_occlusion(k,j) = occS;
            reference_geometry(k,j) = old_geometry(k,j) && ~occS;

            % New physical screening with all exclusion angles zero.
            new_geometry(k,j) = calc_visibility( ...
                r_target, r_observer, r_sun, mu, LU, 0, 0, 0);

            % Existing combined screening.
            ok_exclusion = calc_exclusion( ...
                r_target, r_observer, r_sun, ...
                mu, sun_min, moon_min);

            old_visibility(k,j) = ...
                old_geometry(k,j) && ok_exclusion;

            % New combined screening without Earth exclusion.
            new_visibility(k,j) = calc_visibility( ...
                r_target, r_observer, r_sun, ...
                mu, LU, sun_min, moon_min, 0);

            % New combined screening with Earth exclusion.
            earth_visibility(k,j) = calc_visibility( ...
                r_target, r_observer, r_sun, ...
                mu, LU, sun_min, moon_min, earth_min);
        end

        fprintf('Observer %d/%d: orbit %d, slot %d complete.\n', ...
            j, num_obs, observer_pairs(j,1), observer_pairs(j,2));
    end

    % ---------------- Compare screening ----------------
    % Physical reference includes Earth, Moon, and Sun.
    geometry_mismatch = reference_geometry ~= new_geometry;

    % Compare directly with the existing Sun/Moon exclusion workflow.
    screening_mismatch = old_visibility ~= new_visibility;

    % Increasing Earth exclusion must never create a visible measurement.
    earth_failure = earth_visibility & ~new_visibility;

    additional_earth_rejection = new_visibility & ~earth_visibility;

    % ---------------- Record mismatches ----------------
    bad = geometry_mismatch | screening_mismatch | earth_failure;

    bad_idx = find(bad);
    [time_idx, observer_idx] = ind2sub(size(bad), bad_idx);

    mismatch_table = table( ...
        observer_pairs(observer_idx,1), ...
        observer_pairs(observer_idx,2), ...
        time_idx + 1, ...
        t_eval(time_idx), ...
        old_geometry(bad_idx), ...
        sun_occlusion(bad_idx), ...
        reference_geometry(bad_idx), ...
        new_geometry(bad_idx), ...
        old_visibility(bad_idx), ...
        new_visibility(bad_idx), ...
        earth_visibility(bad_idx), ...
        'VariableNames', { ...
            'orbit_index', 'slot_index', 'epoch_index', 'time_TU', ...
            'old_geometry', 'sun_occlusion', ...
            'reference_geometry', 'new_geometry', ...
            'old_visibility', 'new_visibility', 'earth_visibility'});

    % Keep the geometries and masks available for inspection.
    results(m).scenario = case_names(m);
    results(m).truthInfo = case_info{m};
    results(m).t = t_eval;
    results(m).observer_pairs = observer_pairs;
    results(m).observer_ICs = observer_ICs;
    results(m).observer_positions = observer_positions;
    results(m).target_positions = s_target(2:end,1:3);
    results(m).sun_positions = sun_positions;

    results(m).old_geometry = old_geometry;
    results(m).reference_geometry = reference_geometry;
    results(m).new_geometry = new_geometry;
    results(m).sun_occlusion = sun_occlusion;

    results(m).old_visibility = old_visibility;
    results(m).new_visibility = new_visibility;
    results(m).earth_visibility = earth_visibility;
    results(m).mismatches = mismatch_table;

    summary_data(m,:) = [ ...
        numel(old_visibility), ...
        nnz(~reference_geometry), ...
        nnz(old_visibility), ...
        nnz(new_visibility), ...
        nnz(geometry_mismatch), ...
        nnz(screening_mismatch), ...
        nnz(earth_failure), ...
        nnz(additional_earth_rejection), ...
        nnz(sun_occlusion)];

    fprintf('\nObserver/epoch combinations:       %d\n', ...
        numel(old_visibility));
    fprintf('Physical occlusions, all bodies:    %d\n', ...
        nnz(~reference_geometry));
    fprintf('Sun physical occlusions:            %d\n', ...
        nnz(sun_occlusion));
    fprintf('Physical occlusion mismatches:      %d\n', ...
        nnz(geometry_mismatch));
    fprintf('Combined screening mismatches:      %d\n', ...
        nnz(screening_mismatch));
    fprintf('Earth monotonicity failures:        %d\n', ...
        nnz(earth_failure));
    fprintf('Additional Earth rejections:       %d\n', ...
        nnz(additional_earth_rejection));

    if ~isempty(mismatch_table)
        fprintf('\nFirst mismatches:\n');
        disp(mismatch_table(1:min(10, height(mismatch_table)), :));
    end
end

% ---------------- Final summary ----------------
summary_table = array2table(summary_data, ...
    'VariableNames', { ...
        'ObserverEpochs', ...
        'PhysicalOcclusions', ...
        'OldVisible', ...
        'NewVisible', ...
        'GeometryMismatches', ...
        'ScreeningMismatches', ...
        'EarthFailures', ...
        'AdditionalEarthRejections', ...
        'SunOcclusions'}, ...
    'RowNames', cellstr(case_names(:)));

fprintf('\n--- Visibility test summary ---\n');
disp(summary_table);

assert(all(summary_data(:,5:7) == 0, 'all'), ...
    ['Visibility tests found discrepancies. Inspect ', ...
     'results(1).mismatches and results(2).mismatches.']);

fprintf('\nAll sampled trajectory visibility comparisons passed.\n');

if transferInfo.exitflag <= 0
    fprintf(['The transfer solver did not report convergence; ', ...
        'resolve that separately before using the trajectory.\n']);
end

% ---------------- Independent Sun occlusion reference ----------------
function hit = test_segment_sphere(p0, p1, center, radius)
    % Closest point on the finite observer-to-target segment.

    v = p1 - p0;
    vv = dot(v, v);

    if vv == 0
        hit = norm(p0 - center) <= radius;
        return;
    end

    fraction = dot(center - p0, v) / vv;
    fraction = min(max(fraction, 0), 1);

    closest = p0 + fraction*v;
    offset = closest - center;

    hit = dot(offset, offset) <= radius^2;
end
