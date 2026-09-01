% ----- load_and_filter_data.m ----- %
% this script loads and filters data from the JPL Periodic Orbit Database
% based on shape and whether the orbit collides with the moon
close all;
clear;
clc
tic
projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();
dataPath = projectPaths.rawData;
files = dir(fullfile(dataPath,'*.csv'));
assert(~isempty(files), ...
    'No JPL CSV files found in %s. Use the existing catalog MAT file if available.', dataPath);
data = cell(length(files),1); % preallocate cell based on size of each file
parfor i = 1:length(files)
    Ti = readtable(fullfile(dataPath, files(i).name), "VariableNamingRule", "preserve");
    Ti.sourceFile = repmat(string(files(i).name), height(Ti), 1);  % <-- add this
    data{i} = Ti;
end
T = vertcat(data{:}); % concatenate all data into one table

% JPL Constants
mu = 1.215058560962404E-2;
LU = 384400;     % km
TU = 375695;     % seconds
VU = LU / TU;    % km/s
tol = 5/LU; % tolerance on moon radius to check for collision detection
R_moon = 1737.1/LU; % radius of the moon in LU
options = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);
N = height(T);

% ---------------- Lunar Gateway truth trajectory ----------------
dt_lg     = 0.01;   % TU
N_periods = 1;

s_lg_ic     = [1.02202108343387, 0, -0.182096487798513, 0, -0.103255420206012, 0]';
tspan_lg_ic = [0, 1.51110546287394];

tspan_lg = tspan_lg_ic(1):dt_lg:N_periods*tspan_lg_ic(2);
[t_lg, s_lg] = ode45(@(t,s) cr3bp_dynamics(t,s,mu), tspan_lg, s_lg_ic, options);

% propagate orbits and detect collision with moon
timeCell      = cell(N,1);
stateCell     = cell(N,1);
collidesVec   = false(N,1);
zAmplitudeVec = nan(N,1);
periluneAltitudeVec  = nan(N,1);
apoluneAltitudeVec   = nan(N,1);
xAmplitudeVec        = nan(N,1);
yAmplitudeVec        = nan(N,1);
inPlaneAmplitudeVec  = nan(N,1);
outPlaneAmplitudeVec = nan(N,1);
x0s     = T{:, "x0 (LU) "};        % extract data from each row of the table
y0s    = T{:, "y0 (LU) "};
z0s     = T{:, "z0 (LU) "};
vx0s    = T{:, "vx0 (LU/TU) "};
vy0s    = T{:, "vy0 (LU/TU) "};
vz0s    = T{:, "vz0 (LU/TU) "};
periods = T{:, "Period (TU) "};

% parallel for loop to integrate each orbit
parfor j = 1:N
    % Initial conditions
    s0 = [x0s(j), y0s(j), z0s(j), ...
          vx0s(j), vy0s(j), vz0s(j)];
    tspan = [0 periods(j)];
    options_event = odeset(options, ...
        'Events', @(t,s) moonImpactEvent(t,s,mu,R_moon));
    % Integrate
    [t, state, te] = ode45( ...
        @(t,s) cr3bp_dynamics(t,s,mu), ...
        tspan, s0, options_event);
    % Store trajectories
    timeCell{j}  = t;
    stateCell{j} = state;
    % Collision
    if ~isempty(te)
        collidesVec(j) = true;
        continue
    end

    % Moon-relative position
    rRel = state(:,1:3) - [1-mu, 0, 0];
    rMoon = vecnorm(rRel, 2, 2);
    
    % Moon-relative altitudes
    periluneAltitudeVec(j) = ...
        (min(rMoon) - R_moon) * LU;
    
    apoluneAltitudeVec(j) = ...
        (max(rMoon) - R_moon) * LU;
    
    % Half peak-to-peak amplitudes
    xAmplitudeVec(j) = ...
        0.5 * (max(state(:,1)) - min(state(:,1))) * LU;
    
    yAmplitudeVec(j) = ...
        0.5 * (max(state(:,2)) - min(state(:,2))) * LU;
    
    inPlaneAmplitudeVec(j) = hypot( ...
        xAmplitudeVec(j), yAmplitudeVec(j));
    
    outPlaneAmplitudeVec(j) = ...
        0.5 * (max(state(:,3)) - min(state(:,3))) * LU;
    
    % Retain the original nondimensional column if existing code needs it
    zAmplitudeVec(j) = max(abs(state(:,3)));
end
% add columns to table (time history, state history, collision,
% z-amplitude)
T.time        = timeCell;
T.state       = stateCell;
T.collides    = collidesVec;
T.zAmplitude  = zAmplitudeVec;
T.periluneAltitude_km  = periluneAltitudeVec;
T.apoluneAltitude_km   = apoluneAltitudeVec;
T.xAmplitude_km        = xAmplitudeVec;
T.yAmplitude_km        = yAmplitudeVec;
T.inPlaneAmplitude_km  = inPlaneAmplitudeVec;
T.outPlaneAmplitude_km = outPlaneAmplitudeVec;

% filter out orbits that collide with moon
T = T(~T.collides, :);

% create N-length array of strings to store orbit family classification
N = height(T);
states_local_fam = T.state; % cell array
orbitFamilies = strings(N, 1);
sourceFiles_local = T.sourceFile;
parfor k=1:N
     % --- override family if this row is a DRO file row ---
    if contains(sourceFiles_local(k), "distant_retrograde", "IgnoreCase", true)
        orbitFamilies(k) = "DRO";
        continue
    end
    s = states_local_fam{k}; % extract state for each orbit
    x_bar = mean(s(:,1)); % determine mean x-pos
    if x_bar < 1 - mu % filter into L1/L2 based on mean x_pos
        lp = "L1";
    else
        lp = "L2";
    end
    z_bar = mean(s(:,3)); % determine mean z-pos
    r_moon = [1-mu, 0, 0];
    r_rel = s(:,1:3) - [1-mu,0,0];        % relative to Moon
    r_orb_min = min(vecnorm(r_rel,2,2));  % minimum distance
    mag_r_thresh = 0.05 % LU, threshold for rectilinear orbit
    if z_bar > 0 % filter into N/S based on mean z_pos
        dir = "N"
        if r_orb_min < mag_r_thresh
            rect = "NRH"
        else
            rect = "H"
        end
    else
        dir = "S"
        if r_orb_min < mag_r_thresh
            rect = "NRH"
        else
            rect = "H"
        end
    end
    orbitFamilies(k) = dir + rect + lp; % classify orbit family
end
T.orbitFamily = orbitFamilies;

% exclude orbits near the LG trajectory
nearLG_thresh = 5e-3;   % LU (tune). 0.005 LU ~ 1900 km

rLG = s_lg(:,1:3);
states_local = T.state;      % cell array
N_local      = numel(states_local);
nearLG_score = nan(N_local,1);
nearLG = false(N_local,1);

parfor j = 1:N_local
    s = states_local{j}; % Mx6 integrated trajectory
    r = s(:,1:3);

    % Compute mean of point-to-curve min distance (cheap proximity metric)
    M = size(r,1);
    dmin = zeros(M,1);
    for k = 1:M
        diffs = rLG - r(k,:);              % NL x 3
        d2    = sum(diffs.^2, 2);          % NL x 1
        dmin(k) = sqrt(min(d2));           % scalar
    end
    score = mean(dmin);

    nearLG_score(j) = score;
    nearLG(j) = (score < nearLG_thresh);
end

T.nearLG_score = nearLG_score;
T.nearLG = nearLG;

fprintf('Excluding %d/%d as near-LG (thresh=%.3g LU)\n', nnz(T.nearLG), height(T), nearLG_thresh);
T = T(~T.nearLG, :);

% ---------------- Orbit-family selection ----------------
K = 50;

DRO_STABILITY_MAX = 1 + 1e-8;
DRO_LHS_SEED = 20260831;

T.stability = T.("Stability index  ");
T.period_TU = T.("Period (TU) ");

families = unique(T.orbitFamily);
keepMask = false(height(T),1);

for f = 1:numel(families)

    familyName = families(f);
    familyIdx = find(T.orbitFamily == familyName);

    if familyName == "DRO"

        stableMask = ...
            isfinite(T.stability(familyIdx)) & ...
            T.stability(familyIdx) <= DRO_STABILITY_MAX;

        candidateIdx = familyIdx(stableMask);

        fprintf(["DRO candidates: %d total, " ...
                 "%d satisfying stability <= %.8f\n"], ...
            numel(familyIdx), numel(candidateIdx), ...
            DRO_STABILITY_MAX);

        assert(numel(candidateIdx) >= K, ...
            ["Fewer than %d stable DRO candidates remain after " ...
             "collision and near-Gateway screening."], K);

        localTake = select_dro_lhs( ...
            T(candidateIdx,:), K, DRO_LHS_SEED);

        take = candidateIdx(localTake);

    else

        stability = T.stability(familyIdx);
        stability(~isfinite(stability)) = inf;

        [~, order] = sort(stability, "ascend");
        take = familyIdx(order(1:min(K,numel(order))));

    end

    keepMask(take) = true;
end

fprintf("Keeping %d total orbits after family selection.\n", ...
    nnz(keepMask));

T = T(keepMask,:);

T.orbitID = strings(height(T),1);

for i = 1:height(T)

    orbitKey = [ ...
        T.state{i}(1,:), ...
        T.period_TU(i)];

    T.orbitID(i) = "orb_" + study_hash(orbitKey);
end

% finally, sort orbits by z-amplitude
T = sortrows(T, ...
    ["orbitFamily", "apoluneAltitude_km", ...
     "period_TU", "orbitID"]);

referencePath = fullfile(projectRoot, ...
    "data", "transfer_reference.mat");

if isfile(referencePath)

    Sref = load(referencePath, "transferRef");
    transferRef = Sref.transferRef;

    transferRef.dep.newIndex = ...
        find_reference_orbit(T, transferRef.dep.state0);

    transferRef.arr.newIndex = ...
        find_reference_orbit(T, transferRef.arr.state0);

    transferRef.dep.orbitID = ...
        T.orbitID(transferRef.dep.newIndex);

    transferRef.arr.orbitID = ...
        T.orbitID(transferRef.arr.newIndex);

    save(referencePath, "transferRef");

    fprintf("Remapped transfer endpoints:\n");
    fprintf("  Departure: row %d, slot %d\n", ...
        transferRef.dep.newIndex, transferRef.dep.slot);
    fprintf("  Arrival:   row %d, slot %d\n", ...
        transferRef.arr.newIndex, transferRef.arr.slot);
end
toc




% --- HELPER FUNCTIONS --- %
% Function to detect if orbit will collide with the moon
function [value, isTerminal, direction] = moonImpactEvent(t,s,mu,R_moon)
% t - integration time
% s - state
% mu - mass ratio
% R_moon - radius of moon in LU
% returns value (distance between current integration state and the moon),
% isTerminal (flag to stop integration), and direction (forward
% integration)
    r_moon = [1-mu, 0, 0];
    r = [s(1), s(2), s(3)];
    dist = norm(r-r_moon);
    value = dist - R_moon;
    isTerminal = 1;
    direction = -1;
end

function take = select_dro_lhs(Tdro, K, seed)
% Select actual DRO catalog rows using Latin-hypercube targets distributed
% over the Moon-relative apolune-altitude range.

assert(height(Tdro) >= K, ...
    "The DRO candidate table must contain at least K rows.");

value = Tdro.apoluneAltitude_km;

assert(all(isfinite(value)), ...
    "DRO apolune altitudes contain nonfinite values.");

valueMin = min(value);
valueMax = max(value);

assert(valueMax > valueMin, ...
    "The DRO apolune-altitude range is zero.");

valueNormalized = ...
    (value - valueMin) ./ (valueMax - valueMin);

previousRng = rng;
cleanup = onCleanup(@() rng(previousRng));
rng(seed, "twister");

% One target in every equal-width portion of [0,1].
targets = lhsdesign(K, 1, "Criterion", "none");
targets = sort(targets);

available = true(height(Tdro),1);
take = zeros(K,1);

for k = 1:K

    distance = abs(valueNormalized - targets(k));
    distance(~available) = inf;

    [~, selected] = min(distance);

    take(k) = selected;
    available(selected) = false;
end
end

function index = find_reference_orbit(T, referenceState)

initialStates = zeros(height(T),6);

for i = 1:height(T)
    initialStates(i,:) = T.state{i}(1,:);
end

distance = vecnorm( ...
    initialStates - referenceState, 2, 2);

[minimumDistance, index] = min(distance);

assert(minimumDistance < 1e-10, ...
    ["The original transfer orbit was not retained in the " ...
    "new catalog. Minimum state difference: %.3e"], ...
    minimumDistance);
end