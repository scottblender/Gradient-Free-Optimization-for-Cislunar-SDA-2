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

% Normalize the source identifier column once, before expensive propagation.
tableNames = string(T.Properties.VariableNames);
idMatch = find(strcmpi(strtrim(tableNames), "id"));

assert(numel(idMatch) == 1, ...
    'Expected exactly one source Id column, but found %d.', ...
    numel(idMatch));

sourceIdName = tableNames(idMatch);
if sourceIdName ~= "Id"
    T.Id = T.(sourceIdName);
end

% JPL Id values restart in each CSV, so use source file + Id globally.
sourceStem = erase(lower(strtrim(string(T.sourceFile))), ".csv");
sourceIds = strtrim(string(T.Id));
T.orbitID = sourceStem + ":" + sourceIds;

assert(all(strlength(sourceIds) > 0), ...
    'One or more source Id values are empty.');
assert(numel(unique(T.orbitID)) == height(T), ...
    'The composite sourceFile + Id identifiers are not unique.');

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

% Exclude observer orbits whose complete geometry is too similar to
% either the nominal Lunar Gateway orbit or its north-south mirror. The
% mirrored comparison prevents a flipped Gateway-like NRHO from entering
% the observer database.
nearLG_thresh = 1.5e-2;  % LU mean curve distance (~5766 km)

rLG = s_lg(:,1:3);
rLGMirror = rLG;
rLGMirror(:,3) = -rLGMirror(:,3);

states_local = T.state;
N_local = numel(states_local);
nearLG_score_nominal = nan(N_local,1);
nearLG_score_mirrored = nan(N_local,1);
nearLG_score = nan(N_local,1);
nearLG = false(N_local,1);

parfor j = 1:N_local
    r = states_local{j}(:,1:3);

    nominalScore = mean_curve_distance(r,rLG);
    mirroredScore = mean_curve_distance(r,rLGMirror);
    score = min(nominalScore,mirroredScore);

    nearLG_score_nominal(j) = nominalScore;
    nearLG_score_mirrored(j) = mirroredScore;
    nearLG_score(j) = score;
    nearLG(j) = score < nearLG_thresh;
end

T.nearLG_score_nominal = nearLG_score_nominal;
T.nearLG_score_mirrored = nearLG_score_mirrored;
T.nearLG_score = nearLG_score;
T.nearLG = nearLG;

fprintf([ ...
    'Excluding %d/%d as Gateway-like using nominal and mirrored ' ...
    'geometry (threshold = %.3g LU).\n'], ...
    nnz(T.nearLG),height(T),nearLG_thresh);
T = T(~T.nearLG,:);

% ---------------- Orbit-family selection ----------------
% Select 50 representative trajectories from the full eligible population
% of every family using one-dimensional Latin-hypercube targets along a
% geometry coordinate that spans the family continuation. Stability is
% deliberately NOT used as either a filter or a sampling coordinate so it
% remains an outcome variable for the optimization study.
K = 50;
LHS_SEED_BASE = 20260901;

T.stability = T.("Stability index  ");
T.period_TU = T.("Period (TU) ");

families = sort(unique(T.orbitFamily));
keepMask = false(height(T),1);

for f = 1:numel(families)

    familyName = families(f);
    familyIdx = find(T.orbitFamily == familyName);

    assert(numel(familyIdx) >= K, ...
        ['Fewer than %d eligible %s candidates remain after collision ' ...
         'and near-Gateway screening.'], K, familyName);

    if familyName == "DRO"
        % DROs are planar, so out-of-plane amplitude cannot parameterize
        % the family. Moon-relative apolune altitude provides a monotonic
        % geometric progression across the available DRO trajectories.
        sampleValue = T.apoluneAltitude_km(familyIdx);
        sampleCoordinate = "apolune altitude";
    else
        % Halo and NRHO families are naturally spanned by their
        % out-of-plane extent. Use the propagated maximum |z| amplitude,
        % not stability, to distribute the 50 representatives.
        sampleValue = T.zAmplitude(familyIdx);
        sampleCoordinate = "z amplitude";
    end

    localTake = select_family_lhs( ...
        sampleValue, K, LHS_SEED_BASE + f);
    take = familyIdx(localTake);

    fprintf('%s candidates: %d eligible; selected %d by LHS over %s.
', ...
        familyName, numel(familyIdx), numel(take), sampleCoordinate);

    keepMask(take) = true;
end

fprintf("Keeping %d total orbits after all-family LHS selection.
", ...
    nnz(keepMask));

T = T(keepMask,:);

assert(ismember("orbitID", string(T.Properties.VariableNames)), ...
    "The selected catalog does not contain orbitID.");

catalogIds = string(T.orbitID);
assert(all(strlength(catalogIds) > 0), ...
    "One or more selected orbits have an empty orbitID.");
assert(numel(unique(catalogIds)) == height(T), ...
    "The selected catalog contains duplicate orbitID values.");

% Preserve the original global z-amplitude ordering for Halo and
% near-rectilinear Halo families. Organize the LHS-sampled DRO population
% by ascending apolune altitude, then append it to the catalog.
isDROFinal = T.orbitFamily == "DRO";

haloCatalog = sortrows(T(~isDROFinal,:), ...
    ["zAmplitude", "orbitID"]);
droCatalog = sortrows(T(isDROFinal,:), ...
    ["apoluneAltitude_km", "orbitID"]);

T = [haloCatalog; droCatalog];

catalogPath = fullfile(projectPaths.data, ...
    "JPL_CR3BP_OrbitCatalog.mat");

save(catalogPath, "T", "t_lg", "s_lg", "dt_lg", "-v7.3");
fprintf("Saved orbit catalog to:\n  %s\n", catalogPath);

toc




% --- HELPER FUNCTIONS --- %

function score = mean_curve_distance(candidatePosition,referencePosition)
% Mean nearest-neighbor distance from a candidate orbit to a reference
% curve. This is phase independent and retains the original inexpensive
% screening definition.

numberOfPoints = size(candidatePosition,1);
minimumDistance = zeros(numberOfPoints,1);

for k = 1:numberOfPoints
    difference = referencePosition-candidatePosition(k,:);
    minimumDistance(k) = sqrt(min(sum(difference.^2,2)));
end

score = mean(minimumDistance);
end

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

function take = select_family_lhs(value, K, seed)
% Select actual catalog rows using Latin-hypercube targets distributed over
% a scalar geometric coordinate spanning one eligible orbit family.

value = value(:);
assert(numel(value) >= K, ...
    "The family candidate set must contain at least K rows.");
assert(all(isfinite(value)), ...
    "The family LHS coordinate contains nonfinite values.");

valueMin = min(value);
valueMax = max(value);
assert(valueMax > valueMin, ...
    "The family LHS coordinate range is zero.");

valueNormalized = (value - valueMin) ./ (valueMax - valueMin);

previousRng = rng;
cleanup = onCleanup(@() rng(previousRng)); %#ok<NASGU>
rng(seed, "twister");

% Draw one target in every equal-width stratum of [0,1]. The nearest
% currently available catalog trajectory is assigned to each target so the
% result remains a subset of the original JPL orbit population.
targets = lhsdesign(K, 1, "Criterion", "none");
targets = sort(targets);

available = true(numel(value),1);
take = zeros(K,1);

for k = 1:K
    distance = abs(valueNormalized - targets(k));
    distance(~available) = inf;
    [~, selected] = min(distance);
    take(k) = selected;
    available(selected) = false;
end
end
