% ----- load_and_filter_data.m ----- %
% this script loads and filters data from the JPL Periodic Orbit Database
% based on shape and whether the orbit collides with the moon
close all;
clear;
clc
tic
path = "C:\Users\scott\Documents\MATLAB\Gradient-Free-Optimization-for-Cislunar-SDA-2\JPL_Data";
files = dir(fullfile(path,'*.csv'));
data = cell(length(files),1); % preallocate cell based on size of each file
parfor i = 1:length(files) % loop through each file and store each file in a table within the cell
    data{i} = readtable(fullfile(path, files(i).name), "VariableNamingRule", "preserve");
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
    % z-amplitude
    zAmplitudeVec(j) = max(abs(state(:,3)));
end
% add columns to table (time history, state history, collision,
% z-amplitude)
T.time        = timeCell;
T.state       = stateCell;
T.collides    = collidesVec;
T.zAmplitude  = zAmplitudeVec;

% filter out orbits that collide with moon
T = T(~T.collides, :);

% create N-length array of strings to store orbit family classification
N = height(T);
states_local_fam = T.state; % cell array
orbitFamilies = strings(N, 1);
parfor k=1:N
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

nearLG = false(N_local,1);
nearLG_score = nan(N_local,1);

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

% keep top K most stable per family
K = 50;

% Grab stability column (your column name has extra spaces)
stab = T.("Stability index  ");
T.stability = stab;  % optional: normalize the name

families = unique(T.orbitFamily);
keepMask = false(height(T),1);

for f = 1:numel(families)
    fam = families(f);
    idx = find(T.orbitFamily == fam);

    stab_f = T.stability(idx);
    stab_f(~isfinite(stab_f)) = inf;

    % If "more stable" means SMALLER stability index, sort ascending.
    % If it means larger, flip the sort.
    [~, ord] = sort(stab_f, 'ascend');

    take = idx(ord(1:min(K, numel(ord))));
    keepMask(take) = true;
end

fprintf('Keeping %d total orbits after top-%d per family filter.\n', nnz(keepMask), K);
T = T(keepMask,:);

% finally, sort orbits by z-amplitude
T = sortrows(T, 'zAmplitude');
save('JPL_CR3BP_OrbitCatalog.mat','T','t_lg','s_lg','dt_lg','-v7.3');
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