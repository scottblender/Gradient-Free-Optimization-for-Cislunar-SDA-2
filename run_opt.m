% ---- run_optimization.m ---- %
clear; close all; clc;

% load in filtered and sorted JPL data
S = load('JPL_CR3BP_OrbitCatalog.mat'); 
T1 = S.T;

% User-specified Inputs
% Options: 'GA', 'PSO', 'BAYESIAN'
OPTIMIZER_MODE = 'GA';

% Number of observers to optimize
nvars = 3;

% Stopping Criteria
N_STALL = 25;

% JPL Constants
mu = 1.215058560962404E-2;
LU = 384400;     % km
TU = 375695;     % seconds
VU = LU / TU;    % km/s

ode_opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-13);

% --- EKF Parameters ---
pos_var = (1 / LU)^2;
vel_var = (10 / (VU * 1000))^2;
P_0_base = diag([pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]);
Q_k = diag(repmat(1e-8, 6, 1));
R_k_base = diag([1e-8, 1e-8]);

% --- Lunar Gateway ICs ---
s_lg = [1.02202108343387, 0, -0.182096487798513, 0, -0.103255420206012, 0]';
tspan_lg = [0, 1.51110546287394];

% MILP-Implementation
num_orbits = int32(height(T1)); % number of candidate orbits 
slots_per_orbit = 100;          % number of discrete slots per orbit

tf    = T1.("Period (TU) ");
states = T1.("state");
times  = T1.("time");

time_interp  = cell(num_orbits, 1);
state_interp = cell(num_orbits, 1);

parfor i = 1:num_orbits
    t_query = linspace(0, tf(i), slots_per_orbit)';  % query times for this orbit
    time_interp{i} = t_query;

    t_raw = times{i};
    s_raw = states{i};

    % interpolated propagated trajectories for specific time slots
    state_interp{i} = interp1(t_raw, s_raw, t_query, 'linear', 'extrap');
end
