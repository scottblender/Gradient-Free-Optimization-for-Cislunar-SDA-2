function caseDatabase = build_target_case_database(outputPath)
%BUILD_TARGET_CASE_DATABASE Create fixed, reproducible target study cases.
%
% The target-case database is intentionally independent of the observer
% orbit catalog. It stores case definitions and initial/maneuver states,
% not full propagated truth trajectories.

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

if nargin < 1 || strlength(string(outputPath)) == 0
    outputPath = projectPaths.targetCaseDatabase;
end
outputPath = char(outputPath);

outputDir = fileparts(outputPath);
if ~isfolder(outputDir), mkdir(outputDir); end

mu = 1.215058560962404E-2;
LU = 384400;      % km
TU = 375695;      % s
VU = LU / TU;     % km/s
odeOptions = odeset('RelTol',1e-13,'AbsTol',1e-13);

caseDatabase = struct();
caseDatabase.schemaVersion = 1;
caseDatabase.definition = "fixed_target_cases_v1";
caseDatabase.frame = "Earth-Moon barycentric rotating CR3BP";
caseDatabase.created = string(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
caseDatabase.constants = struct( ...
    'mu',mu, ...
    'LU_km',LU, ...
    'TU_s',TU, ...
    'VU_km_s',VU);

% -------------------------------------------------------------------------
% Case 1: nominal Lunar Gateway NRHO.
% -------------------------------------------------------------------------
gateway = struct();
gateway.state0 = [ ...
    1.02202108343387; ...
    0; ...
   -0.182096487798513; ...
    0; ...
   -0.103255420206012; ...
    0];
gateway.period_TU = 1.51110546287394;
gateway.dt_TU = 0.001;
gateway.Nperiods = 1;
gateway.duration_TU = gateway.Nperiods * gateway.period_TU;
gateway.definition = "nominal_gateway_nrho";
caseDatabase.gateway = gateway;

% -------------------------------------------------------------------------
% Case 2: fixed-endpoint low-thrust transfer.
% These are the authoritative states used by the transfer solve. They do
% not depend on an observer-catalog row, orbit ID, or candidate slot.
% -------------------------------------------------------------------------
lowThrust = struct();
lowThrust.definition = "fixed_boundary_states";
lowThrust.departureState = [ ...
     0.8688395541375723; ...
     0.1110680873881317; ...
    -0.10760863551490674; ...
     0.10657734318058584; ...
     0.14953221747069609; ...
     0.19541894435638577];
lowThrust.arrivalState = [ ...
     1.0740681350221752; ...
     3.2857158725587851e-27; ...
    -0.20204469729197141; ...
     8.9272842881401699e-15; ...
    -0.19102742171285914; ...
    -9.1945220211817261e-15];
lowThrust.dt_TU = 0.001;
lowThrust.solverMode = "LOW_THRUST_CLASS";

lowThrust.lowthrust = struct();
lowThrust.lowthrust.sigma = 1.0;
lowThrust.lowthrust.m0 = 1.0;
lowThrust.lowthrust.Tmax = 0.3672;
lowThrust.lowthrust.ve = 39.8;
lowThrust.lowthrust.tf_guess = 2.0;
lowThrust.lowthrust.tf_lb = 0.1;
lowThrust.lowthrust.tf_ub = 12.0;
lowThrust.lowthrust.lambda_guess = ...
    [-0.25;0.75;0.35;-0.20;0.40;0.10;0.05];
lowThrust.lowthrust.lambda_lb = -20*ones(7,1);
lowThrust.lowthrust.lambda_ub =  20*ones(7,1);
lowThrust.lowthrust.w_pos_indirect = 1;
lowThrust.lowthrust.w_vel_indirect = 1;
lowThrust.lowthrust.w_norm_indirect = 1;
lowThrust.lowthrust.w_mass_indirect = 1;
caseDatabase.lowThrust = lowThrust;

% -------------------------------------------------------------------------
% Case 3: 10 m/s prograde impulse at nominal Gateway perilune.
% Store both sides of the instantaneous maneuver so downstream truth
% generation does not need to re-identify perilune or reconstruct the burn.
% -------------------------------------------------------------------------
impulse = build_gateway_impulse_definition( ...
    gateway,mu,VU,odeOptions);
caseDatabase.gatewayImpulse = impulse;

save(outputPath,'caseDatabase','-v7');

csvPath = fullfile(outputDir,'TargetCaseInitialConditions.csv');
write_case_initial_conditions(csvPath,caseDatabase);

fprintf('Saved target-case database to:\n  %s\n',outputPath);
fprintf('Saved target-case IC table to:\n  %s\n',csvPath);
end


function impulse = build_gateway_impulse_definition( ...
    gateway,mu,VU,odeOptions)

searchSamples = 4001;
nominalSolution = ode45( ...
    @(t,s) cr3bp_dynamics(t,s,mu), ...
    [0,gateway.period_TU],gateway.state0,odeOptions);

searchTimes = linspace(0,gateway.period_TU,searchSamples);
searchStates = deval(nominalSolution,searchTimes).';
moonPosition = [1-mu,0,0];
moonDistance = vecnorm(searchStates(:,1:3)-moonPosition,2,2);
[~,coarseIndex] = min(moonDistance);

if coarseIndex == 1 || coarseIndex == numel(searchTimes)
    periluneEpoch = searchTimes(coarseIndex);
else
    lowerTime = searchTimes(coarseIndex-1);
    upperTime = searchTimes(coarseIndex+1);
    searchOptions = optimset('Display','off','TolX',1e-13);
    periluneEpoch = fminbnd( ...
        @(t) moon_distance(nominalSolution,t,mu), ...
        lowerTime,upperTime,searchOptions);
end

preBurnState = deval(nominalSolution,periluneEpoch);
rMoon = preBurnState(1:3)-[1-mu;0;0];
vMoonInertial = preBurnState(4:6) + cross([0;0;1],rMoon);
directionUnit = vMoonInertial/norm(vMoonInertial);

deltaV_m_s = 10;
deltaV_LU_TU = (deltaV_m_s/1000)/VU;
deltaVVector = deltaV_LU_TU*directionUnit;

postBurnState = preBurnState;
postBurnState(4:6) = postBurnState(4:6)+deltaVVector;

impulse = struct();
impulse.definition = "fixed_gateway_perilune_impulse";
impulse.nominalGatewayState0 = gateway.state0;
impulse.nominalPeriod_TU = gateway.period_TU;
impulse.periluneEpoch_TU = periluneEpoch;
impulse.periluneDistance_LU = norm(rMoon);
impulse.preBurnState = preBurnState;
impulse.postBurnState = postBurnState;
impulse.deltaV_m_s = deltaV_m_s;
impulse.deltaV_LU_TU = deltaV_LU_TU;
impulse.deltaVVector_LU_TU = deltaVVector;
impulse.direction = "PROGRADE";
impulse.duration_TU = 1.5;
impulse.dt_TU = 0.001;
impulse.periluneSearchSamples = searchSamples;
end


function distance = moon_distance(solution,time,mu)

state = deval(solution,time);
distance = norm(state(1:3)-[1-mu;0;0]);
end


function write_case_initial_conditions(fileName,caseDatabase)

LU = caseDatabase.constants.LU_km;
VU = caseDatabase.constants.VU_km_s;

caseName = [ ...
    "Lunar Gateway"; ...
    "Low-thrust transfer"; ...
    "Low-thrust transfer"; ...
    "Gateway perilune impulse"; ...
    "Gateway perilune impulse"];
condition = ["Initial";"Departure";"Arrival";"Pre-burn";"Post-burn"];
epoch_TU = [ ...
    0; ...
    0; ...
    NaN; ...
    caseDatabase.gatewayImpulse.periluneEpoch_TU; ...
    caseDatabase.gatewayImpulse.periluneEpoch_TU];

state = [ ...
    caseDatabase.gateway.state0.'; ...
    caseDatabase.lowThrust.departureState.'; ...
    caseDatabase.lowThrust.arrivalState.'; ...
    caseDatabase.gatewayImpulse.preBurnState.'; ...
    caseDatabase.gatewayImpulse.postBurnState.'];

deltaV_m_s = [NaN;NaN;NaN;NaN;caseDatabase.gatewayImpulse.deltaV_m_s];

stateDim = state;
stateDim(:,1:3) = stateDim(:,1:3)*LU;
stateDim(:,4:6) = stateDim(:,4:6)*VU;

initialConditions = table( ...
    caseName,condition,epoch_TU,deltaV_m_s, ...
    state(:,1),state(:,2),state(:,3),state(:,4),state(:,5),state(:,6), ...
    stateDim(:,1),stateDim(:,2),stateDim(:,3), ...
    stateDim(:,4),stateDim(:,5),stateDim(:,6), ...
    'VariableNames',{ ...
    'caseName','condition','epoch_TU','deltaV_m_s', ...
    'x_LU','y_LU','z_LU','vx_LU_TU','vy_LU_TU','vz_LU_TU', ...
    'x_km','y_km','z_km','vx_km_s','vy_km_s','vz_km_s'});

writetable(initialConditions,fileName);
end
