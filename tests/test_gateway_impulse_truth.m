function audit = test_gateway_impulse_truth()
% Fast truth-generation test for the Gateway-perilune impulse scenario.

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
setup_project();

mu = 1.215058560962404E-2;
LU = 384400;
TU = 375695;
VU = LU / TU;

missionCfg = target_case_config("GATEWAY_IMPULSE");
cfg = missionCfg.impulse;

odeOptions = odeset( ...
    'RelTol',1e-13, ...
    'AbsTol',1e-13);

[t, state, info] = build_truth_gateway_impulse( ...
    cfg, mu, odeOptions);

assert(string(info.type) == "GATEWAY_IMPULSE", ...
    'Unexpected truth type.');
assert(t(1) == 0, ...
    'The post-impulse truth must begin at t = 0.');
assert(abs(t(end) - cfg.duration_TU) <= 10*eps(cfg.duration_TU), ...
    'The truth does not end at 1.5 TU.');
assert(all(isfinite(t)) && all(isfinite(state), 'all'), ...
    'The impulse truth contains nonfinite values.');
assert(all(diff(t) > 0), ...
    'Truth epochs are not strictly increasing.');

stateJump = info.statePostImpulse - info.statePreImpulse;

assert(norm(stateJump(1:3)) <= 1e-14, ...
    'Position changed across the instantaneous impulse.');
assert(norm(stateJump(4:6) - info.deltaVVector_LU_TU) <= 1e-14, ...
    'Velocity jump does not match the requested impulse.');
assert(abs(norm(info.deltaVVector_LU_TU) - ...
    cfg.deltaV_LU_TU) <= 1e-14, ...
    'Impulse magnitude is incorrect.');

rMoon = info.statePreImpulse(1:3) - [1-mu; 0; 0];
vMoonInertial = info.statePreImpulse(4:6) + ...
    cross([0; 0; 1], rMoon);

alignment = dot( ...
    info.deltaVVector_LU_TU / norm(info.deltaVVector_LU_TU), ...
    vMoonInertial / norm(vMoonInertial));

assert(alignment >= 1-1e-12, ...
    'The impulse is not prograde.');

% Confirm that the selected phase is a local minimum in Moon distance.
phaseStep = cfg.period / (cfg.periluneSearchSamples - 1);
tBefore = max(0, info.periluneEpochNominal_TU - phaseStep);
tAfter = min(cfg.period, ...
    info.periluneEpochNominal_TU + phaseStep);

nominalSolution = ode45( ...
    @(time,s) cr3bp_dynamics(time,s,mu), ...
    [0,cfg.period], cfg.s0, odeOptions);

distanceAt = @(time) norm( ...
    deval(nominalSolution,time,1:3) - [1-mu;0;0]);

distanceBefore = distanceAt(tBefore);
distanceAtPerilune = distanceAt(info.periluneEpochNominal_TU);
distanceAfter = distanceAt(tAfter);

assert(distanceAtPerilune <= distanceBefore + 1e-12 && ...
    distanceAtPerilune <= distanceAfter + 1e-12, ...
    'The impulse epoch is not a local perilune.');

Rmoon = 1737.1 / LU;
moonDistance = vecnorm( ...
    state(:,1:3) - [1-mu,0,0], 2, 2);
minimumAltitude_km = ...
    (min(moonDistance) - Rmoon) * LU;

assert(minimumAltitude_km > 100, ...
    'The post-impulse trajectory violates the 100 km Moon keep-out.');

audit = struct();
audit.periluneEpoch_TU = info.periluneEpochNominal_TU;
audit.periluneAltitude_km = ...
    (info.periluneDistance_LU - Rmoon) * LU;
audit.deltaV_m_s = info.deltaV_m_s;
audit.direction = info.direction;
audit.duration_TU = info.duration_TU;
audit.minimumPostImpulseAltitude_km = minimumAltitude_km;
audit.progradeAlignment = alignment;
audit.numEpochs = numel(t);

fprintf('\n--- Gateway impulse truth ---\n');
fprintf('Nominal perilune epoch:       %.12f TU\n', ...
    audit.periluneEpoch_TU);
fprintf('Nominal perilune altitude:    %.3f km\n', ...
    audit.periluneAltitude_km);
fprintf('Impulse:                      %.3f m/s %s\n', ...
    audit.deltaV_m_s, audit.direction);
fprintf('Tracking duration:            %.3f TU\n', ...
    audit.duration_TU);
fprintf('Minimum post-impulse altitude: %.3f km\n', ...
    audit.minimumPostImpulseAltitude_km);
fprintf('Prograde alignment:           %.12f\n', ...
    audit.progradeAlignment);
fprintf('Truth epochs:                 %d\n', ...
    audit.numEpochs);
fprintf('\nGateway impulse truth test passed.\n');
end
