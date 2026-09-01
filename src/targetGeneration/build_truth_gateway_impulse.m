function [t_target, s_target, info] = build_truth_gateway_impulse( ...
    cfg, mu, ode_opts)
% Build a post-maneuver Gateway truth trajectory.
% The nominal Gateway orbit is searched over one period for minimum
% Moon-centered distance. The impulse is applied at that state, and the
% returned truth begins immediately after the burn at t = 0.

requiredFields = {
    's0'
    'period'
    'dt'
    'duration_TU'
    'deltaV_m_s'
    'deltaV_LU_TU'
    'direction'
    'periluneSearchSamples'
};

for k = 1:numel(requiredFields)
    assert(isfield(cfg, requiredFields{k}), ...
        'Gateway impulse config is missing %s.', requiredFields{k});
end

validateattributes(cfg.s0, {'numeric'}, ...
    {'vector','numel',6,'real','finite'});
validateattributes(cfg.period, {'numeric'}, ...
    {'scalar','real','finite','positive'});
validateattributes(cfg.dt, {'numeric'}, ...
    {'scalar','real','finite','positive'});
validateattributes(cfg.duration_TU, {'numeric'}, ...
    {'scalar','real','finite','positive'});
validateattributes(cfg.deltaV_LU_TU, {'numeric'}, ...
    {'scalar','real','finite','positive'});
validateattributes(cfg.periluneSearchSamples, {'numeric'}, ...
    {'scalar','real','finite','integer','>=',101});

nominalSolution = ode45( ...
    @(t,s) cr3bp_dynamics(t,s,mu), ...
    [0, cfg.period], cfg.s0(:), ode_opts);

searchTimes = linspace( ...
    0, cfg.period, cfg.periluneSearchSamples);

searchStates = deval(nominalSolution, searchTimes).';
moonPosition = [1-mu, 0, 0];
moonDistance = vecnorm( ...
    searchStates(:,1:3) - moonPosition, 2, 2);

[~, coarseIndex] = min(moonDistance);

if coarseIndex == 1 || coarseIndex == numel(searchTimes)
    periluneEpoch = searchTimes(coarseIndex);
else
    lowerTime = searchTimes(coarseIndex-1);
    upperTime = searchTimes(coarseIndex+1);

    searchOptions = optimset( ...
        'Display','off', ...
        'TolX',1e-13);

    periluneEpoch = fminbnd( ...
        @(t) moon_distance(nominalSolution,t,mu), ...
        lowerTime, upperTime, searchOptions);
end

statePreImpulse = deval(nominalSolution, periluneEpoch);
rMoon = statePreImpulse(1:3) - [1-mu; 0; 0];

% Moon-relative inertial velocity expressed in the rotating basis.
vMoonInertial = statePreImpulse(4:6) + ...
    cross([0; 0; 1], rMoon);

directionName = upper(string(cfg.direction));

switch directionName
    case "PROGRADE"
        directionUnit = vMoonInertial / norm(vMoonInertial);

    case "RETROGRADE"
        directionUnit = -vMoonInertial / norm(vMoonInertial);

    case "RADIAL_OUTWARD"
        directionUnit = rMoon / norm(rMoon);

    case "RADIAL_INWARD"
        directionUnit = -rMoon / norm(rMoon);

    otherwise
        error('Unknown impulse direction: %s', directionName);
end

deltaV = cfg.deltaV_LU_TU * directionUnit;

statePostImpulse = statePreImpulse;
statePostImpulse(4:6) = ...
    statePostImpulse(4:6) + deltaV;

t_target = (0:cfg.dt:cfg.duration_TU).';
if t_target(end) < cfg.duration_TU
    t_target = [t_target; cfg.duration_TU];
end

[t_target, s_target] = ode45( ...
    @(t,s) cr3bp_dynamics(t,s,mu), ...
    t_target, statePostImpulse, ode_opts);

info = struct();
info.type = "GATEWAY_IMPULSE";
info.builder = "build_truth_gateway_impulse";
info.nominalPeriod_TU = cfg.period;
info.periluneEpochNominal_TU = periluneEpoch;
info.periluneDistance_LU = norm(rMoon);
info.direction = directionName;
info.deltaV_m_s = cfg.deltaV_m_s;
info.deltaV_LU_TU = cfg.deltaV_LU_TU;
info.deltaVVector_LU_TU = deltaV;
info.statePreImpulse = statePreImpulse;
info.statePostImpulse = statePostImpulse;
info.duration_TU = cfg.duration_TU;
info.dt = cfg.dt;
end


function distance = moon_distance(solution, time, mu)

state = deval(solution, time);
distance = norm(state(1:3) - [1-mu; 0; 0]);
end
