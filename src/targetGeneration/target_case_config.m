function missionCfg = target_case_config(missionType)
%TARGET_CASE_CONFIG Convert a stored fixed study case into run configuration.
%
% Fixed target scenarios are defined only in TargetCaseDatabase.mat. The
% observer-orbit catalog is not used to identify target endpoints.

caseDatabase = load_target_case_database();
missionType = upper(string(missionType));

missionCfg = struct();
missionCfg.type = missionType;

switch missionType
    case "LUNAR_GATEWAY"
        c = caseDatabase.gateway;
        missionCfg.gateway = struct( ...
            's0',c.state0(:), ...
            'period',c.period_TU, ...
            'dt',c.dt_TU, ...
            'Nperiods',c.Nperiods);

    case "LOW_THRUST_TRANSFER"
        c = caseDatabase.lowThrust;
        missionCfg.transfer = struct();
        missionCfg.transfer.fixedDepartureState = c.departureState(:).';
        missionCfg.transfer.fixedTargetState = c.arrivalState(:).';
        missionCfg.transfer.dt = c.dt_TU;
        missionCfg.transfer.solverMode = c.solverMode;
        missionCfg.transfer.lowthrust = c.lowthrust;

    case "GATEWAY_IMPULSE"
        c = caseDatabase.gatewayImpulse;
        missionCfg.impulse = struct( ...
            's0',c.nominalGatewayState0(:), ...
            'period',c.nominalPeriod_TU, ...
            'dt',c.dt_TU, ...
            'duration_TU',c.duration_TU, ...
            'deltaV_m_s',c.deltaV_m_s, ...
            'deltaV_LU_TU',c.deltaV_LU_TU, ...
            'direction',c.direction, ...
            'periluneSearchSamples',c.periluneSearchSamples);

    otherwise
        error('Unsupported fixed target case: %s',missionType);
end
end
