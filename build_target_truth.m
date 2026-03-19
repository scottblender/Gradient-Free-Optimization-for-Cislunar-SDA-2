function [t_target, s_target, info] = build_target_truth( ...
    missionCfg, T1, orbit_database, times, states, mu, ode_opts)

missionType = upper(string(missionCfg.type));

switch missionType
    case "LUNAR_GATEWAY"
        [t_target, s_target, info] = build_truth_gateway( ...
            missionCfg.gateway, mu, ode_opts);

    case "PERIODIC_ORBIT"
        [t_target, s_target, info] = build_truth_periodic_orbit( ...
            missionCfg.periodic, T1, times, states, mu, ode_opts);
    
    case "LOW_THRUST_TRANSFER"

        solver = LowThrustTransferSolver( ...
            missionCfg.transfer, ...
            T1, orbit_database, times, states, mu, ode_opts);

        [t_target, s_target, info] = solver.solve();
    
    otherwise
        error("Unknown mission type: %s", missionType);
end
end