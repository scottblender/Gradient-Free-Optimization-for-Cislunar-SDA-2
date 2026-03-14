classdef FuelOptimalTransferSolver
    properties
        cfg
        T1
        orbit_database
        times
        states
        mu
        ode_opts
    end

    methods
        function obj = FuelOptimalTransferSolver(cfg, T1, orbit_database, times, states, mu, ode_opts)
            obj.cfg = cfg;
            obj.T1 = T1;
            obj.orbit_database = orbit_database;
            obj.times = times;
            obj.states = states;
            obj.mu = mu;
            obj.ode_opts = ode_opts;
        end

        function [t_target, s_target, info] = solve(obj) %#ok<STOUT>
            error("FuelOptimalTransferSolver:NotImplemented", ...
                "Fuel-optimal transfer solver not implemented yet.");
        end
    end
end