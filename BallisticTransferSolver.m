classdef BallisticTransferSolver
    properties
        cfg
        T1
        orbit_database
        times
        states
        mu
        ode_opts
        arrInterp
        arrPeriod
    end

    methods
        function obj = BallisticTransferSolver(cfg, T1, orbit_database, times, states, mu, ode_opts)
            obj.cfg = cfg;
            obj.T1 = T1;
            obj.orbit_database = orbit_database;
            obj.times = times;
            obj.states = states;
            obj.mu = mu;
            obj.ode_opts = ode_opts;

            [obj.arrInterp, obj.arrPeriod] = obj.buildArrivalInterpolant();
        end

        function [t_target, s_target, info] = solve(obj)
            s_dep = obj.getDepartureState();

            z0 = obj.initialGuess();
            [lb, ub] = obj.bounds();

            objfun = @(z) obj.cost(z, s_dep);

            opts = optimoptions('fmincon', ...
                'Display', 'iter', ...
                'Algorithm', 'sqp', ...
                'MaxIterations', 100, ...
                'MaxFunctionEvaluations', 1000);

            [z_best, fval, exitflag, output] = fmincon( ...
                objfun, z0, [], [], [], [], lb, ub, [], opts);

            [t_target, s_target, finalData] = obj.propagate(z_best, s_dep);

            info = obj.packInfo(z_best, fval, finalData, exitflag, output);
        end
    end

    methods (Access = private)
        function s_dep = getDepartureState(obj)
            iDep = obj.cfg.depOrbitIndex;
            jDep = obj.cfg.depSlot;
            s_dep = obj.orbit_database{iDep}(jDep,:).';
        end

        function z0 = initialGuess(obj)
            if isfield(obj.cfg.ballistic, 'phase_guess')
                phase_guess = obj.cfg.ballistic.phase_guess;
            else
                phase_guess = 0.5 * obj.arrPeriod;
            end

            z0 = [obj.cfg.ballistic.dv_guess(:); ...
                  obj.cfg.ballistic.tf_guess; ...
                  phase_guess];
        end

        function [lb, ub] = bounds(obj)
            lb = [-obj.cfg.ballistic.dv_max; ...
                  -obj.cfg.ballistic.dv_max; ...
                  -obj.cfg.ballistic.dv_max; ...
                   obj.cfg.ballistic.tf_lb; ...
                   0];

            ub = [ obj.cfg.ballistic.dv_max; ...
                   obj.cfg.ballistic.dv_max; ...
                   obj.cfg.ballistic.dv_max; ...
                   obj.cfg.ballistic.tf_ub; ...
                   obj.arrPeriod];
        end

        function J = cost(obj, z, s_dep)
            try
                [~, ~, data] = obj.propagate(z, s_dep);
                e = data.residual;

                w_pos   = 1e4;
                w_vel   = 1e4;
                w_dv    = 1;
                w_tf    = 1e-1;
                w_phase = 0; %#ok<NASGU>

                J = w_pos*sum(e(1:3).^2) + ...
                    w_vel*sum(e(4:6).^2) + ...
                    w_dv*sum(z(1:3).^2) + ...
                    w_tf*z(4);
            catch
                J = 1e12;
            end
        end

        function [t_out, s_out, data] = propagate(obj, z, s_dep)
            dv0   = z(1:3);
            tf    = z(4);
            phase = z(5);

            x0 = s_dep;
            x0(4:6) = x0(4:6) + dv0(:);

            t_out = (0:obj.cfg.dt:tf).';
            if isempty(t_out) || t_out(end) < tf
                t_out = [t_out; tf];
            end

            [t_out, s_out] = ode45(@(t,s) cr3bp_dynamics(t,s,obj.mu), ...
                t_out, x0, obj.ode_opts);

            x_arr = obj.evalArrivalState(phase);
            xf    = s_out(end,:).';
            resid = xf - x_arr;

            data = struct();
            data.residual = resid;
            data.phase = phase;
            data.xf = xf;
            data.x_arr = x_arr;
        end

        function x_arr = evalArrivalState(obj, phase)
            phaseWrapped = mod(phase, obj.arrPeriod);
            x_arr = obj.arrInterp(phaseWrapped).';
        end

        function [F, period] = buildArrivalInterpolant(obj)
            iArr = obj.cfg.arrOrbitIndex;

            t_raw = obj.times{iArr}(:);
            s_raw = obj.states{iArr};

            [t_unique, idx_u] = unique(t_raw);
            s_unique = s_raw(idx_u, :);

            period = t_unique(end);

            if t_unique(1) ~= 0
                error('Arrival orbit time history must start at t = 0.');
            end

            if t_unique(end) ~= period || any(abs(s_unique(1,:) - s_unique(end,:)) > 1e-10)
                t_aug = [t_unique; period];
                s_aug = [s_unique; s_unique(1,:)];
            else
                t_aug = t_unique;
                s_aug = s_unique;
            end

            F = griddedInterpolant(t_aug, s_aug, 'spline');
        end

        function info = packInfo(obj, z_best, fval, finalData, exitflag, output)
            info = struct();
            info.type = "BALLISTIC_TRANSFER";
            info.builder = "BallisticTransferSolver";
            info.depOrbitIndex = obj.cfg.depOrbitIndex;
            info.depSlot = obj.cfg.depSlot;
            info.arrOrbitIndex = obj.cfg.arrOrbitIndex;
            info.dt = obj.cfg.dt;
            info.dv0 = z_best(1:3);
            info.tf = z_best(4);
            info.phase = z_best(5);
            info.objectiveValue = fval;
            info.finalResidual = finalData.residual;
            info.finalResidualNorm = norm(finalData.residual);
            info.exitflag = exitflag;
            info.output = output;
        end
    end
end