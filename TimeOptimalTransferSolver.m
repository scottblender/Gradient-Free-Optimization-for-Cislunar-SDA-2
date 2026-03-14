classdef TimeOptimalTransferSolver
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
        function obj = TimeOptimalTransferSolver(cfg, T1, orbit_database, times, states, mu, ode_opts)
            obj.cfg = cfg;
            obj.T1 = T1;
            obj.orbit_database = orbit_database;
            obj.times = times;
            obj.states = states;
            obj.mu = mu;
            obj.ode_opts = ode_opts;
        end

        function [t_target, s_target, info] = solve(obj)
            tr  = obj.getTransferCfg();
            pmp = tr.pmp;

            arrIdx = tr.arrOrbitIndex;
            Tarr   = obj.T1.("Period (TU) ")(arrIdx);

            % ---- defaults / guesses ----
            if ~isfield(pmp,'lambda0_guess') || isempty(pmp.lambda0_guess)
                pmp.lambda0_guess = 1e-2 * ones(7,1);
            end
            if ~isfield(pmp,'tau_arr_guess') || isempty(pmp.tau_arr_guess)
                pmp.tau_arr_guess = 0.5 * Tarr;
            end
            if ~isfield(pmp,'tf_guess') || isempty(pmp.tf_guess)
                pmp.tf_guess = 1.0;
            end
            if ~isfield(pmp,'tf_lb') || isempty(pmp.tf_lb)
                pmp.tf_lb = 0.05;
            end
            if ~isfield(pmp,'tf_ub') || isempty(pmp.tf_ub)
                pmp.tf_ub = 10.0;
            end
            if ~isfield(pmp,'m0') || isempty(pmp.m0)
                pmp.m0 = 1.0;
            end
            if ~isfield(pmp,'Tmax') || isempty(pmp.Tmax)
                error('TimeOptimalTransferSolver:MissingTmax', ...
                    'cfg.transfer.pmp.Tmax must be provided in nondimensional units.');
            end
            if ~isfield(pmp,'ve') || isempty(pmp.ve)
                error('TimeOptimalTransferSolver:MissingVe', ...
                    'cfg.transfer.pmp.ve must be provided in nondimensional units.');
            end
            if ~isfield(tr,'dt') || isempty(tr.dt)
                tr.dt = 0.001;
            end

            theta_guess = obj.tf_to_theta(pmp.tf_guess, pmp.tf_lb, pmp.tf_ub);

            % y = [lambda0(7); tau_arr; theta_tf]
            y0 = [pmp.lambda0_guess(:); pmp.tau_arr_guess; theta_guess];

            opts = optimoptions('fsolve', ...
                'Display', 'iter', ...
                'FunctionTolerance', 1e-10, ...
                'StepTolerance', 1e-12, ...
                'OptimalityTolerance', 1e-10, ...
                'MaxFunctionEvaluations', 2e4, ...
                'MaxIterations', 500, ...
                'FiniteDifferenceType', 'central', ...
                'ScaleProblem', 'jacobian');

            [ysol, fval, exitflag, output] = fsolve(@(y)obj.residual(y, pmp), y0, opts);

            lambda0 = ysol(1:7);
            tau_arr = mod(ysol(8), Tarr);
            tf      = obj.theta_to_tf(ysol(9), pmp.tf_lb, pmp.tf_ub);

            if tf <= 0
                error('TimeOptimalTransferSolver:InvalidFinalTime', ...
                    'Recovered nonpositive final time tf = %.16g.', tf);
            end

            % rebuild final solution trajectory
            x0_6 = obj.getDepartureState();
            x0   = [x0_6; pmp.m0];
            X0   = [x0; lambda0];

            dt = tr.dt;
            t_eval = (0:dt:tf).';
            if isempty(t_eval) || t_eval(1) ~= 0
                t_eval = [0; t_eval];
            end
            if t_eval(end) < tf
                t_eval = [t_eval; tf];
            elseif t_eval(end) > tf
                t_eval(end) = tf;
            end

            odeAug = @(t,X) obj.augmentedDynamics(t, X, pmp);
            [t_target, Xtraj] = ode113(odeAug, t_eval, X0, obj.ode_opts);

            % return only physical 6-state trajectory
            s_target = Xtraj(:,1:6);

            x_dep = x0_6;
            x_arr = obj.getOrbitState(arrIdx, tau_arr).';

            Xf = Xtraj(end,:).';
            xf = Xf(1:7);
            lf = Xf(8:14);

            Hf = obj.hamiltonian(xf, lf, pmp);

            fprintf('\nTimeOptimalTransferSolver fsolve exitflag = %d\n', exitflag);
            fprintf('Residual norm = %.6e\n', norm(fval));
            disp('Residual vector:')
            disp(fval(:).')

            info = struct();
            info.method         = "Indirect PMP single-shooting (fixed start, free final phase)";
            info.depOrbitIndex  = tr.depOrbitIndex;
            info.depSlot        = tr.depSlot;
            info.arrOrbitIndex  = tr.arrOrbitIndex;
            info.tau_arr        = tau_arr;
            info.tf             = tf;
            info.lambda0        = lambda0(:).';
            info.exitflag       = exitflag;
            info.output         = output;
            info.residual       = fval(:).';
            info.residual_norm  = norm(fval);
            info.x_dep          = x_dep(:).';
            info.x_arr          = x_arr(:).';
            info.x_final        = xf(:).';
            info.lambda_final   = lf(:).';
            info.H_final        = Hf;
            info.Xtraj          = Xtraj;
            info.mass_traj      = Xtraj(:,7);
        end
    end

    methods (Access = private)
        function F = residual(obj, y, pmp)
            tr = obj.getTransferCfg();

            arrIdx = tr.arrOrbitIndex;
            Tarr   = obj.T1.("Period (TU) ")(arrIdx);

            lambda0 = y(1:7);
            tau_arr = mod(y(8), Tarr);
            tf      = obj.theta_to_tf(y(9), pmp.tf_lb, pmp.tf_ub);

            if ~isfinite(tf) || tf <= 0
                F = 1e3 * ones(9,1);
                return
            end

            % fixed departure state
            x_dep_6 = obj.getDepartureState();
            x0      = [x_dep_6; pmp.m0];
            X0      = [x0; lambda0];

            % propagate augmented state/costate system
            odeAug = @(t,X) obj.augmentedDynamics(t, X, pmp);
            try
                [~, X] = ode113(odeAug, [0 tf], X0, obj.ode_opts);
            catch
                F = 1e3 * ones(9,1);
                return
            end

            if isempty(X) || any(~isfinite(X(end,:)))
                F = 1e3 * ones(9,1);
                return
            end

            Xf = X(end,:).';
            xf = Xf(1:7);
            lf = Xf(8:14);

            % free final phase on arrival orbit
            x_arr_6 = obj.getOrbitState(arrIdx, tau_arr).';

            % 6 endpoint match equations
            Fmatch = xf(1:6) - x_arr_6;

            % free final mass
            Fmass = lf(7);

            % free final time
            Hf = obj.hamiltonian(xf, lf, pmp);

            % free final phase transversality
            arr_tangent_6 = obj.cr3bpStateDynamics(x_arr_6, [0;0;0]);
            arr_tangent_7 = [arr_tangent_6; 0];
            Farr = lf.' * arr_tangent_7;

            F = [Fmatch; Fmass; Hf; Farr];

            if any(~isfinite(F))
                F = 1e3 * ones(9,1);
            end
        end

        function dX = augmentedDynamics(obj, ~, X, pmp)
            x = X(1:7);
            l = X(8:14);

            u = obj.optimalControl(x, l);
            fx = obj.stateDynamics(x, u, pmp);

            % numerical Jacobian for costate dynamics
            A = obj.numericalJacobian(@(xx)obj.stateDynamics(xx, u, pmp), x);
            ldot = -A.' * l;

            dX = [fx; ldot];
        end

        function u = optimalControl(~, ~, l)
            lv = l(4:6);
            nlv = norm(lv);

            if nlv < 1e-12
                u = [0;0;0];
            else
                u = -lv / nlv;
            end
        end

        function H = hamiltonian(obj, x, l, pmp)
            u = obj.optimalControl(x, l);
            f = obj.stateDynamics(x, u, pmp);
            H = 1 + l.' * f;
        end

        function f = stateDynamics(obj, x, u, pmp)
            % x = [rx; ry; rz; vx; vy; vz; m]
            rx = x(1); ry = x(2); rz = x(3);
            vx = x(4); vy = x(5); vz = x(6);
            m  = x(7);

            mu = obj.mu;

            r1 = sqrt((rx + mu)^2 + ry^2 + rz^2);
            r2 = sqrt((rx - 1 + mu)^2 + ry^2 + rz^2);

            ax_g = 2*vy + rx ...
                - (1-mu)*(rx + mu)/r1^3 ...
                - mu*(rx - 1 + mu)/r2^3;

            ay_g = -2*vx + ry ...
                - (1-mu)*ry/r1^3 ...
                - mu*ry/r2^3;

            az_g = -(1-mu)*rz/r1^3 ...
                - mu*rz/r2^3;

            if m <= 1e-12
                aT = [0;0;0];
                mdot = 0;
            else
                aT = (pmp.Tmax / m) * u;
                mdot = -pmp.Tmax / pmp.ve;
            end

            f = [
                vx
                vy
                vz
                ax_g + aT(1)
                ay_g + aT(2)
                az_g + aT(3)
                mdot
            ];
        end

        function f6 = cr3bpStateDynamics(obj, x6, u)
            rx = x6(1); ry = x6(2); rz = x6(3);
            vx = x6(4); vy = x6(5); vz = x6(6);

            mu = obj.mu;

            r1 = sqrt((rx + mu)^2 + ry^2 + rz^2);
            r2 = sqrt((rx - 1 + mu)^2 + ry^2 + rz^2);

            ax_g = 2*vy + rx ...
                - (1-mu)*(rx + mu)/r1^3 ...
                - mu*(rx - 1 + mu)/r2^3;

            ay_g = -2*vx + ry ...
                - (1-mu)*ry/r1^3 ...
                - mu*ry/r2^3;

            az_g = -(1-mu)*rz/r1^3 ...
                - mu*rz/r2^3;

            f6 = [
                vx
                vy
                vz
                ax_g + u(1)
                ay_g + u(2)
                az_g + u(3)
            ];
        end

        function A = numericalJacobian(~, fun, x)
            fx = fun(x);
            n  = numel(x);
            m  = numel(fx);
            A  = zeros(m,n);

            for k = 1:n
                dx = zeros(n,1);
                h = 1e-7 * max(1, abs(x(k)));
                dx(k) = h;

                fp = fun(x + dx);
                fm = fun(x - dx);

                A(:,k) = (fp - fm) / (2*h);
            end
        end

        function s = getOrbitState(obj, orbitIdx, tau)
            t_raw = obj.times{orbitIdx}(:);
            s_raw = obj.states{orbitIdx};

            [t_unique, iu] = unique(t_raw);
            s_unique = s_raw(iu,:);

            Tper = obj.T1.("Period (TU) ")(orbitIdx);
            tau  = mod(tau, Tper);

            if abs(tau - Tper) < 1e-12
                tau = 0.0;
            end

            tau = max(min(tau, max(t_unique)), min(t_unique));

            Finterp = griddedInterpolant(t_unique, s_unique, 'spline');
            s = Finterp(tau);
        end

        function s_dep = getDepartureState(obj)
            tr = obj.getTransferCfg();
            s_dep = obj.orbit_database{tr.depOrbitIndex}(tr.depSlot,:).';
        end

        function tf = theta_to_tf(~, theta, tf_lb, tf_ub)
            tf = 0.5 * ((tf_ub + tf_lb) + (tf_ub - tf_lb) * sin(theta));
        end

        function theta = tf_to_theta(~, tf, tf_lb, tf_ub)
            xi = (2*tf - (tf_ub + tf_lb)) / (tf_ub - tf_lb);
            xi = max(-1, min(1, xi));
            theta = asin(xi);
        end

        function tr = getTransferCfg(obj)
            tr = obj.cfg;
            if isfield(tr, 'transfer')
                tr = tr.transfer;
            end
        end
    end
end