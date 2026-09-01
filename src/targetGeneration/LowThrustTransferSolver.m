classdef LowThrustTransferSolver
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
        function obj = LowThrustTransferSolver(cfg, T1, orbit_database, times, states, mu, ode_opts)
            obj.cfg = cfg;
            obj.T1 = T1;
            obj.orbit_database = orbit_database;
            obj.times = times;
            obj.states = states;
            obj.mu = mu;
            obj.ode_opts = ode_opts;
        end

        function [t_target, s_target, info] = solve(obj)
            tr = obj.getTransferCfg();
            lt = tr.lowthrust;

            x_dep = obj.getDepartureState();
            x_arr = obj.getArrivalTargetState();

            z0 = obj.initialGuess();

            fun = @(z) obj.boundaryResidual(z, x_dep, x_arr);

            opts = optimoptions(@fsolve, ...
                'Algorithm', 'levenberg-marquardt', ...
                'Display', 'iter', ...
                'MaxIterations', 50000, ...
                'MaxFunctionEvaluations', 1e6, ...
                'StepTolerance', 1e-16, ...
                'FunctionTolerance', 1e-16, ...
                'OptimalityTolerance', 1e-16);

            [z_sol, residual, exitflag, output] = fsolve(fun, z0, opts);

            % Optional post-check against user bounds
            [lb, ub] = obj.bounds();
            if any(z_sol < lb) || any(z_sol > ub)
                warning('LowThrustTransferSolver:SolutionOutsideBounds', ...
                    'fsolve solution lies outside configured bounds.');
            end

            [t_target, s_target, finalData] = obj.propagate(z_sol, x_dep, x_arr);

            info = struct();
            info.type              = "LOW_THRUST_TRANSFER";
            info.builder           = "LowThrustTransferSolver";
            info.method            = "INDIRECT_SINGLE_SHOOTING_FIXED_ENDPOINT_FSOLVE";
            info.depOrbitIndex     = tr.depOrbitIndex;
            info.depSlot           = tr.depSlot;
            info.arrOrbitIndex     = tr.arrOrbitIndex;
            info.arrSlot           = obj.getArrivalSlot();
            info.dt                = tr.dt;
            info.Tmax              = lt.Tmax;
            info.ve                = lt.ve;
            info.m0                = lt.m0;
            info.sigma             = obj.getSigma();
            info.tf                = z_sol(end);
            info.lambda0           = z_sol(1:7);
            info.finalResidual     = finalData.residual;
            info.finalResidualNorm = norm(finalData.residual);
            info.exitflag          = exitflag;
            info.output            = output;
            info.resnorm           = norm(residual)^2;
            info.residualVector    = residual;
            info.x_dep             = x_dep;
            info.xf                = finalData.xf;
            info.x_arr             = finalData.x_arr;
            if isfield(tr,'fixedDepartureState') && ...
                    ~isempty(tr.fixedDepartureState)
                info.departureStateSource = "FIXED_STATE";
            else
                info.departureStateSource = "ORBIT_DATABASE_SLOT";
            end
            info.lambda_f          = finalData.lambda_f;
            info.mass_final        = finalData.mass_final;
            info.controls          = finalData.controls;
            info.switchingFunction = finalData.switchingFunction;
        end

        function [t_out, s_out, data] = runInitialGuess(obj)
            x_dep = obj.getDepartureState();
            x_arr = obj.getArrivalTargetState();
            z0 = obj.initialGuess();
            [t_out, s_out, data] = obj.propagate(z0, x_dep, x_arr);
        end
    end

    methods (Access = private)
        function tr = getTransferCfg(obj)
            tr = obj.cfg;
            if isfield(tr, 'transfer')
                tr = tr.transfer;
            end
        end

        function x_dep = getDepartureState(obj)
            tr = obj.getTransferCfg();

            if isfield(tr,'fixedDepartureState') && ...
                    ~isempty(tr.fixedDepartureState)
                x_dep = tr.fixedDepartureState(:);
                if numel(x_dep) ~= 6
                    error( ...
                        'transfer.fixedDepartureState must have 6 elements.');
                end
                return;
            end

            iDep = tr.depOrbitIndex;
            jDep = tr.depSlot;
            x_dep = obj.orbit_database{iDep}(jDep,:).';
        end

        function jArr = getArrivalSlot(obj)
            tr = obj.getTransferCfg();

            if isfield(tr, 'arrSlot') && ~isempty(tr.arrSlot)
                jArr = tr.arrSlot;
            else
                jArr = 1;
            end
        end

        function x_arr = getArrivalTargetState(obj)
            tr = obj.getTransferCfg();

            if isfield(tr, 'fixedTargetState') && ~isempty(tr.fixedTargetState)
                x_arr = tr.fixedTargetState(:);
                if numel(x_arr) ~= 6
                    error('transfer.fixedTargetState must have 6 elements.');
                end
                return;
            end

            if isfield(tr, 'lowthrust') && isfield(tr.lowthrust, 'fixed_target_state') ...
                    && ~isempty(tr.lowthrust.fixed_target_state)
                x_arr = tr.lowthrust.fixed_target_state(:);
                if numel(x_arr) ~= 6
                    error('lowthrust.fixed_target_state must have 6 elements.');
                end
                return;
            end

            iArr = tr.arrOrbitIndex;
            jArr = obj.getArrivalSlot();
            x_arr = obj.orbit_database{iArr}(jArr,:).';
        end

        function sigma = getSigma(obj)
            tr = obj.getTransferCfg();
            lt = tr.lowthrust;

            if isfield(lt,'sigma') && ~isempty(lt.sigma)
                sigma = lt.sigma;
            else
                sigma = 1.0;
            end
        end

        function z0 = initialGuess(obj)
            tr = obj.getTransferCfg();
            lt = tr.lowthrust;

            if isfield(lt,'lambda_guess') && ~isempty(lt.lambda_guess)
                lam0 = lt.lambda_guess(:);
                if numel(lam0) ~= 7
                    error('lowthrust.lambda_guess must have 7 elements.');
                end
            else
                lam0 = [
                    -0.25
                     0.75
                     0.35
                    -0.20
                     0.40
                     0.10
                     0.05
                ];
            end

            % Normalize first 6 costates to match transversality convention
            nlam0 = norm(lam0(1:6));
            if nlam0 > 0
                lam0(1:6) = lam0(1:6) / nlam0;
            end

            tf_guess = lt.tf_guess;
            z0 = [lam0; tf_guess];
        end

        function [lb, ub] = bounds(obj)
            tr = obj.getTransferCfg();
            lt = tr.lowthrust;

            lamBnd = 100 * ones(7,1);

            if isfield(lt,'lambda_lb') && ~isempty(lt.lambda_lb)
                lam_lb = lt.lambda_lb(:);
            else
                lam_lb = -lamBnd;
            end

            if isfield(lt,'lambda_ub') && ~isempty(lt.lambda_ub)
                lam_ub = lt.lambda_ub(:);
            else
                lam_ub = lamBnd;
            end

            lb = [lam_lb; lt.tf_lb];
            ub = [lam_ub; lt.tf_ub];
        end

        function r = boundaryResidual(obj, z, x_dep, x_arr)
            try
                [lb, ub] = obj.bounds();

                % Soft guard since fsolve does not support bounds
                if any(z < lb) || any(z > ub)
                    r = 1e6 * ones(8,1);
                    return;
                end

                [~, ~, data] = obj.propagate(z, x_dep, x_arr);

                tr = obj.getTransferCfg();
                lt = tr.lowthrust;

                if isfield(lt,'w_pos_indirect'),   w_pos   = lt.w_pos_indirect;   else, w_pos   = 1; end
                if isfield(lt,'w_vel_indirect'),   w_vel   = lt.w_vel_indirect;   else, w_vel   = 1; end
                if isfield(lt,'w_norm_indirect'),  w_norm  = lt.w_norm_indirect;  else, w_norm  = 1; end
                if isfield(lt,'w_mass_indirect'),  w_mass  = lt.w_mass_indirect;  else, w_mass  = 1; end

                pos_err = data.xf(1:3) - x_arr(1:3);
                vel_err = data.xf(4:6) - x_arr(4:6);

                % Match old solver structure:
                % norm(lambda_rv(tf)) = 1
                costate_norm_cond = norm(data.lambda_f(1:6)) - 1;

                % Free final mass
                mass_trans = data.lambda_f(7);

                r = [
                    sqrt(max(w_pos,0))  * pos_err
                    sqrt(max(w_vel,0))  * vel_err
                    sqrt(max(w_norm,0)) * costate_norm_cond
                    sqrt(max(w_mass,0)) * mass_trans
                ];

                if any(~isfinite(r))
                    r = 1e6 * ones(8,1);
                end
            catch ME
                fprintf(2, 'LowThrustTransferSolver.boundaryResidual failed: %s\n', ME.message);
                r = 1e6 * ones(8,1);
            end
        end

        function [t_out, s_out, data] = propagate(obj, z, x_dep, x_arr)
            tr = obj.getTransferCfg();
            lt = tr.lowthrust;

            Tmax  = lt.Tmax;
            ve    = lt.ve;
            m0    = lt.m0;
            sigma = obj.getSigma();

            lam0  = z(1:7);
            tf    = z(end);

            if tf <= 0
                error('LowThrustTransferSolver:InvalidTF', 'Time of flight must be positive.');
            end

            X0 = [x_dep(:); m0; lam0(:)];

            t_eval = (0:tr.dt:tf).';
            if isempty(t_eval) || t_eval(1) ~= 0
                t_eval = [0; t_eval];
            end
            if t_eval(end) < tf
                t_eval = [t_eval; tf];
            elseif t_eval(end) > tf
                t_eval(end) = tf;
            end

            odeFun = @(t,X) obj.indirectDynamics(t, X, sigma, Tmax, ve);
            [t_all, X_all] = ode45(odeFun, t_eval, X0, obj.ode_opts);

            if isempty(t_all) || any(~isfinite(X_all(:)))
                error('LowThrustTransferSolver:IntegrationFailure', ...
                    'State-costate propagation failed.');
            end

            xf6   = X_all(end,1:6).';
            mf    = X_all(end,7);
            lam_f = X_all(end,8:14).';

            t_out = t_all;
            s_out = X_all(:,1:6);

            data = struct();
            data.residual          = [xf6 - x_arr; norm(lam_f(1:6)) - 1; lam_f(7)];
            data.xf                = xf6;
            data.x_arr             = x_arr;
            data.mass_final        = mf;
            data.lambda_f          = lam_f;
            data.controls          = obj.reconstructControls(X_all, sigma, Tmax);
            data.switchingFunction = obj.reconstructSwitchingFunction(X_all, Tmax, ve);
        end

        function dX = indirectDynamics(obj, ~, X, sigma, Tmax, ve)
            % State
            x  = X(1);
            y  = X(2);
            z  = X(3);
            vx = X(4);
            vy = X(5);
            vz = X(6);
            m  = X(7);

            % Costate
            lx  = X(8);
            ly  = X(9);
            lz  = X(10);
            lvx = X(11);
            lvy = X(12);
            lvz = X(13);
            lm  = X(14);

            mu = obj.mu;
            pi1 = 1 - mu;

            r1 = norm([x + mu, y, z]);
            r2 = norm([x - pi1, y, z]);

            r1_2  = 1 / r1^2;
            r2_2  = 1 / r2^2;
            pr1_3 = pi1 * r1_2 / r1;
            mr2_3 = mu  * r2_2 / r2;
            pr1_5 = pr1_3 * r1_2;
            mr2_5 = mr2_3 * r2_2;

            C1 = pr1_3 + mr2_3;
            C2 = pr1_5 + mr2_5;
            D0 = (x + mu)  * pr1_3 + (x - pi1) * mr2_3;
            D1 = (x + mu)  * pr1_5 + (x - pi1) * mr2_5;
            D2 = (x + mu)^2 * pr1_5 + (x - pi1)^2 * mr2_5;

            lv = [lvx; lvy; lvz];
            nlv = norm(lv);

            if m <= 1e-12 || nlv <= 1e-14 || sigma <= 0 || Tmax <= 0
                aT    = [0;0;0];
                mdot  = 0;
                lmdot = 0;
            else
                uhat = -lv / nlv;
                aT   = (sigma * Tmax / m) * uhat;
                mdot = -sigma * Tmax / ve;

                % lambda_m_dot = -dH/dm
                lmdot = (lv.' * aT) / m;
            end

            dX = [
                vx
                vy
                vz
                2*vy + x - D0 + aT(1)
                (1 - C1)*y - 2*vx + aT(2)
                -C1*z + aT(3)
                mdot
                lvx*(C1 - 3*D2 - 1) - 3*y*lvy*D1 - 3*z*lvz*D1
                -3*y*lvx*D1 + lvy*(C1 - 3*y^2*C2 - 1) - 3*y*z*lvz*C2
                -3*z*lvx*D1 - 3*y*z*lvy*C2 + lvz*(C1 - 3*z^2*C2)
                2*lvy - lx
                -2*lvx - ly
                -lz
                lmdot
            ];
        end

        function U = reconstructControls(~, X_all, sigma, Tmax)
            N = size(X_all,1);
            U = zeros(N,3);

            if sigma <= 0 || Tmax <= 0
                return;
            end

            for k = 1:N
                lv = X_all(k,11:13).';
                nlv = norm(lv);
                if nlv > 1e-14
                    U(k,:) = (-lv / nlv).';
                end
            end
        end

        function S = reconstructSwitchingFunction(~, X_all, Tmax, ve)
            N = size(X_all,1);
            S = zeros(N,1);

            for k = 1:N
                m  = X_all(k,7);
                lv = X_all(k,11:13).';
                lm = X_all(k,14);

                if m <= 1e-12
                    S(k) = NaN;
                else
                    S(k) = 1 - Tmax * (norm(lv)/m + lm/ve);
                end
            end
        end
    end
end