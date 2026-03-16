classdef LowThrustTransferSolver
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
        function obj = LowThrustTransferSolver(cfg, T1, orbit_database, times, states, mu, ode_opts)
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
            tr = obj.getTransferCfg();
            lt = tr.lowthrust;

            s_dep = obj.getDepartureState();

            z0 = obj.initialGuess();
            [lb, ub] = obj.bounds();

            resfun = @(z) obj.residualVector(z, s_dep);

            opts = optimoptions('lsqnonlin', ...
                'Display', 'iter', ...
                'MaxIterations', 300, ...
                'MaxFunctionEvaluations', 5e4, ...
                'StepTolerance', 1e-10, ...
                'FunctionTolerance', 1e-10, ...
                'OptimalityTolerance', 1e-10);

            [z_best, resnorm, residual, exitflag, output] = lsqnonlin( ...
                resfun, z0, lb, ub, opts);

            [t_target, s_target, finalData] = obj.propagate(z_best, s_dep);

            info = struct();
            info.type              = "LOW_THRUST_TRANSFER";
            info.builder           = "LowThrustTransferSolver";
            info.method            = "DIRECT_SINGLE_SHOOTING_LSQNONLIN";
            info.depOrbitIndex     = tr.depOrbitIndex;
            info.depSlot           = tr.depSlot;
            info.arrOrbitIndex     = tr.arrOrbitIndex;
            info.dt                = tr.dt;
            info.Nseg              = lt.Nseg;
            info.Tmax              = lt.Tmax;
            info.ve                = lt.ve;
            info.m0                = lt.m0;
            info.tf                = z_best(end-1);
            info.phase             = z_best(end);
            info.resnorm           = resnorm;
            info.finalResidual     = finalData.residual;
            info.finalResidualNorm = norm(finalData.residual);
            info.exitflag          = exitflag;
            info.output            = output;
            info.xf                = finalData.xf;
            info.x_arr             = finalData.x_arr;
            info.mass_final        = finalData.mass_final;
            info.controls          = finalData.controls;
            info.residualVector    = residual;
        end
    end

    methods (Access = private)
        function tr = getTransferCfg(obj)
            tr = obj.cfg;
            if isfield(tr, 'transfer')
                tr = tr.transfer;
            end
        end

        function s_dep = getDepartureState(obj)
            tr = obj.getTransferCfg();
            iDep = tr.depOrbitIndex;
            jDep = tr.depSlot;
            s_dep = obj.orbit_database{iDep}(jDep,:).';
        end

        function z0 = initialGuess(obj)
            tr = obj.getTransferCfg();
            lt = tr.lowthrust;

            Nseg = lt.Nseg;

            % z = [alpha_1 beta_1 ... alpha_N beta_N tf phase]
            ang = zeros(2*Nseg,1);

            if isfield(lt,'angles_guess') && ~isempty(lt.angles_guess)
                ag = lt.angles_guess(:);
                if numel(ag) == 2*Nseg
                    ang = ag;
                end
            else
                % Mildly nontrivial default guess to avoid perfectly symmetric zero-control seed
                for k = 1:Nseg
                    ang(2*k-1) = 0.05 * (k-1);  % alpha_k
                    ang(2*k)   = 0.0;           % beta_k
                end
            end

            tf_guess = lt.tf_guess;

            if isfield(lt,'phase_guess') && ~isempty(lt.phase_guess)
                phase_guess = lt.phase_guess;
            else
                phase_guess = 0.5 * obj.arrPeriod;
            end

            z0 = [ang; tf_guess; phase_guess];
        end

        function [lb, ub] = bounds(obj)
            tr = obj.getTransferCfg();
            lt = tr.lowthrust;

            Nseg = lt.Nseg;

            alpha_lb = -pi * ones(Nseg,1);
            alpha_ub =  pi * ones(Nseg,1);

            beta_lb  = -pi/2 * ones(Nseg,1);
            beta_ub  =  pi/2 * ones(Nseg,1);

            lb_ang = zeros(2*Nseg,1);
            ub_ang = zeros(2*Nseg,1);

            lb_ang(1:2:end) = alpha_lb;
            lb_ang(2:2:end) = beta_lb;
            ub_ang(1:2:end) = alpha_ub;
            ub_ang(2:2:end) = beta_ub;

            lb = [lb_ang; lt.tf_lb; 0];
            ub = [ub_ang; lt.tf_ub; obj.arrPeriod];
        end

        function r = residualVector(obj, z, s_dep)
            tr = obj.getTransferCfg();
            lt = tr.lowthrust;

            if isfield(lt,'w_pos'),     w_pos  = lt.w_pos;     else, w_pos  = 1e4; end
            if isfield(lt,'w_vel'),     w_vel  = lt.w_vel;     else, w_vel  = 1e3; end
            if isfield(lt,'w_tf'),      w_tf   = lt.w_tf;      else, w_tf   = 1e-2; end
            if isfield(lt,'w_smooth'),  w_sm1  = lt.w_smooth;  else, w_sm1  = 1e-2; end
            if isfield(lt,'w_smooth2'), w_sm2  = lt.w_smooth2; else, w_sm2  = 1e-1; end
            if isfield(lt,'w_ctrl'),    w_ctrl = lt.w_ctrl;    else, w_ctrl = 1e-4; end

            Nseg = lt.Nseg;

            try
                [~, ~, data] = obj.propagate(z, s_dep);

                pos_err = data.xf(1:3) - data.x_arr(1:3);
                vel_err = data.xf(4:6) - data.x_arr(4:6);
                tf = z(end-1);

                ang = z(1:2*Nseg);
                alpha = ang(1:2:end);
                beta  = ang(2:2:end);

                % Build thrust direction vectors at segment nodes
                U = zeros(Nseg,3);
                for k = 1:Nseg
                    U(k,:) = [cos(beta(k))*cos(alpha(k)), ...
                              cos(beta(k))*sin(alpha(k)), ...
                              sin(beta(k))];
                end

                % Smoothness on Cartesian thrust directions
                dU  = diff(U,1,1);
                ddU = diff(U,2,1);

                r = [
                    sqrt(max(w_pos,0))  * pos_err
                    sqrt(max(w_vel,0))  * vel_err
                    sqrt(max(w_tf,0))   * tf
                    sqrt(max(w_sm1,0))  * dU(:)
                    sqrt(max(w_sm2,0))  * ddU(:)
                    sqrt(max(w_ctrl,0)) * alpha
                    sqrt(max(w_ctrl,0)) * beta
                ];

                if any(~isfinite(r))
                    r = 1e6 * ones(size(r));
                end
            catch ME
                fprintf(2, 'LowThrustTransferSolver.residualVector failed: %s\n', ME.message);
                nRes = 3 + 3 + 1 + 3*max(Nseg-1,0) + 3*max(Nseg-2,0) + Nseg + Nseg;
                r = 1e6 * ones(nRes,1);
            end
        end

        function [t_out, s_out, data] = propagate(obj, z, s_dep)
            tr = obj.getTransferCfg();
            lt = tr.lowthrust;

            Nseg  = lt.Nseg;
            Tmax  = lt.Tmax;
            ve    = lt.ve;
            m0    = lt.m0;
            tf    = z(end-1);
            phase = z(end);

            if tf <= 0
                error('LowThrustTransferSolver:InvalidTF', 'Time of flight must be positive.');
            end

            ang = z(1:2*Nseg);
            alpha = ang(1:2:end);
            beta  = ang(2:2:end);

            U = zeros(Nseg,3);
            for k = 1:Nseg
                U(k,:) = [cos(beta(k))*cos(alpha(k)), ...
                          cos(beta(k))*sin(alpha(k)), ...
                          sin(beta(k))];
            end

            x0 = [s_dep(:); m0];

            segEdges = linspace(0, tf, Nseg+1);
            t_all = 0;
            x_all = x0.';
            xk = x0;

            for k = 1:Nseg
                tk0 = segEdges(k);
                tkf = segEdges(k+1);
                uk  = U(k,:).';

                odeFun = @(t,x) obj.lowThrustDynamics(t, x, uk, Tmax, ve);

                tk_eval = (tk0:tr.dt:tkf).';
                if isempty(tk_eval) || tk_eval(1) ~= tk0
                    tk_eval = [tk0; tk_eval];
                end
                if tk_eval(end) < tkf
                    tk_eval = [tk_eval; tkf];
                elseif tk_eval(end) > tkf
                    tk_eval(end) = tkf;
                end

                [t_seg, x_seg] = ode45(odeFun, tk_eval, xk, obj.ode_opts);

                if k == 1
                    t_all = t_seg;
                    x_all = x_seg;
                else
                    t_all = [t_all; t_seg(2:end)]; %#ok<AGROW>
                    x_all = [x_all; x_seg(2:end,:)]; %#ok<AGROW>
                end

                xk = x_seg(end,:).';

                if ~all(isfinite(xk))
                    error('LowThrustTransferSolver:IntegrationFailure', ...
                        'Propagation produced non-finite state.');
                end
            end

            x_arr = obj.evalArrivalState(phase);
            xf6   = x_all(end,1:6).';
            resid = xf6 - x_arr;

            t_out = t_all;
            s_out = x_all(:,1:6);

            data = struct();
            data.residual   = resid;
            data.phase      = phase;
            data.xf         = xf6;
            data.x_arr      = x_arr;
            data.mass_final = x_all(end,7);
            data.controls   = U;
        end

        function xdot = lowThrustDynamics(obj, ~, x, u, Tmax, ve)
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
                aT = (Tmax / m) * u;
                mdot = -Tmax / ve;
            end

            xdot = [
                vx
                vy
                vz
                ax_g + aT(1)
                ay_g + aT(2)
                az_g + aT(3)
                mdot
            ];
        end

        function x_arr = evalArrivalState(obj, phase)
            phaseWrapped = mod(phase, obj.arrPeriod);
            x_arr = obj.arrInterp(phaseWrapped).';
        end

        function [F, period] = buildArrivalInterpolant(obj)
            tr = obj.getTransferCfg();
            iArr = tr.arrOrbitIndex;

            t_raw = obj.times{iArr}(:);
            s_raw = obj.states{iArr};

            [t_unique, idx_u] = unique(t_raw);
            s_unique = s_raw(idx_u, :);

            period = obj.T1.("Period (TU) ")(iArr);

            if abs(t_unique(1)) > 1e-12
                error('Arrival orbit time history must start at t = 0.');
            end

            if abs(t_unique(end) - period) > 1e-10 || any(abs(s_unique(1,:) - s_unique(end,:)) > 1e-8)
                t_aug = [t_unique; period];
                s_aug = [s_unique; s_unique(1,:)];
            else
                t_aug = t_unique;
                s_aug = s_unique;
            end

            F = griddedInterpolant(t_aug, s_aug, 'pchip');
        end
    end
end