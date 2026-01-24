function J = objective_wrapper(inputs, orbit_database, s_lg, t_lg, P0, Q, R, mu, opt_flag, solverName, dq)
    try
        if istable(inputs)
            x = table2array(inputs);
        else 
            x = inputs;
        end
    
        x = round(x); % in case any optimizer accidentally passes a double
    
        % determine which orbits and slots are inputted
        orbit_indices = x(1:2:end);
        slot_indices = x(2:2:end);
    
        % orbit initial conditions
        num_obs = length(orbit_indices);
        observer_ICs = zeros(num_obs,6);
    
        for k=1:num_obs
            o_idx = orbit_indices(k);
            s_idx = slot_indices(k);
            
            o_idx = max(1, min(o_idx, length(orbit_database)));
            s_idx = max(1, min(s_idx, size(orbit_database{1}, 1)));
            
            observer_ICs(k, :) = orbit_database{o_idx}(s_idx, :);
        end
    
        % run EKF
        [s_est, cov_hist] = cr3bp_ekf(observer_ICs, s_lg, t_lg, P0, Q, R, mu);
    
        J = compute_cost(s_lg, s_est, cov_hist, opt_flag, solverName, dq);
    catch ME
       if strcmp(opt_mode, 'MOO')
            J = [1e6, 1e6, 1e6]; % Penalty for all 3 objectives
       else
            J = 1e6;
       end
    end

end