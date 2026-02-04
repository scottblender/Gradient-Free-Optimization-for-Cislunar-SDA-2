function J_total = compute_cost(s_act, s_ekf, cov, stabilities_vec, opt_flag, solverName, dq)
    % COMPUTE_COST - Computes cost and logs components in parallel-safe way
    %
    % Inputs:
    %   s_act - N_timesteps x 6 state matrix (truth)
    %   s_ekf - N_timesteps x 6 estimated state matrix
    %   cov - N_timesteps x 6 x 6 covariance matrix
    %   opt_flag - 'SOO' or 'MOO' for single or multi-objective
    %   norm_refs - normalization constants [J1, J2, J3]
    %   solverName - string, name of optimization algorithm
    %
    % Output:
    %   J_out - scalar (SOO) or vector (MOO)
    
    % --- Defaults --- %
    if nargin < 6
        solverName = "unknown";
    end
    
    weights = [1, 0.1, 10];  % component weights

    % --- State RMSE --- %
    err = s_act - s_ekf;
    rmse = sqrt(mean(sum(err.^2,2)));
    J_1 = weights(1)*log(rmse);
    
    % --- Covariance Determinant Terms --- %
    N = size(cov, 1);
    det_vals = zeros(N, 1);
    for k = 1:N
        % Extract the 6x6 matrix for time step k
        P_k = squeeze(cov(k, :, :));
        det_vals(k) = det(P_k);
    end
    det_term = mean(log(det_vals));
    J_2 = weights(2)*det_term;

    % --- Stability/Station-Keeping Term --- %
    stability_term = stabilities_vec;
    J_3 = weights(3)*mean(stability_term);
    
    % --- Scalar or vectorized total cost --- %
    switch upper(opt_flag)
        case "SOO"
            J_total = J_1 + J_2 + J_3;
        case "MOO"
            J_total = [J_1; J_2; J_3];
        otherwise
            error('opt_flag must be either "SOO" or "MOO"');
    end
    
    % log data
    if ~isempty(dq)
        % Create the data row: [Iter(placeholder), Solver, J1, J2, J3, Total]
        logRow = [string(solverName), J_1, J_2, J_3, J_total(:)'];
        send(dq, logRow);
    end
end
