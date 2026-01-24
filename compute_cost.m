function J_total = compute_cost(s_act, s_ekf, cov, opt_flag, solverName, dq)
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
    if nargin < 5
        solverName = "unknown";
    end
    
    weights = [1, 1, 0.1];  % component weights

    % --- State RMSE --- %
    err = s_act - s_ekf;
    rmse = sqrt(mean(sum(err.^2,2)));
    J_1 = weights(1)*log(rmse);
    
    % --- Covariance Trace and Determinant Terms --- %
   % Sum diagonal elements (k,1,1) + (k,2,2) ... for all k
    tr_vals = cov(:,1,1) + cov(:,2,2) + cov(:,3,3) + ...
              cov(:,4,4) + cov(:,5,5) + cov(:,6,6);
    tr_term = mean(tr_vals);
    J_2 = weights(2)*log(tr_term);
    N = size(cov, 1);
    det_vals = zeros(N, 1);
    for k = 1:N
        % Extract the 6x6 matrix for time step k
        P_k = squeeze(cov(k, :, :));
        det_vals(k) = det(P_k);
    end
    det_term = mean(log(det_vals));
    J_3 = weights(3)*det_term;
    
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
    if nargin >= 6 && ~isempty(dq)
        % Create the data row: [Iter(placeholder), Solver, J1, J2, J3, Total]
        logRow = [string(solverName), J_1, J_2, J_3, J_total(:)'];
        send(dq, logRow);
    end
end
