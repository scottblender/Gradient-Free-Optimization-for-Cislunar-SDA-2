function J_out = compute_cost(s_act, s_ekf, cov, opt_flag, norm_refs, solverName)
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
        norm_refs = [1,1,1];
    end
    if nargin < 6
        solverName = "unknown";
    end
    
    weights = [1, 1, 1];  % component weights
    
    % --- Setup parallel-safe logging --- %
    persistent dq iterCounter
    if isempty(dq)
        dq = parallel.pool.DataQueue;
        afterEach(dq, @logCallback);
        assignin('base','costComponentLog', []); % initialize log
        iterCounter = 0;
    end
    
    % --- Increment iteration --- %
    iterCounter = iterCounter + 1;
    
    % --- State RMSE --- %
    err = s_act - s_ekf;
    rmse = sqrt(mean(sum(err.^2,2)));
    J_1 = (weights(1)/norm_refs(1))*rmse;
    
    % --- Covariance Trace and Determinant Terms --- %
    cov_perm = permute(cov,[2 3 1]);
    tr_term = mean(squeeze(trace(cov_perm)));
    J_2 = (weights(2)/norm_refs(2))*tr_term;
    
    det_vals = squeeze(det(cov_perm));
    det_vals(det_vals <= 0) = eps;
    det_term = mean(log(det_vals));
    J_3 = (weights(3)/norm_refs(3))*det_term;
    
    % --- Scalar or vectorized total cost --- %
    switch upper(opt_flag)
        case "SOO"
            J_total = J_1 + J_2 + J_3;
        case "MOO"
            J_total = [J_1; J_2; J_3];
        otherwise
            error('opt_flag must be either "SOO" or "MOO"');
    end
    
    % --- Send to DataQueue for logging --- %
    logRow = [iterCounter, string(solverName), J_1, J_2, J_3, J_total(:)'];
    send(dq, logRow);
    
    % --- Return total cost --- %
    J_out = J_total;
    
    % --- Nested callback --- %
    function logCallback(data)
        log = evalin('base','costComponentLog');
        log = [log; data];
        assignin('base','costComponentLog', log);
    end
end
