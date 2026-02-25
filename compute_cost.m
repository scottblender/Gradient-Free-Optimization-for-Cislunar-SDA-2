function [J_total, J_1, J_2, J_3] = compute_cost(s_act, s_ekf, cov, stabilities_vec, opt_flag, costFlags)
    % ---------------- defaults ----------------
    if nargin < 6 || isempty(costFlags)
        costFlags = struct('J1', true, 'J2', true, 'J3', true);
    end
    if ~isfield(costFlags,'J1'), costFlags.J1 = true; end
    if ~isfield(costFlags,'J2'), costFlags.J2 = true; end
    if ~isfield(costFlags,'J3'), costFlags.J3 = true; end

    weights = [1, 0.1, 10];  % component weights

    % --- J1: State RMSE ---
    err = s_act - s_ekf;
    rmse = sqrt(mean(sum(err.^2,2)));
    J1_raw = weights(1)*log(rmse);

    % --- J2: Covariance determinant term ---
    N = size(cov, 1);
    det_vals = zeros(N, 1);
    for k = 1:N
        P_k = squeeze(cov(k, :, :));
        d = det(P_k);
        % guard: det can go <= 0 numerically; keep log real
        det_vals(k) = max(d, realmin);
    end
    det_term = mean(log(det_vals));
    J2_raw = weights(2)*det_term;

    % --- J3: Stability term ---
    J3_raw = weights(3)*mean(stabilities_vec);

    % Apply toggles (inactive components become constant 0)
    J_1 = double(logical(costFlags.J1)) * J1_raw;
    J_2 = double(logical(costFlags.J2)) * J2_raw;
    J_3 = double(logical(costFlags.J3)) * J3_raw;

    % Total cost format
    switch upper(string(opt_flag))
        case "SOO"
            J_total = J_1 + J_2 + J_3;
        case "MOO"
            % Keep fixed dimension for solvers; inactive objectives are constant 0.
            J_total = [J_1; J_2; J_3];
        otherwise
            error('opt_flag must be either "SOO" or "MOO"');
    end
end