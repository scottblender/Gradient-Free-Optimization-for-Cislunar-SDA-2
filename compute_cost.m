function [J_total, J_1, J_2, J_3] = compute_cost(s_act, s_ekf, cov, stabilities_vec, opt_flag, costFlags, costCfg)

    % ---------------- defaults ----------------
    if nargin < 6 || isempty(costFlags)
        costFlags = struct('J1', true, 'J2', true, 'J3', true);
    end
    if ~isfield(costFlags,'J1'), costFlags.J1 = true; end
    if ~isfield(costFlags,'J2'), costFlags.J2 = true; end
    if ~isfield(costFlags,'J3'), costFlags.J3 = true; end

    if nargin < 7 || isempty(costCfg)
        error('compute_cost requires costCfg with mission-specific thresholds.');
    end

    if ~isfield(costCfg,'pos_rmse_acc'), error('Missing costCfg.pos_rmse_acc'); end
    if ~isfield(costCfg,'vel_rmse_acc'), error('Missing costCfg.vel_rmse_acc'); end
    if ~isfield(costCfg,'sigma_pos_acc'), error('Missing costCfg.sigma_pos_acc'); end
    if ~isfield(costCfg,'sigma_vel_acc'), error('Missing costCfg.sigma_vel_acc'); end
    if ~isfield(costCfg,'stability_acc') || isempty(costCfg.stability_acc)
        costCfg.stability_acc = 1.0;
    end
    if ~isfield(costCfg,'weights') || isempty(costCfg.weights)
        costCfg.weights = [1, 1, 1];
    end

    weights = costCfg.weights;

    % ---------------- J1: RMSE ----------------
    err = s_act - s_ekf;

    pos_err = err(:,1:3);
    vel_err = err(:,4:6);

    rmse_pos = sqrt(mean(sum(pos_err.^2,2)));
    rmse_vel = sqrt(mean(sum(vel_err.^2,2)));

    rmse_pos = max(rmse_pos, realmin);
    rmse_vel = max(rmse_vel, realmin);

    J1_raw = weights(1) * ( ...
        rmse_pos / max(costCfg.pos_rmse_acc, realmin) + ...
        rmse_vel / max(costCfg.vel_rmse_acc, realmin) );

    % ---------------- J2: determinant ----------------
    N = size(cov,1);
    det_pos_vals = zeros(N,1);
    det_vel_vals = zeros(N,1);

    for k = 1:N
        P_k = squeeze(cov(k,:,:));
        P_k = 0.5 * (P_k + P_k.');

        P_pos = P_k(1:3,1:3);
        P_vel = P_k(4:6,4:6);

        det_pos_vals(k) = max(det(P_pos), realmin);
        det_vel_vals(k) = max(det(P_vel), realmin);
    end

    eff_sigma_pos = mean(det_pos_vals.^(1/6));
    eff_sigma_vel = mean(det_vel_vals.^(1/6));

    eff_sigma_pos = max(eff_sigma_pos, realmin);
    eff_sigma_vel = max(eff_sigma_vel, realmin);

    J2_raw = weights(2) * ( ...
        eff_sigma_pos / max(costCfg.sigma_pos_acc, realmin) + ...
        eff_sigma_vel / max(costCfg.sigma_vel_acc, realmin) );

    % ---------------- J3: stability ----------------
    J3_raw = weights(3) * (mean(stabilities_vec) / max(costCfg.stability_acc, realmin));

    % ---------------- apply toggles ----------------
    J_1 = double(logical(costFlags.J1)) * J1_raw;
    J_2 = double(logical(costFlags.J2)) * J2_raw;
    J_3 = double(logical(costFlags.J3)) * J3_raw;

    % ---------------- total ----------------
    switch upper(string(opt_flag))
        case "SOO"
            J_total = J_1 + J_2 + J_3;
        case "MOO"
            J_total = [J_1; J_2; J_3];
        otherwise
            error('opt_flag must be either "SOO" or "MOO"');
    end
end