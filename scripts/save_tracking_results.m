function runState = save_tracking_results(DataDir, runState, ...
    t, truth, estimate, covariance, available, screeningCount, observers)
% Save one selected design's diagnostics, separately from search FE.
% covariance(k,:,:) uses the original normalized state units.
s = runState.settings;
t = t(:);
N = numel(t);
nObs = height(observers);
LU = s.LU;
VU = s.LU/s.TU;

assert(N >= 2 && all(diff(t) > 0), 'Invalid tracking time grid.');
assert(isequal(size(truth),[N 6]) && isequal(size(estimate),[N 6]), ...
    'Tracking state dimensions do not match.');
assert(isequal(size(covariance),[N 6 6]), 'Invalid covariance dimensions.');
assert(all(isfinite(truth(:))) && all(isfinite(estimate(:))) && ...
    all(isfinite(covariance(:))), 'Nonfinite tracking data.');
available = available(:);
assert(numel(available) == N && all(isfinite(available(2:end))) && ...
    all(available(2:end) >= 0 & available(2:end) <= nObs) && ...
    all(available(2:end) == round(available(2:end))), ...
    'Invalid available-observer counts.');

% The current EKF does not screen its initialization sample.
available(1) = NaN;
err = estimate - truth;
posError = vecnorm(err(:,1:3),2,2)*LU;
velError = vecnorm(err(:,4:6),2,2)*VU;

% Use compute_cost itself so metrics match the optimization definition.
allFlags = struct('J1',true,'J2',true,'J3',true);
[~, j1, j2, j3] = compute_cost(truth, estimate, covariance, ...
    observers.stability_index, 'SOO', allFlags, s.cost);
[Jcheck, ~, ~, ~] = compute_cost(truth, estimate, covariance, ...
    observers.stability_index, 'SOO', s.costFlags, s.cost);
assert(isfinite(Jcheck) && abs(Jcheck-runState.bestJ) <= ...
    1e-9*max(1,abs(runState.bestJ)), ...
    'Saved best design does not reproduce its optimization cost.');

effPos = zeros(N,1);
effVel = zeros(N,1);
for k = 1:N
    P = squeeze(covariance(k,:,:));
    P = (P+P')/2;
    effPos(k) = max(det(P(1:3,1:3)),realmin)^(1/6);
    effVel(k) = max(det(P(4:6,4:6)),realmin)^(1/6);
end

m = struct();
m.rmse_pos_km = sqrt(mean(posError.^2));
m.rmse_vel_kms = sqrt(mean(velError.^2));
m.peak_pos_error_km = max(posError);
m.peak_vel_error_kms = max(velError);
m.mean_effective_sigma_pos_km = mean(effPos)*LU;
m.mean_effective_sigma_vel_kms = mean(effVel)*VU;
m.mean_stability = mean(observers.stability_index);
m.J1_normalized = (m.rmse_pos_km/LU)/s.cost.pos_rmse_acc + ...
    (m.rmse_vel_kms/VU)/s.cost.vel_rmse_acc;
m.J2_normalized = mean(effPos)/s.cost.sigma_pos_acc + ...
    mean(effVel)/s.cost.sigma_vel_acc;
m.J3_normalized = m.mean_stability/s.cost.stability_acc;
% Weighted components BEFORE toggles, including disabled objectives.
m.J1_weighted = j1;
m.J2_weighted = j2;
m.J3_weighted = j3;
m.J_recheck = Jcheck;
m.screening_count = screeningCount;

% Sample-based coverage; no fabricated continuous visibility intervals.
m.coverage_epoch_fraction = mean(available(2:end) > 0);
m.mean_available_observers = mean(available(2:end));
m.available_pair_fraction = mean(available(2:end))/nObs;

tracking = struct();
tracking.t_TU = t;
tracking.truth = truth;
tracking.estimate = estimate;
tracking.covariance = covariance;
tracking.availableObsCount = available;
tracking.units = "State: LU and LU/TU; time: TU";
tracking.coverageConvention = "Epoch samples 2:end; initial count unknown";
runState.metrics = m;
runState.observers = observers;
runState.validationStatus = "passed";
runState.validationEvaluations = 1;
% Write tracking first. A completed metadata record must reference saved data.
save(fullfile(DataDir,'tracking_data.mat'), 'tracking', '-v7');
save(fullfile(DataDir,'optimization_run.mat'), 'runState', '-v7');
end
