function selection = select_reviewer2_representative_runs(report,study)
%SELECT_REVIEWER2_REPRESENTATIVE_RUNS Pick typical, not best-seed, geometries.
%
% Geometry cannot be averaged across discrete orbit/slot decisions. For a
% qualitative geometry figure, choose the saved realization whose final
% objective is closest to the 20-run group mean. Statistical comparisons in
% the paper must still use the group mean +/- sample standard deviation.
%
% study = "comparison", "baseline", or "objective_screening".

study = lower(string(study));
assert(isstruct(report) && isfield(report,'results') && isfield(report,'runMetrics'), ...
    'A processed Reviewer-2 report with results/runMetrics is required.');

switch study
    case "comparison"
        assert(isfield(report,'summary'),'Comparison report requires summary metadata.');
        selection = comparison_selection(report.results,report.runMetrics,report.summary);
    case "baseline"
        selection = baseline_selection(report.results,report.runMetrics);
    case "objective_screening"
        selection = objective_selection(report.results,report.runMetrics);
    otherwise
        error('Study:UnknownRepresentativeStudy','Unknown study selector: %s',study);
end
end


function selection = comparison_selection(R,M,S)
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
optimizers = ["GA","PSO","ABC","ACO"];
n = numel(missions)*numel(optimizers);
selection = empty_selection(n);
row = 0;
for mission = missions
    for optimizer = optimizers
        row = row+1;
        group = R(R.Mission == mission & R.Optimizer == optimizer,:);
        summaryRow = S(S.mission == mission & S.optimizer == optimizer,:);
        assert(height(group) == 1 && height(summaryRow) == 1, ...
            'Missing comparison aggregate/summary group.');
        candidates = M(M.comparison_key == summaryRow.comparison_key & ...
            M.optimizer == optimizer,:);
        selection = fill_row(selection,row,mission,lower(optimizer),optimizer, ...
            candidates,group.BestJMean,group.BestJStd);
    end
end
end


function selection = baseline_selection(R,M)
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
observerCounts = [3 5 7 10];
n = numel(missions)*numel(observerCounts);
selection = empty_selection(n);
row = 0;
for mission = missions
    for nObs = observerCounts
        row = row+1;
        group = R(R.Mission == mission & R.Measurement == "ANGLES_ONLY" & ...
            R.NumObservers == nObs & R.NPeriods == 1,:);
        assert(height(group) == 1,'Missing AO/p1 baseline aggregate group.');
        candidates = M(M.comparison_key == group.ComparisonKey,:);
        selection = fill_row(selection,row,mission,"o"+string(nObs), ...
            string(nObs)+" observers",candidates,group.BestJMean,group.BestJStd);
    end
end
end


function selection = objective_selection(R,M)
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
configs = ["combined_on","combined_off","j1_only","j2_only","j3_only"];
n = numel(missions)*numel(configs);
selection = empty_selection(n);
row = 0;
for mission = missions
    for config = configs
        row = row+1;
        group = R(R.Mission == mission & string(R.Configuration) == config,:);
        assert(height(group) == 1,'Missing objective/screening aggregate group.');
        candidates = M(M.comparison_key == group.ComparisonKey,:);
        selection = fill_row(selection,row,mission,config,configuration_label(config), ...
            candidates,group.BestJMean,group.BestJStd);
    end
end
end


function T = empty_selection(n)
T = table(strings(n,1),strings(n,1),strings(n,1),strings(n,1), ...
    nan(n,1),nan(n,1),nan(n,1),nan(n,1),nan(n,1), ...
    'VariableNames',{'Mission','PanelKey','PanelLabel','RunFile', ...
    'RepresentativeObjective','GroupMeanObjective','GroupStdObjective', ...
    'RepresentativeSeed','ObjectiveDeviationFromMean'});
end


function T = fill_row(T,row,mission,key,label,candidates,mu,sigma)
assert(height(candidates) == 20, ...
    'Representative geometry requires the complete 20-run group.');
assert(all(isfinite(candidates.bestJ)),'Representative-run objectives must be finite.');
[deviation,idx] = min(abs(double(candidates.bestJ)-double(mu)));
T.Mission(row) = string(mission);
T.PanelKey(row) = string(key);
T.PanelLabel(row) = string(label);
T.RunFile(row) = string(candidates.run_file(idx));
T.RepresentativeObjective(row) = double(candidates.bestJ(idx));
T.GroupMeanObjective(row) = double(mu);
T.GroupStdObjective(row) = double(sigma);
T.RepresentativeSeed(row) = double(candidates.seed(idx));
T.ObjectiveDeviationFromMean(row) = deviation;
end


function label = configuration_label(config)
switch string(config)
    case "combined_on", label = "Combined, screening ON";
    case "combined_off", label = "Combined, screening OFF";
    case "j1_only", label = "J_1 only";
    case "j2_only", label = "J_2 only";
    case "j3_only", label = "J_3 only";
    otherwise, label = string(config);
end
end
