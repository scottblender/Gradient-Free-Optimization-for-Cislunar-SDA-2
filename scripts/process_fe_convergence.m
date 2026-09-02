function [summary,inventory] = process_fe_convergence( ...
    root,studyID,seeds,budget,makePlots,optimizers)
% Process schema-v2 results. Save DATA ONLY; optional display-only previews.
% Example:
% Comparison:
% process_fe_convergence(comparisonRoot,"reviewer2_comparison_v1",0:19,6000,false)
% GA baseline:
% process_fe_convergence(baselineRoot,"reviewer2_baseline_v1",0:19,6000,false,"GA")
projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
paths = setup_project();
if nargin < 1 || isempty(root), root = fullfile(paths.runs,'COMPARISON'); end
if nargin < 2 || isempty(studyID), studyID = "reviewer2_comparison_v1"; end
if nargin < 3 || isempty(seeds), seeds = 0:19; end
if nargin < 4 || isempty(budget), budget = 6000; end
if nargin < 5, makePlots = false; end
if nargin < 6, optimizers = ["GA","PSO","BAYESIAN","ABC","ACO"]; end
seeds = seeds(:)';
optimizers = upper(string(optimizers(:)'));
validateattributes(seeds,{'numeric'},{'integer','nonnegative','finite','nonempty'});
validateattributes(budget,{'numeric'},{'scalar','integer','positive','finite'});
assert(numel(unique(seeds)) == numel(seeds),'Duplicate expected seeds.');
assert(numel(unique(optimizers)) == numel(optimizers),'Duplicate optimizers.');

files = dir(fullfile(root,'**','optimization_run.mat'));
assert(~isempty(files),'No optimization_run.mat files under %s.',root);
n = numel(files);
inventory = table(strings(n,1),strings(n,1),strings(n,1),nan(n,1), ...
    false(n,1),strings(n,1),'VariableNames', ...
    {'run_file','comparison_key','optimizer','seed','valid','reason'});
runs = cell(n,1);

for k = 1:n
    file = fullfile(files(k).folder,files(k).name);
    inventory.run_file(k) = string(file);
    try
        S = load(file,'runState');
        r = S.runState;
        assert(isfield(r,'schemaVersion') && r.schemaVersion == 2, ...
            'Older or unsupported result format.');
        assert(string(r.studyID) == string(studyID),'Different study ID.');
        assert(r.maxEvaluations == budget,'Different FE budget.');
        assert(ismember(string(r.optimizer),optimizers) && ...
            ismember(r.optimizerSeed,seeds),'Optimizer/seed not requested.');

        inventory.comparison_key(k) = string(r.comparisonKey);
        inventory.optimizer(k) = string(r.optimizer);
        inventory.seed(k) = r.optimizerSeed;
        assert(study_hash(r.comparison) == string(r.comparisonKey) && ...
            isequaln(r.settings,r.comparison.settings) && ...
            r.comparison.budget == budget, 'Inconsistent comparison metadata.');
        assert(string(r.status) == "completed" && ...
            string(r.termination) == "budget_reached", ...
            'Run failed, stopped early, or exceeded the requested budget.');
        assert(string(r.validationStatus) == "passed" && ...
            r.objectiveErrorCount == 0,'Validation or objective evaluation failed.');
        assert(isfile(fullfile(files(k).folder,'tracking_data.mat')), ...
            'Missing final tracking data.');
        assert(r.nEvaluations == budget,'Reported search FE does not match budget.');
        assert(isfinite(r.solverFunctionEvaluations) && ...
            r.solverFunctionEvaluations >= budget, 'Invalid solver FE total.');
        expectedPostSearch = double(string(r.optimizer) == "GA");
        assert(isfield(r,'postSearchFunctionEvaluations') && ...
            r.postSearchFunctionEvaluations == expectedPostSearch && ...
            r.solverFunctionEvaluations == budget+expectedPostSearch, ...
            'Unexpected native solver-call pattern.');

        H = r.history;
        assert(istable(H) && all(ismember({'fe','bestJ'},H.Properties.VariableNames)) ...
            && ~isempty(H),'Invalid history format.');
        assert(all(isfinite(H.fe)) && all(H.fe > 0 & H.fe == round(H.fe)) && ...
            all(diff(H.fe) > 0) && H.fe(end) == budget, ...
            'FE history must increase and end at the prescribed budget.');
        assert(all(isfinite(H.bestJ)) && ...
            all(diff(H.bestJ) <= 1e-12*max(1,abs(H.bestJ(1:end-1)))), ...
            'History is not finite and best-so-far. No automatic repair applied.');
        assert(isfinite(r.bestJ) && ...
            abs(H.bestJ(end)-r.bestJ) <= 1e-9*max(1,abs(r.bestJ)), ...
            'Final history value does not match saved best cost.');
        assert(isfield(r,'metrics') && ...
            abs(r.metrics.J_recheck-r.bestJ) <= 1e-9*max(1,abs(r.bestJ)), ...
            'Saved design does not reproduce its cost.');

        inventory.valid(k) = true;
        inventory.reason(k) = "valid";
        runs{k} = r;
    catch ME
        inventory.reason(k) = string(ME.message);
    end
end

% Reject duplicate identities; never silently pick the first or best run.
keys = unique(inventory.comparison_key(inventory.comparison_key ~= ""));
for key = keys'
    for opt = optimizers
        for seed = seeds
            idx = find(inventory.comparison_key == key & ...
                inventory.optimizer == opt & inventory.seed == seed);
            if numel(idx) > 1
                inventory.valid(idx) = false;
                inventory.reason(idx) = "Duplicate optimizer/seed for this configuration";
            elseif isempty(idx)
                inventory(end+1,:) = {"",key,opt,seed,false,"Missing expected run"};
            end
        end
    end
end

stamp = string(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
outDir = fullfile(root,"FE_DATA_"+stamp);
assert(~isfolder(outDir),'Analysis output directory already exists.');
mkdir(outDir);
summary = table();
runMetrics = table();
groupStatus = table();

for key = keys'
    rows = find(inventory.comparison_key == key);
    complete = all(inventory.valid(rows));
    reason = "";
    % Different settings for repeated runs of the same algorithm are invalid.
    if complete
        for opt = optimizers
            idx = rows(inventory.optimizer(rows) == opt);
            optionTexts = strings(numel(idx),1);
            for j = 1:numel(idx)
                optionTexts(j) = string(runs{idx(j)}.solverSettingsText);
            end
            if numel(unique(optionTexts)) ~= 1
                complete = false;
                reason = "Solver settings differ between repetitions";
            end
        end
    else
        reason = "Missing, duplicate, failed, or invalid runs; see inventory";
    end
    groupStatus = [groupStatus; table(key,complete,reason, ...
        'VariableNames',{'comparison_key','complete','reason'})]; %#ok<AGROW>
    if ~complete, continue; end

    % Preserve early BO data. Values before another algorithm's first
    % checkpoint remain NaN; no best value is invented.
    feGrid = (1:budget)';
    curves = struct([]);
    representative = runs{rows(1)};
    comparison = representative.comparison;

    for a = 1:numel(optimizers)
        opt = optimizers(a);
        idx = rows(inventory.optimizer(rows) == opt);
        [~,order] = ismember(seeds,inventory.seed(idx));
        idx = idx(order);
        traces = nan(numel(feGrid),numel(seeds));
        costs = nan(numel(seeds),1);
        runtimes = costs;
        solverCalls = costs;

        for j = 1:numel(idx)
            r = runs{idx(j)};
            H = r.history;
            valid = feGrid >= H.fe(1) & feGrid <= H.fe(end);
            if height(H) == 1
                assert(H.fe(1) == budget);
                traces(valid,j) = H.bestJ(1);
            else
                traces(valid,j) = interp1( ...
                    H.fe,H.bestJ,feGrid(valid),'previous');
            end
            assert(all(isfinite(traces(valid,j))) && ...
                all(isnan(traces(~valid,j))), ...
                'Alignment would require extrapolation or backfilling.');
            costs(j) = r.bestJ;
            runtimes(j) = r.runtime_s;
            solverCalls(j) = r.solverFunctionEvaluations;
            identity = table(key,opt,r.optimizerSeed,r.bestJ,r.nEvaluations, ...
                r.solverFunctionEvaluations,r.solverCallDifference, ...
                r.runtime_s,r.validationRuntime_s,inventory.run_file(idx(j)), ...
                'VariableNames',{'comparison_key','optimizer','seed','bestJ', ...
                'search_fe','solver_calls','solver_call_difference', ...
                'optimization_runtime_s','validation_runtime_s','run_file'});
            runMetrics = [runMetrics; [identity struct2table(r.metrics)]]; %#ok<AGROW>
        end
        curves(a).optimizer = opt;
        curves(a).seeds = seeds;
        curves(a).fe = feGrid;
        curves(a).bestJ = traces;
        curves(a).firstRecordedFE = min( ...
            cellfun(@(r) r.history.fe(1),runs(idx)));
        curves(a).mean = mean(traces,2,'omitnan');
        curves(a).std = std(traces,0,2,'omitnan');
        if numel(seeds) == 1, curves(a).std(:) = NaN; end
        curves(a).median = median(traces,2);
        curves(a).q25 = prctile(traces,25,2);
        curves(a).q75 = prctile(traces,75,2);

        row = table(key,opt,string(representative.settings.mission.type), ...
            string(representative.settings.measurements.type), ...
            representative.settings.mission.optimization.numObservers, ...
            numel(seeds),budget,mean(costs),sample_std(costs),median(costs), ...
            mean(runtimes),sample_std(runtimes),min(solverCalls),max(solverCalls), ...
            'VariableNames',{'comparison_key','optimizer','mission','measurement', ...
            'num_observers','n_runs','fe_budget','bestJ_mean','bestJ_std', ...
            'bestJ_median','runtime_mean_s','runtime_std_s', ...
            'solver_calls_min','solver_calls_max'});
        summary = [summary; row]; %#ok<AGROW>
    end
    save(fullfile(outDir,"convergence_"+key+".mat"),'comparison','curves','-v7');
    if makePlots
        try
        figure('Color','w'); hold on; grid on;
        for a = 1:numel(curves)
            stairs(curves(a).fe,curves(a).mean,'LineWidth',1.5, ...
                'DisplayName',curves(a).optimizer);
        end
        xlabel('Function evaluations'); ylabel('Mean best-so-far objective');
        legend('Location','best'); % Display only. No figure export.
        catch ME
            warning('Study:PreviewFailed','Preview failed: %s',ME.message);
        end
    end
end
writetable(inventory,fullfile(outDir,'run_inventory.csv'));
if ~isempty(groupStatus)
    writetable(groupStatus,fullfile(outDir,'comparison_status.csv'));
end
if ~isempty(summary)
    writetable(summary,fullfile(outDir,'FE_summary.csv'));
    writetable(runMetrics,fullfile(outDir,'final_run_metrics.csv'));
else
    warning('Study:NoCompleteGroups', ...
        'No complete comparison groups. Inspect run_inventory.csv.');
end
fprintf('Saved analysis data only:\n%s\n',outDir);
end

function value = sample_std(x)
if numel(x) < 2, value = NaN; else, value = std(x); end
end
