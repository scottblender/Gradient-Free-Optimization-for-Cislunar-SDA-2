function summary = test_small_fe_case(algorithms, seeds, missionType)
% Small integration test of the current run_opt workflow.
%
% Examples:
%   test_small_fe_case
%   test_small_fe_case("PSO", 0)
%   test_small_fe_case(["GA","PSO","BAYESIAN","ABC","ACO"], 0)
%   test_small_fe_case("GA", 0, "LOW_THRUST_TRANSFER")
%   test_small_fe_case("GA", 0, "GATEWAY_IMPULSE")
%
% This runs the selected real objective and final diagnostic replay.
% One additional objective evaluation checks the saved winner.
% That validation evaluation is outside the optimization budget.

if nargin < 1 || isempty(algorithms)
    algorithms = "GA";
end

if nargin < 2 || isempty(seeds)
    seeds = 0;
end

if nargin < 3 || isempty(missionType)
    missionType = "LUNAR_GATEWAY";
end

algorithms = upper(string(algorithms));
missionType = upper(string(missionType));

assert(isscalar(missionType) && ...
    ismember(missionType, ...
    ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"]), ...
    'Unknown pilot mission type.');

assert(all(ismember(algorithms, ...
    ["GA","PSO","BAYESIAN","ABC","ACO"]), 'all'), ...
    'Unknown optimizer.');

validateattributes(seeds, {'numeric'}, ...
    {'vector','nonempty','real','finite','integer', ...
     '>=',0,'<=',2^32-1});

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
paths = setup_project();

budget = 120;

% Explicit settings prevent inherited environment values changing the case.
settings = {
    'MAX_EVALS',        '120'
    'MAX_ITERS',        '100000'
    'USE_PARALLEL_OPT', '0'
    'MISSION_TYPE',     char(missionType)
    'MEAS_MODEL',       'ANGLES_ONLY'
    'NUM_OBSERVERS',    '3'
    'NPERIODS',         '1'
    'USE_SCREENING',    '1'
    'USE_J1',           '1'
    'USE_J2',           '1'
    'USE_J3',           '1'
    'MEAS_NOISE_SEED',  '1001'
    'EKF_DT',           '0.01'
    'STUDY_ID',         char("reviewer2_pilot_" + lower(missionType))
    'MAKE_PLOTS',       '0'
    'IMPULSE_DV_MPS',    '10'
    'IMPULSE_DIRECTION', 'PROGRADE'
    'IMPULSE_DURATION_TU','1.5'
};

envNames = [
    settings(:,1)
    {'OPTIMIZER_MODE'; 'SEED'; 'RUN_DIR'; 'SAFE_FALLBACK_FILE'}
];

oldValues = cellfun(@getenv, envNames, 'UniformOutput', false);
oldFolder = pwd;

cleanup = onCleanup(@() restore_environment( ...
    envNames, oldValues, oldFolder)); %#ok<NASGU>

for k = 1:size(settings,1)
    setenv(settings{k,1}, settings{k,2});
end

% Keep pilot results outside results/runs so the full-study processing
% script cannot accidentally combine them with publication runs.
stamp = char(datetime('now', 'Format','yyyyMMdd_HHmmss_SSS'));
pilotRoot = fullfile(paths.results, ['FE_PILOT_' stamp]);

assert(~isfolder(pilotRoot), 'Pilot output directory already exists.');
mkdir(pilotRoot);

template = struct( ...
    'optimizer',"", ...
    'seed',0, ...
    'searchFE',NaN, ...
    'totalSolverCalls',NaN, ...
    'postSearchCalls',NaN, ...
    'searchBudgetOK',false, ...
    'solverCallPatternOK',false, ...
    'historyOK',false, ...
    'bestOK',false, ...
    'recheckOK',false, ...
    'termination',"", ...
    'message',"", ...
    'runDir',"");

records = repmat(template, numel(algorithms)*numel(seeds), 1);
rowNumber = 0;

for a = 1:numel(algorithms)
    for s = 1:numel(seeds)

        rowNumber = rowNumber + 1;
        row = template;

        alg = algorithms(a);
        seed = seeds(s);

        runDir = fullfile(pilotRoot, char(alg), ...
            sprintf('seed_%03d', seed));
        mkdir(runDir);

        row.optimizer = alg;
        row.seed = seed;
        row.runDir = string(runDir);

        setenv('OPTIMIZER_MODE', char(alg));
        setenv('SEED', num2str(seed));
        setenv('RUN_DIR', runDir);

        fprintf('\nPilot: %s, %s, seed %d, budget %d\n', ...
            missionType, alg, seed, budget);

        recheckJ = NaN;

        try
            recheckJ = run_one_case(projectDir);
        catch ME
            row.message = string(ME.message);
            warning('FEPilot:RunFailed', ...
                '%s seed %d: %s', alg, seed, ME.message);
        end

        % A run can save optimization results successfully and then fail
        % during final plotting. Inspect the saved results either way.
        try
            resultFile = fullfile(runDir, 'data', ...
                'optimization_run.mat');

            assert(isfile(resultFile), ...
                'No optimization_run.mat was saved.');

            saved = load(resultFile, 'runState');
            R = saved.runState;
            H = R.history;

            assert(istable(H) && ~isempty(H), ...
                'The saved history is empty or is not a table.');

            assert(all(ismember({'fe','bestJ'}, ...
                H.Properties.VariableNames)), ...
                'The saved history is missing fe or bestJ.');

            fe = double(H.fe(:));
            best = double(H.bestJ(:));
            tol = 1e-10 * max(1, abs(R.bestJ));

            row.searchFE = R.nEvaluations;
            row.totalSolverCalls = R.solverFunctionEvaluations;
            row.postSearchCalls = R.postSearchFunctionEvaluations;

            row.searchBudgetOK = R.nEvaluations == budget;
            expectedPostSearch = double(alg == "GA");
            row.solverCallPatternOK = ...
                R.postSearchFunctionEvaluations == expectedPostSearch && ...
                R.solverFunctionEvaluations == budget+expectedPostSearch;

            row.historyOK = ...
                all(isfinite(fe)) && ...
                all(fe >= 1 & fe == floor(fe)) && ...
                all(diff(fe) > 0) && ...
                all(isfinite(best)) && ...
                all(diff(best) <= tol) && ...
                fe(end) == R.nEvaluations;

            row.bestOK = isfinite(R.bestJ) && ...
                abs(R.bestJ - min(best)) <= tol;

            row.recheckOK = isfinite(recheckJ) && ...
                abs(recheckJ - R.bestJ) <= tol;

            row.termination = string(R.termination);

        catch ME
            row.message = row.message + " " + string(ME.message);
            warning('FEPilot:ResultCheck', ...
                '%s seed %d: %s', alg, seed, ME.message);
        end

        records(rowNumber) = row;

        % Save after every run.
        summary = struct2table(records(1:rowNumber), 'AsArray', true);
        writetable(summary, fullfile(pilotRoot, 'pilot_summary.csv'));
    end
end

disp(summary(:, {
    'optimizer','seed','searchFE','totalSolverCalls', ...
    'postSearchCalls','searchBudgetOK','solverCallPatternOK', ...
    'historyOK','bestOK', ...
    'recheckOK','termination'
}));

fprintf('\nPilot results saved to:\n%s\n', pilotRoot);
end


function recheckJ = run_one_case(projectDir)
% Separate workspace: run_opt begins with "clear".
% It must not clear the pilot controller's variables or cleanup object.

run(fullfile(projectDir, 'run_opt.m'));

% Fixed-noise validation of the saved design.
% RawObjFcn bypasses optimizer counting.
recheckJ = RawObjFcn(runState.bestX);
end


function restore_environment(names, values, oldFolder)

for k = 1:numel(names)
    setenv(names{k}, values{k});
end

if isfolder(oldFolder)
    cd(oldFolder);
end
end
