function results = test_parallel_speed(nRepeats, missionType)
%TEST_PARALLEL_SPEED Compare serial vs parallel runtime using the real study.
%
% Examples:
%   test_parallel_speed
%   test_parallel_speed(3)
%   test_parallel_speed(3,"LOW_THRUST_TRANSFER")
%
% Runs the same:
%   - GA
%   - 120 FE
%   - 3 observers
%   - angles only
%   - optimizer seed 0
%   - measurement seed 1001
%
% Reports:
%   optimization runtime from runState.runtime_s
%   total wall-clock runtime
%   serial/parallel speedup

if nargin < 1 || isempty(nRepeats)
    nRepeats = 2;
end

if nargin < 2 || isempty(missionType)
    missionType = "LUNAR_GATEWAY";
end

missionType = upper(string(missionType));

assert(ismember(missionType, ...
    ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"]), ...
    'Unknown mission type.');

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
paths = setup_project();

budget = 120;
optimizerSeed = 0;
measurementSeed = 1001;

stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
testRoot = fullfile(paths.results, ...
    ['PARALLEL_SPEED_TEST_' stamp]);
mkdir(testRoot);

% Preserve current environment.
envNames = { ...
    'MAX_EVALS'
    'USE_PARALLEL_OPT'
    'MISSION_TYPE'
    'MEAS_MODEL'
    'NUM_OBSERVERS'
    'NPERIODS'
    'USE_SCREENING'
    'USE_J1'
    'USE_J2'
    'USE_J3'
    'SEED'
    'MEAS_NOISE_SEED'
    'MAKE_PLOTS'
    'OPTIMIZER_MODE'
    'STUDY_ID'
    'RUN_DIR'
    'IMPULSE_DV_MPS'
    'IMPULSE_DIRECTION'
    'IMPULSE_DURATION_TU'
    };

oldValues = cellfun(@getenv, envNames, ...
    'UniformOutput', false);

cleanup = onCleanup(@() restore_environment( ...
    envNames, oldValues)); %#ok<NASGU>

% Fixed study settings.
setenv('MAX_EVALS', num2str(budget));
setenv('MISSION_TYPE', char(missionType));
setenv('MEAS_MODEL', 'ANGLES_ONLY');
setenv('NUM_OBSERVERS', '3');
setenv('NPERIODS', '1');
setenv('USE_SCREENING', '1');
setenv('USE_J1', '1');
setenv('USE_J2', '1');
setenv('USE_J3', '1');
setenv('SEED', num2str(optimizerSeed));
setenv('MEAS_NOISE_SEED', num2str(measurementSeed));
setenv('MAKE_PLOTS', '0');
setenv('OPTIMIZER_MODE', 'GA');
setenv('STUDY_ID', 'parallel_speed_test');

setenv('IMPULSE_DV_MPS', '10');
setenv('IMPULSE_DIRECTION', 'PROGRADE');
setenv('IMPULSE_DURATION_TU', '1.5');

modeNames = ["Serial","Parallel"];
useParallel = [false,true];

nRows = 2*nRepeats;

Mode = strings(nRows,1);
Repeat = zeros(nRows,1);
Workers = zeros(nRows,1);
OptimizationRuntime_s = nan(nRows,1);
WallRuntime_s = nan(nRows,1);
BestJ = nan(nRows,1);

row = 0;

for m = 1:2

    % Make the comparison clean.
    p = gcp('nocreate');
    if ~isempty(p)
        delete(p);
    end

    for r = 1:nRepeats

        row = row + 1;

        setenv('USE_PARALLEL_OPT', ...
            num2str(double(useParallel(m))));

        runDir = fullfile(testRoot, ...
            lower(char(modeNames(m))), ...
            sprintf('repeat_%02d',r));

        mkdir(runDir);
        setenv('RUN_DIR',runDir);

        fprintf('\n----------------------------------------\n');
        fprintf('%s run %d/%d\n', ...
            modeNames(m),r,nRepeats);
        fprintf('Mission: %s | FE: %d\n', ...
            missionType,budget);
        fprintf('----------------------------------------\n');

        wallTimer = tic;

        R = run_case(projectDir);

        wallElapsed = toc(wallTimer);

        Mode(row) = modeNames(m);
        Repeat(row) = r;
        OptimizationRuntime_s(row) = R.runtime_s;
        WallRuntime_s(row) = wallElapsed;
        BestJ(row) = R.bestJ;

        if isfield(R.settings,'workerCount')
            Workers(row) = R.settings.workerCount;
        end

        fprintf('Optimization runtime: %.3f s\n', ...
            R.runtime_s);
        fprintf('Wall-clock runtime:    %.3f s\n', ...
            wallElapsed);
        fprintf('Best J:                %.12g\n', ...
            R.bestJ);
    end
end

% Clean up pool after benchmark.
p = gcp('nocreate');
if ~isempty(p)
    delete(p);
end

results = table( ...
    Mode,Repeat,Workers, ...
    OptimizationRuntime_s,WallRuntime_s,BestJ);

disp(results);

serialOpt = mean( ...
    results.OptimizationRuntime_s(results.Mode=="Serial"));
parallelOpt = mean( ...
    results.OptimizationRuntime_s(results.Mode=="Parallel"));

serialWall = mean( ...
    results.WallRuntime_s(results.Mode=="Serial"));
parallelWall = mean( ...
    results.WallRuntime_s(results.Mode=="Parallel"));

optSpeedup = serialOpt/parallelOpt;
wallSpeedup = serialWall/parallelWall;

fprintf('\n========== SPEED SUMMARY ==========\n');
fprintf('Mission: %s\n',missionType);
fprintf('FE budget: %d\n',budget);
fprintf('Repeats per mode: %d\n\n',nRepeats);

fprintf('Mean optimization runtime:\n');
fprintf('  Serial:   %.3f s\n',serialOpt);
fprintf('  Parallel: %.3f s\n',parallelOpt);
fprintf('  Speedup:  %.2fx\n\n',optSpeedup);

fprintf('Mean full wall-clock runtime:\n');
fprintf('  Serial:   %.3f s\n',serialWall);
fprintf('  Parallel: %.3f s\n',parallelWall);
fprintf('  Speedup:  %.2fx\n',wallSpeedup);

fprintf('\nResults saved under:\n%s\n',testRoot);
fprintf('===================================\n');

writetable(results, ...
    fullfile(testRoot,'parallel_speed_results.csv'));

end


function R = run_case(projectDir)
%RUN_CASE Give run_opt its own workspace because run_opt begins with clear.

run(fullfile(projectDir,'run_opt.m'));

R = runState;

end


function restore_environment(names,values)

for k = 1:numel(names)
    setenv(names{k},values{k});
end

end