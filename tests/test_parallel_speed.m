function results = test_parallel_speed(nRepeats, missionType)
%TEST_PARALLEL_SPEED Serial/parallel GA timing and convergence at 6000 FE.
% test_parallel_speed(3)                 % LG, one period, three AO observers
% test_parallel_speed(3,"LOW_THRUST_TRANSFER") % optional other target
% Repetitions use the same seeds: timing repetitions, not independent trials.
% Saved histories include actual callback times, excluding pool startup.
% Use plot_parallel_speed to export the saved LG comparison.
if nargin < 1 || isempty(nRepeats), nRepeats = 2; end
if nargin < 2 || isempty(missionType), missionType = "LUNAR_GATEWAY"; end
validateattributes(nRepeats,{'numeric'},{'scalar','integer','positive','finite'});
missionType = upper(string(missionType));
assert(isscalar(missionType) && ismember(missionType, ...
    ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"]),'Unknown mission type.');
projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir); paths = setup_project();
budget = 6000;
stamp = char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
testRoot = fullfile(paths.root,'MANUSCRIPT_OUTPUT', ...
    ['parallel_speed_' lower(char(missionType)) '_' stamp]);
mkdir(testRoot);
envNames = {'MAX_EVALS','USE_PARALLEL_OPT','MISSION_TYPE','MEAS_MODEL', ...
    'NUM_OBSERVERS','NPERIODS','USE_SCREENING','USE_J1','USE_J2','USE_J3', ...
    'SEED','MEAS_NOISE_SEED','MAKE_PLOTS','OPTIMIZER_MODE','STUDY_ID','RUN_DIR', ...
    'IMPULSE_DV_MPS','IMPULSE_DIRECTION','IMPULSE_DURATION_TU'};
oldValues = cellfun(@getenv,envNames,'UniformOutput',false);
cleanup = onCleanup(@() restore_environment(envNames,oldValues)); %#ok<NASGU>
values = {'6000','0',char(missionType),'ANGLES_ONLY','3','1','1','1','1','1', ...
    '0','1001','0','GA','parallel_speed_test','', '10','PROGRADE','1.5'};
for k = 1:numel(envNames), setenv(envNames{k},values{k}); end
modeNames = ["Serial","Parallel"];
nRows = 2*nRepeats;
Mode = strings(nRows,1); Repeat = zeros(nRows,1); Workers = zeros(nRows,1);
OptimizationRuntime_s = nan(nRows,1); WallRuntime_s = nan(nRows,1);
BestJ = nan(nRows,1); SearchFE = nan(nRows,1); SolverCalls = nan(nRows,1);
RunDirectory = strings(nRows,1); histories = cell(nRows,1);
row = 0;
% Alternate which mode runs first to reduce systematic execution-order bias.
for r = 1:nRepeats
    order = [1 2]; if mod(r,2)==0, order = [2 1]; end
    for m = order
        pool = gcp('nocreate');
        if m==1 && ~isempty(pool), delete(pool); end
        row = row+1;
        setenv('USE_PARALLEL_OPT',num2str(m==2));
        runDir = fullfile(testRoot,lower(char(modeNames(m))),sprintf('repeat_%02d',r));
        mkdir(runDir); setenv('RUN_DIR',runDir);
        fprintf('\n%s run %d/%d | %s | %d FE\n',modeNames(m),r,nRepeats,missionType,budget);
        timer = tic; R = run_case(projectDir); wallElapsed = toc(timer);
        H = R.history;
        assert(all(ismember({'fe','bestJ','elapsed_s'},H.Properties.VariableNames)), ...
            'Missing callback timing; update run_opt.m before running this test.');
        assert(R.searchFunctionEvaluations==budget && H.fe(end)==budget, ...
            'Benchmark did not complete the prescribed FE budget.');
        Mode(row)=modeNames(m); Repeat(row)=r; Workers(row)=R.settings.workerCount;
        OptimizationRuntime_s(row)=R.runtime_s; WallRuntime_s(row)=wallElapsed;
        BestJ(row)=R.bestJ; SearchFE(row)=R.searchFunctionEvaluations;
        SolverCalls(row)=R.solverFunctionEvaluations; RunDirectory(row)=string(runDir);
        histories{row}=H;
        writetable(H,fullfile(runDir,'convergence_history.csv'));
        % Save completed repetitions incrementally so interrupted tests remain auditable.
        results = table(Mode(1:row),Repeat(1:row),Workers(1:row), ...
            OptimizationRuntime_s(1:row),WallRuntime_s(1:row),BestJ(1:row), ...
            SearchFE(1:row),SolverCalls(1:row),RunDirectory(1:row), ...
            'VariableNames',{'Mode','Repeat','Workers','OptimizationRuntime_s', ...
            'WallRuntime_s','BestJ','SearchFE','SolverCalls','RunDirectory'});
        completed = histories(1:row);
        benchmark = struct('mission',missionType,'budget',budget,'nRepeats',nRepeats, ...
            'optimizerSeed',0,'measurementSeed',1001,'results',results, ...
            'histories',{completed},'complete',row==nRows);
        save(fullfile(testRoot,'parallel_speed_convergence.mat'),'benchmark');
        writetable(results,fullfile(testRoot,'parallel_speed_results.csv'));
    end
end
pool = gcp('nocreate'); if ~isempty(pool), delete(pool); end
disp(results);
serial = results.Mode=="Serial"; parallel = results.Mode=="Parallel";
summary = sprintf(['GA %s | %d FE | %d fixed-seed timing repetitions per mode\n' ...
    'Mean optimization runtime: serial %.3f s; parallel %.3f s; speedup %.3fx\n' ...
    'Mean wall runtime: serial %.3f s; parallel %.3f s; speedup %.3fx\n' ...
    'Wall runtime includes setup/validation and may include pool startup.\n'], ...
    missionType,budget,nRepeats,mean(results.OptimizationRuntime_s(serial)), ...
    mean(results.OptimizationRuntime_s(parallel)), ...
    mean(results.OptimizationRuntime_s(serial))/mean(results.OptimizationRuntime_s(parallel)), ...
    mean(results.WallRuntime_s(serial)),mean(results.WallRuntime_s(parallel)), ...
    mean(results.WallRuntime_s(serial))/mean(results.WallRuntime_s(parallel)));
fprintf('%s\nSaved benchmark: %s\n',summary,testRoot);
fid=fopen(fullfile(testRoot,'parallel_speed_summary.txt'),'w');
assert(fid>=0,'Cannot write benchmark summary.');
fileCleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s',summary);
end

function R = run_case(projectDir)
% run_opt begins with clear; isolate it from the benchmark workspace.
run(fullfile(projectDir,'run_opt.m'));
R = runState;
end

function restore_environment(names,values)
for k=1:numel(names), setenv(names{k},values{k}); end
end
