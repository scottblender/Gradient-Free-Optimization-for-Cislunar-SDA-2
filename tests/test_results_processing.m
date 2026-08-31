function test_results_processing()
% Synthetic tests: no orbit propagation and no optimizer calls.
projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
setup_project();
root = tempname;
mkdir(root);
cleanup = onCleanup(@() rmdir(root,'s'));
opts = ["GA","BAYESIAN"];

% Complete paired seeds, unequal checkpoint spacing.
caseDir = fullfile(root,'complete');
make_fixture(caseDir);
[summary,inventory] = process_fe_convergence(caseDir,"test",0:1,120,false,opts);
assert(height(summary) == 2 && all(summary.n_runs == 2));
assert(all(inventory.valid));
out = dir(fullfile(caseDir,'FE_DATA_*','convergence_*.mat'));
D = load(fullfile(out(end).folder,out(end).name),'curves');
assert(D.curves(1).fe(1) == 60, 'An unobserved early FE was invented.');
assert(all(D.curves(1).bestJ(1:60,1) == 8));
assert(all(D.curves(2).bestJ(1:60,1) == 6));
assert(D.curves(1).bestJ(end,1) == 3);

% A genuinely single-checkpoint history needs no interpolation.
caseDir = fullfile(root,'single_checkpoint');
make_fixture(caseDir);
for opt = opts
    for seed = 0:1
        file = fixture_file(caseDir,opt,seed);
        S = load(file,'runState');
        runState = S.runState;
        runState.history = runState.history(end,:);
        save(file,'runState');
    end
end
[summary,~] = process_fe_convergence(caseDir,"test",0:1,120,false,opts);
assert(height(summary) == 2);

% A worsening "best-so-far" history must not be silently repaired.
caseDir = fullfile(root,'bad_history');
make_fixture(caseDir);
file = fixture_file(caseDir,"GA",0);
S = load(file,'runState');
runState = S.runState;
runState.history.bestJ = [8;9];
runState.bestJ = 9;
runState.metrics.J_recheck = 9;
save(file,'runState');
[summary,inventory] = process_fe_convergence(caseDir,"test",0:1,120,false,opts);
assert(isempty(summary));
assert(any(contains(inventory.reason,"No automatic repair")));

% A completed duplicate is not an additional independent run.
caseDir = fullfile(root,'duplicate');
make_fixture(caseDir);
dupDir = fullfile(caseDir,'duplicate','data');
mkdir(dupDir);
copyfile(fixture_file(caseDir,"GA",0),fullfile(dupDir,'optimization_run.mat'));
tracking = [];
save(fullfile(dupDir,'tracking_data.mat'),'tracking');
[summary,inventory] = process_fe_convergence(caseDir,"test",0:1,120,false,opts);
assert(isempty(summary));
assert(nnz(contains(inventory.reason,"Duplicate")) == 2);

% A different measurement-noise seed creates a different comparison group.
caseDir = fullfile(root,'different_noise');
make_fixture(caseDir);
file = fixture_file(caseDir,"GA",0);
S = load(file,'runState');
runState = S.runState;
runState.settings.measurements.noiseSeed = 999;
runState.comparison.settings = runState.settings;
runState.comparisonKey = study_hash(runState.comparison);
save(file,'runState');
[summary,inventory] = process_fe_convergence(caseDir,"test",0:1,120,false,opts);
assert(isempty(summary));
assert(any(inventory.reason == "Missing expected run"));

% Verify final diagnostics retain disabled objective components.
caseDir = fullfile(root,'tracking');
mkdir(caseDir);
truth = zeros(3,6);
estimate = truth; estimate(:,1) = [0;1;2];
P = repmat(reshape(eye(6),1,6,6),3,1,1);
cost = struct('weights',[1 1 .1],'pos_rmse_acc',1,'vel_rmse_acc',1, ...
    'sigma_pos_acc',1,'sigma_vel_acc',1,'stability_acc',1);
flags = struct('J1',true,'J2',false,'J3',true);
runState = struct('settings',struct('LU',1,'TU',1,'cost',cost,'costFlags',flags));
runState.bestJ = compute_cost(truth,estimate,P,1,'SOO',flags,cost);
observers = table(1,'VariableNames',{'stability_index'});
runState = save_tracking_results(caseDir,runState,[0;1;2],truth,estimate,P, ...
    [1;0;1],1,observers);
S = load(fullfile(caseDir,'tracking_data.mat'),'tracking');
assert(isnan(S.tracking.availableObsCount(1)));
assert(runState.metrics.J2_weighted > 0);
assert(runState.metrics.coverage_epoch_fraction == .5);

% Every output in this test should be data, not a plot or spreadsheet.
allFiles = dir(fullfile(root,'**','*'));
names = string({allFiles.name});
assert(~any(endsWith(names,[".png",".pdf",".eps",".fig",".xlsx"])));
fprintf('All result-processing tests passed.\n');
end

function make_fixture(root)
for opt = ["GA","BAYESIAN"]
    for seed = 0:1
        runState = struct();
        runState.schemaVersion = 2;
        runState.studyID = "test";
        runState.optimizer = opt;
        runState.optimizerSeed = seed;
        runState.maxEvaluations = 120;
        runState.nEvaluations = 120;
        runState.solverFunctionEvaluations = 120;
        runState.solverCallDifference = 0;
        runState.settings = struct('mission',struct('type',"LUNAR_GATEWAY", ...
            'optimization',struct('numObservers',3)), ...
            'measurements',struct('type',"ANGLES_ONLY",'noiseSeed',1001));
        runState.comparison = struct('settings',runState.settings,'budget',120);
        runState.comparisonKey = study_hash(runState.comparison);
        runState.status = "completed";
        runState.termination = "budget_reached";
        runState.validationStatus = "passed";
        runState.objectiveErrorCount = 0;
        runState.solverSettingsText = "fixed settings";
        runState.runtime_s = 1;
        runState.validationRuntime_s = .1;
        if opt == "GA"
            runState.history = table([60;120],[8;3]+seed, ...
                'VariableNames',{'fe','bestJ'});
        else
            runState.history = table([1;30;120],[9;6;2]+seed, ...
                'VariableNames',{'fe','bestJ'});
        end
        runState.bestJ = runState.history.bestJ(end);
        runState.metrics = struct('J_recheck',runState.bestJ);
        file = fixture_file(root,opt,seed);
        mkdir(fileparts(file));
        save(file,'runState');
        tracking = [];
        save(fullfile(fileparts(file),'tracking_data.mat'),'tracking');
    end
end
end

function file = fixture_file(root,opt,seed)
file = fullfile(root,sprintf('%s_%d',opt,seed),'data','optimization_run.mat');
end
