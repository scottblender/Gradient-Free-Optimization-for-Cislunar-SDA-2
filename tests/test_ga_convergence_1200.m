function report = test_ga_convergence_1200(seed,missionType)
%TEST_GA_CONVERGENCE_1200 Run a representative 1200-FE GA convergence test.
%
% This integration diagnostic uses the real optimization objective with the
% same representative comparison-study settings used for Reviewer 2:
%   - GA, population size 60
%   - 1200 admitted search function evaluations
%   - parallel optimization enabled
%   - angles-only measurements
%   - 3 observers
%   - physical screening enabled
%   - fixed measurement-noise seed 1001
%
% The saved GA history is generation-level (60,120,...,1200 FE). For visual
% inspection, the plot is expanded to the complete 1:1200 FE grid using the
% last legitimate incumbent after each recorded checkpoint. No value is
% invented before the initial 60-member population has completed.
%
% Usage:
%   report = test_ga_convergence_1200;
%   report = test_ga_convergence_1200(3);
%   report = test_ga_convergence_1200(0,"GATEWAY_IMPULSE");

if nargin < 1 || isempty(seed), seed = 0; end
if nargin < 2 || isempty(missionType), missionType = "LUNAR_GATEWAY"; end

validateattributes(seed,{'numeric'}, ...
    {'scalar','real','finite','integer','>=',0,'<=',2^32-1});
missionType = upper(string(missionType));
assert(isscalar(missionType) && ismember(missionType, ...
    ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"]), ...
    'Unknown mission type.');

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
paths = setup_project();

budget = 1200;
settings = {
    'MAX_EVALS',         num2str(budget)
    'MAX_ITERS',         '100000'
    'USE_PARALLEL_OPT',  '1'
    'OPTIMIZER_MODE',    'GA'
    'SEED',              num2str(seed)
    'MISSION_TYPE',      char(missionType)
    'MEAS_MODEL',        'ANGLES_ONLY'
    'NUM_OBSERVERS',     '3'
    'NPERIODS',          '1'
    'USE_SCREENING',     '1'
    'USE_J1',            '1'
    'USE_J2',            '1'
    'USE_J3',            '1'
    'MEAS_NOISE_SEED',   '1001'
    'EKF_DT',            '0.01'
    'STUDY_ID',          'ga_convergence_1200_test'
    'MAKE_PLOTS',        '0'
    'IMPULSE_DV_MPS',    '10'
    'IMPULSE_DIRECTION', 'PROGRADE'
    'IMPULSE_DURATION_TU','1.5'
};

envNames = [settings(:,1); {'RUN_DIR';'SAFE_FALLBACK_FILE'}];
oldValues = cellfun(@getenv,envNames,'UniformOutput',false);
oldFolder = pwd;
cleanup = onCleanup(@() restore_environment( ...
    envNames,oldValues,oldFolder)); %#ok<NASGU>

for k = 1:size(settings,1)
    setenv(settings{k,1},settings{k,2});
end

stamp = char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
testRoot = fullfile(paths.results,['GA_CONVERGENCE_1200_' stamp]);
runDir = fullfile(testRoot,'run');
mkdir(runDir);
setenv('RUN_DIR',runDir);

fprintf('\n--- 1200-FE GA convergence diagnostic ---\n');
fprintf('Mission:            %s\n',missionType);
fprintf('Optimizer seed:     %d\n',seed);
fprintf('Measurement seed:   1001\n');
fprintf('Search FE budget:   %d\n',budget);
fprintf('Parallel:           enabled\n\n');

runState = run_one_case(projectDir);
H = runState.history;

assert(istable(H) && all(ismember({'fe','bestJ'}, ...
    H.Properties.VariableNames)), ...
    'Saved GA history is missing fe/bestJ.');
assert(H.fe(1) == 60 && H.fe(end) == budget, ...
    'Expected GA history from FE 60 through FE %d.',budget);
assert(runState.nEvaluations == budget, ...
    'GA did not finish the requested 1200-FE search budget.');

feGrid = (1:budget)';
bestGrid = nan(budget,1);
valid = feGrid >= H.fe(1);
bestGrid(valid) = interp1(double(H.fe),double(H.bestJ), ...
    feGrid(valid),'previous');

assert(all(isnan(bestGrid(1:H.fe(1)-1))), ...
    'Values were backfilled before the first legitimate GA checkpoint.');
assert(all(isfinite(bestGrid(H.fe(1):end))), ...
    'Expanded convergence history contains an unexpected gap.');
assert(all(diff(bestGrid(H.fe(1):end)) <= ...
    1e-12*max(1,abs(bestGrid(H.fe(1):end-1)))), ...
    'Expanded convergence history is not best-so-far.');

fig = figure('Color','w','Units','inches','Position',[1 1 7.2 4.4], ...
    'PaperUnits','inches','PaperSize',[7.2 4.4], ...
    'PaperPosition',[0 0 7.2 4.4],'PaperPositionMode','manual', ...
    'Renderer','painters');
ax = axes(fig);
hold(ax,'on');
box(ax,'on');
grid(ax,'on');

stairs(ax,feGrid(valid),bestGrid(valid),'LineWidth',2.0);
plot(ax,double(H.fe),double(H.bestJ),'o', ...
    'MarkerSize',4.5,'HandleVisibility','off');

xlim(ax,[H.fe(1) budget]);
xlabel(ax,'Function evaluations','FontWeight','bold');
ylabel(ax,'Best-so-far objective','FontWeight','bold');
set(ax,'FontName','Times New Roman','FontSize',12, ...
    'FontWeight','bold','LineWidth',1.2,'Layer','top');
ax.XLabel.FontSize = 14;
ax.YLabel.FontSize = 14;

figureStem = fullfile(testRoot,'ga_convergence_1200');
print(fig,[figureStem '.eps'],'-depsc','-painters');
exportgraphics(fig,[figureStem '.png'],'Resolution',300);

expandedHistory = table(feGrid,bestGrid, ...
    'VariableNames',{'fe','bestJ'});
writetable(expandedHistory,fullfile(testRoot, ...
    'ga_convergence_1200_expanded.csv'));
writetable(H,fullfile(testRoot,'ga_convergence_1200_recorded.csv'));

report = struct();
report.testRoot = string(testRoot);
report.runDirectory = string(runDir);
report.figureEPS = string(figureStem+'.eps');
report.figurePNG = string(figureStem+'.png');
report.recordedHistory = H;
report.expandedHistory = expandedHistory;
report.bestJ = runState.bestJ;
report.searchFE = runState.nEvaluations;
report.solverCalls = runState.solverFunctionEvaluations;
report.runtime_s = runState.runtime_s;

fprintf('\nGA convergence diagnostic complete.\n');
fprintf('Recorded checkpoints: %d\n',height(H));
fprintf('Final best objective: %.12g\n',runState.bestJ);
fprintf('Optimization runtime: %.3f s\n',runState.runtime_s);
fprintf('Saved diagnostic output to:\n%s\n',testRoot);
end

function runState = run_one_case(projectDir)
% Separate workspace because run_opt starts with clear.
run(fullfile(projectDir,'run_opt.m'));
end

function restore_environment(names,values,oldFolder)
for k = 1:numel(names)
    setenv(names{k},values{k});
end
if isfolder(oldFolder), cd(oldFolder); end
end
