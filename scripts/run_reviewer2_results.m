function reports = run_reviewer2_results(studies,saveFigures)
%RUN_REVIEWER2_RESULTS Process completed Reviewer 2 studies and final figures.
%
% This runner never launches optimization. Each selected study is validated
% and reduced to aggregate statistics first. Statistical comparisons printed
% here and used in manuscript figures are group mean +/- sample standard
% deviation across the 20 independent runs. A representative seed is used
% only to visualize one realizable discrete constellation near the group mean.
%
% Final processed outputs are moved out of the raw study trees and saved as:
%   COMPILED_REVIEWER_2_RESULTS/runtime_1200_<timestamp>/
%   COMPILED_REVIEWER_2_RESULTS/comparison_<timestamp>/
%   COMPILED_REVIEWER_2_RESULTS/baseline_<timestamp>/
%   COMPILED_REVIEWER_2_RESULTS/objective_screening_<timestamp>/
% Numerical CSV/MAT products and figure manifests remain at each timestamped
% study root. All manuscript EPS/PNG exports are collected into the single
% figures/ subdirectory beneath that root. Raw optimization results stay under
% results/ and are never modified by this runner.

if nargin < 1 || isempty(studies), studies = "all"; end
if nargin < 2 || isempty(saveFigures), saveFigures = true; end
saveFigures = logical(saveFigures);
studies = lower(string(studies(:)'));

canonicalOrder = ["runtime","comparison","baseline","objective_screening"];
if any(studies == "all")
    assert(numel(studies) == 1,'Use "all" by itself.');
    studies = canonicalOrder;
else
    studies(studies == "objective") = "objective_screening";
    studies(studies == "screening") = "objective_screening";
    unknown = setdiff(studies,canonicalOrder);
    assert(isempty(unknown),'Unknown study selector(s): %s',strjoin(cellstr(unknown),', '));
    studies = unique(studies,'stable');
end

paths = setup_project();
compiledRoot = fullfile(paths.root,'COMPILED_REVIEWER_2_RESULTS');
if ~isfolder(compiledRoot), mkdir(compiledRoot); end
reports = struct();
fprintf('\n=== Reviewer 2 results processing ===\n');
fprintf('Selected studies: %s\n',strjoin(cellstr(studies),', '));
fprintf('Save curated paper figures: %s\n',string(saveFigures));
fprintf('Compiled output root: %s\n',compiledRoot);
fprintf('Reported performance statistics: 20-run mean +/- sample standard deviation\n\n');

% Historical per-pipeline preview routines are hidden. Only the centralized
% curated renderer is intended for manuscript inspection.
originalFigureVisible = get(groot,'defaultFigureVisible');
visibilityCleanup = onCleanup(@() set(groot,'defaultFigureVisible',originalFigureVisible));
set(groot,'defaultFigureVisible','off');

for study = studies
    started = tic;
    fprintf('\n>>> Processing %s\n',upper(strrep(study,'_',' ')));
    switch study
        case "runtime"
            [~,tmp] = evalc('run_reviewer2_runtime_pipeline(false)');
            tmp = relocate_analysis(tmp,compiledRoot,"runtime_1200");
            reports.runtime = tmp;
            fprintf('\n1200-FE aggregate objective/runtime table:\n');
            disp(tmp.formattedTable);
            fprintf('\nBayesian equal-FE runtime comparison:\n');
            disp(tmp.boSlowdown);
            fprintf('\nEqual-FE conclusion metrics:\n');
            disp(tmp.conclusion);

        case "comparison"
            [~,tmp] = evalc('run_reviewer2_comparison_pipeline(false)');
            tmp = relocate_analysis(tmp,compiledRoot,"comparison");
            tmp.results = attach_comparison_keys(tmp.results,tmp.summary);
            reports.comparison = tmp;
            fprintf('\n6000-FE aggregate objective/runtime table:\n');
            disp(tmp.objectiveTable);
            fprintf('\n6000-FE aggregate tracking/design table:\n');
            disp(tmp.trackingTable);
            fprintf('\nBest optimizer by target case, based on mean final objective:\n');
            disp(tmp.bestByMission);
            fprintf('\nOverall optimizer ranking, based on mission-wise mean objective:\n');
            disp(tmp.overallRanking);

        case "baseline"
            [~,tmp] = evalc('run_reviewer2_baseline_pipeline(false)');
            tmp = relocate_analysis(tmp,compiledRoot,"baseline");
            reports.baseline = tmp;
            fprintf('\nBaseline aggregate table:\n');
            disp(tmp.formattedTable);
            fprintf('\nBaseline manuscript contrasts from aggregate means:\n');
            disp(tmp.trends);

        case "objective_screening"
            [~,tmp] = evalc('run_reviewer2_objective_screening_pipeline(false)');
            tmp = relocate_analysis(tmp,compiledRoot,"objective_screening");
            reports.objective_screening = tmp;
            fprintf('\nGA objective/screening aggregate results:\n');
            disp(format_objective_screening_for_console(tmp.results));
            fprintf('\nScreening ON/OFF aggregate contrasts:\n');
            disp(tmp.screeningContrasts);
            fprintf('\nObjective-component metric winners from aggregate means:\n');
            disp(tmp.componentWinners);
    end
    close all force;
    report = reports.(study_field(study));
    fprintf('Analysis: %s\n',report.analysisDirectory);
    fprintf('Figures:  %s\n',report.figureDirectory);
    fprintf('<<< %s complete in %.1f s\n',upper(strrep(study,'_',' ')),toc(started));
end

set(groot,'defaultFigureVisible',originalFigureVisible);

if saveFigures
    fprintf('\n>>> Creating curated journal figures\n');
    reports.paperFigureManifest = make_reviewer2_final_figures(reports,true);
    fprintf('<<< Curated journal figures complete\n');
else
    reports.paperFigureManifest = table();
end

fprintf('\nAll selected result processors completed successfully.\n');
fprintf(['Metric/ranking claims use aggregate mean +/- sample standard deviation. ' ...
    'Representative geometry seeds are recorded only for traceability.\n']);
fprintf(['Trajectory figures are emitted only for comparison and baseline. ' ...
    'All manuscript EPS/PNG files are stored in each study''s figures/ folder. ' ...
    'For local baseline Monte Carlo validation, run ' ...
    'run_reviewer2_baseline_monte_carlo separately.\n']);
end


function tmp = relocate_analysis(tmp,compiledRoot,studyName)
source = string(tmp.analysisDirectory);
assert(isfolder(source),'Pipeline analysis directory does not exist: %s',source);
stamp = string(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
target = string(fullfile(compiledRoot,studyName+"_"+stamp));
assert(~isfolder(target),'Timestamped compiled-results folder already exists: %s',target);
[ok,msg] = movefile(char(source),char(target));
assert(ok,'Could not move processed analysis to %s: %s',target,msg);
tmp.analysisDirectory = target;
tmp.figureDirectory = string(fullfile(char(target),'figures'));
if ~isfolder(tmp.figureDirectory), mkdir(tmp.figureDirectory); end
end


function field = study_field(study)
if study == "objective_screening", field = "objective_screening";
else, field = study; end
end


function R = attach_comparison_keys(R,S)
keys = strings(height(R),1);
for k = 1:height(R)
    row = S(S.mission == R.Mission(k) & S.optimizer == R.Optimizer(k),:);
    assert(height(row) == 1, ...
        'Missing comparison summary key for %s/%s.',R.Mission(k),R.Optimizer(k));
    keys(k) = string(row.comparison_key);
end
R.ComparisonKey = keys;
end


function T = format_objective_screening_for_console(R)
T = table(string(R.Mission),string(R.Configuration),R.NRuns, ...
    compose('%.6g +/- %.3g',R.BestJMean,R.BestJStd), ...
    compose('%.5g +/- %.3g',R.RMSEPosMean_km,R.RMSEPosStd_km), ...
    compose('%.5g +/- %.3g',R.EffectiveSigmaPosMean_km,R.EffectiveSigmaPosStd_km), ...
    compose('%.5g +/- %.3g',R.MeanStabilityMean,R.MeanStabilityStd), ...
    compose('%.5g +/- %.3g',R.ScreeningMean,R.ScreeningStd), ...
    'VariableNames',{'Mission','Configuration','Runs','Objective', ...
    'RMSEPosition_km','EffectiveSigmaPosition_km','MeanStability', ...
    'RejectedMeasurementOpportunities'});
end
