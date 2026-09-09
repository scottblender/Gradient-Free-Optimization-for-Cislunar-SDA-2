function reports = run_reviewer2_results(studies,saveFigures)
%RUN_REVIEWER2_RESULTS Process completed Reviewer 2 studies and final figures.
%
% This runner never launches optimization. Each selected study is validated
% and reduced to aggregate statistics first. Statistical comparisons printed
% here and used in manuscript figures are group mean +/- sample standard
% deviation across the 20 independent runs. A representative seed is used
% only to visualize one realizable discrete constellation near the group mean.
%
% Legacy per-pipeline preview figures are hidden during data processing. The
% only manuscript figures users should inspect are the centralized outputs in
% each newest FE_DATA_*/paper_final directory.

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

setup_project();
reports = struct();
fprintf('\n=== Reviewer 2 results processing ===\n');
fprintf('Selected studies: %s\n',strjoin(cellstr(studies),', '));
fprintf('Save curated paper figures: %s\n',string(saveFigures));
fprintf('Reported performance statistics: 20-run mean +/- sample standard deviation\n\n');

% The individual processors still contain some historical preview routines.
% Hide figures while they validate/aggregate data so those obsolete plots do
% not appear and cannot be mistaken for the final centralized figures.
originalFigureVisible = get(groot,'defaultFigureVisible');
visibilityCleanup = onCleanup(@() set(groot,'defaultFigureVisible',originalFigureVisible)); %#ok<NASGU>
set(groot,'defaultFigureVisible','off');

for study = studies
    started = tic;
    fprintf('\n>>> Processing %s\n',upper(strrep(study,'_',' ')));
    switch study
        case "runtime"
            [~,tmp] = evalc('run_reviewer2_runtime_pipeline(false)');
            reports.runtime = tmp;
            fprintf('\n1200-FE aggregate objective/runtime table:\n');
            disp(tmp.formattedTable);
            fprintf('\nBayesian equal-FE runtime comparison:\n');
            disp(tmp.boSlowdown);
            fprintf('\nEqual-FE conclusion metrics:\n');
            disp(tmp.conclusion);

        case "comparison"
            [~,tmp] = evalc('run_reviewer2_comparison_pipeline(false)');
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
            reports.baseline = tmp;
            fprintf('\nBaseline aggregate table:\n');
            disp(tmp.formattedTable);
            fprintf('\nBaseline manuscript contrasts from aggregate means:\n');
            disp(tmp.trends);

        case "objective_screening"
            [~,tmp] = evalc('run_reviewer2_objective_screening_pipeline(false)');
            reports.objective_screening = tmp;
            fprintf('\nGA objective/screening aggregate results:\n');
            disp(format_objective_screening_for_console(tmp.results));
            fprintf('\nScreening ON/OFF aggregate contrasts:\n');
            disp(tmp.screeningContrasts);
            fprintf('\nObjective-component metric winners from aggregate means:\n');
            disp(tmp.componentWinners);
    end
    close all force;
    fprintf('<<< %s complete in %.1f s\n',upper(strrep(study,'_',' ')),toc(started));
end

% Restore normal MATLAB figure visibility before the centralized paper
% renderer. These are the only figures intended for inspection/publication.
set(groot,'defaultFigureVisible',originalFigureVisible);

if saveFigures
    fprintf('\n>>> Creating curated journal figures\n');
    reports.paperFigureManifest = make_reviewer2_paper_figures(reports,true);
    fprintf('<<< Curated journal figures complete\n');
else
    reports.paperFigureManifest = table();
end

fprintf('\nAll selected result processors completed successfully.\n');
if saveFigures
    fprintf(['Final manuscript figures are under the newest FE_DATA_*/paper_final ' ...
        'directory for each selected study.\n']);
    fprintf(['Metric/ranking claims use aggregate mean +/- sample standard deviation. ' ...
        'Representative geometry seeds are recorded only for traceability.\n']);
end
end


function R = attach_comparison_keys(R,S)
% The comparison aggregate table historically omitted comparison_key even
% though the convergence MAT files are keyed by it. Attach the validated key
% to each mission/optimizer row before the centralized renderer runs.
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
    compose('%.4f +/- %.3f',R.CoverageMean,R.CoverageStd), ...
    'VariableNames',{'Mission','Configuration','Runs','Objective', ...
    'RMSEPosition_km','EffectiveSigmaPosition_km','MeanStability', ...
    'CoverageFraction'});
end
