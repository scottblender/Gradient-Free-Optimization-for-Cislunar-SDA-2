function reports = run_reviewer2_results(studies,saveFigures)
%RUN_REVIEWER2_RESULTS Process completed Reviewer 2 studies and final figures.
%
% This runner never launches optimization. It validates saved schema-v2
% results, builds manuscript tables, then creates one curated, consistent
% paper-final figure set across the selected studies.
%
% Examples:
%   reports = run_reviewer2_results;                       % everything
%   reports = run_reviewer2_results("comparison");        % one study
%   reports = run_reviewer2_results("baseline",false);    % tables only
%   reports = run_reviewer2_results(["runtime","objective_screening"]);
%
% Selectors:
%   runtime             - focused 1200-FE five-method/BO study
%   comparison          - full 6000-FE four-method comparison
%   baseline            - GA AO/AR, observer-count, and duration study
%   objective_screening - GA objective-component and screening study
%   all                 - all four processors (default)
%
% When saveFigures is true, the individual processors are run in data-only
% mode and make_reviewer2_paper_figures creates the final manuscript plots.
% This avoids keeping multiple generations of nearly redundant preview plots.

if nargin < 1 || isempty(studies), studies = "all"; end
if nargin < 2 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);
studies = lower(string(studies(:)'));
assert(~isempty(studies) && all(strlength(studies) > 0), ...
    'At least one nonempty study selector is required.');

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
fprintf('Save curated paper figures: %s\n\n',string(saveFigures));

% Each processor still performs all scientific validation and writes its CSV
% outputs. The centralized paper renderer is the only source of final plots.
for study = studies
    started = tic;
    fprintf('\n>>> Processing %s\n',upper(strrep(study,'_',' ')));
    switch study
        case "runtime"
            reports.runtime = run_reviewer2_runtime_pipeline(false);
        case "comparison"
            reports.comparison = run_reviewer2_comparison_pipeline(false);
        case "baseline"
            reports.baseline = run_reviewer2_baseline_pipeline(false);
        case "objective_screening"
            reports.objective_screening = ...
                run_reviewer2_objective_screening_pipeline(false);
    end
    close all force;
    fprintf('<<< %s complete in %.1f s\n',upper(strrep(study,'_',' ')),toc(started));
end

if saveFigures
    fprintf('\n>>> Creating curated journal figures\n');
    reports.paperFigureManifest = make_reviewer2_paper_figures(reports,true);

    % Overwrite the generic runtime summary with the reviewer-facing equal-FE
    % cost/benefit panel that reports BO's objective gap and runtime penalty.
    if isfield(reports,'runtime')
        plot_reviewer2_runtime_summary(reports.runtime,true);
    end

    % Replace the generic screening panel with the four quantities used to
    % make the screening conclusion directly: total J111 objective, position
    % RMSE, equal-FE runtime, and rejected measurement opportunities.
    if isfield(reports,'objective_screening')
        plot_reviewer2_screening_summary(reports.objective_screening,true);
    end

    if isfield(reports,'baseline')
        durationStem = plot_reviewer2_baseline_duration_matrix(reports.baseline,true);
        durationRow = table("baseline",string(durationStem), ...
            "Gateway duration and observer-count interaction for AO and AR.", ...
            'VariableNames',reports.paperFigureManifest.Properties.VariableNames);
        reports.paperFigureManifest = [reports.paperFigureManifest;durationRow];
        baselineRows = reports.paperFigureManifest( ...
            reports.paperFigureManifest.Study == "baseline",:);
        writetable(baselineRows,fullfile( ...
            char(reports.baseline.analysisDirectory),'paper_figure_manifest.csv'));
    end

    if isfield(reports,'comparison')
        rankingStem = plot_reviewer2_optimizer_ranking(reports.comparison,true);
        rankingRow = table("comparison",string(rankingStem), ...
            "Overall objective rank and target-case win count.", ...
            'VariableNames',reports.paperFigureManifest.Properties.VariableNames);
        reports.paperFigureManifest = [reports.paperFigureManifest;rankingRow];
        comparisonRows = reports.paperFigureManifest( ...
            reports.paperFigureManifest.Study == "comparison",:);
        writetable(comparisonRows,fullfile( ...
            char(reports.comparison.analysisDirectory),'paper_figure_manifest.csv'));
    end
    fprintf('<<< Curated journal figures complete\n');
else
    reports.paperFigureManifest = table();
end

fprintf('\nAll selected result processors completed successfully.\n');
if saveFigures
    fprintf(['Final manuscript figures are under the newest FE_DATA_*/paper_final ' ...
        'directory for each selected study.\n']);
end
end
