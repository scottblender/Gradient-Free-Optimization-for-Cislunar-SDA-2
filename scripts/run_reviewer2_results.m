function reports = run_reviewer2_results(studies,saveFigures)
%RUN_REVIEWER2_RESULTS Process completed Reviewer 2 studies and final figures.
%
% This runner never launches optimization. Each selected study is validated
% and reduced to aggregate statistics first. When saveFigures is true, one
% centralized renderer creates the final journal figures from those aggregate
% reports. Statistical comparisons use mean +/- sample standard deviation;
% geometry panels are representative realizations chosen near group means.
%
% Examples:
%   reports = run_reviewer2_results;
%   reports = run_reviewer2_results("comparison");
%   reports = run_reviewer2_results("baseline",false);
%   reports = run_reviewer2_results(["runtime","objective_screening"]);

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
fprintf('Save curated paper figures: %s\n\n',string(saveFigures));

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
            reports.objective_screening = run_reviewer2_objective_screening_pipeline(false);
    end
    close all force;
    fprintf('<<< %s complete in %.1f s\n',upper(strrep(study,'_',' ')),toc(started));
end

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
        'Geometry CSVs identify representative seeds nearest each group mean.\n']);
end
end
