function reports = run_reviewer2_results(studies,saveFigures)
%RUN_REVIEWER2_RESULTS Process any or all completed Reviewer 2 studies.
%
% This runner never launches optimization. It validates saved schema-v2
% results, builds manuscript tables, and creates journal-ready figures.
%
% Examples:
%   reports = run_reviewer2_results;                       % everything
%   reports = run_reviewer2_results("comparison");        % one study
%   reports = run_reviewer2_results("baseline",false);    % display only
%   reports = run_reviewer2_results(["runtime","objective_screening"]);
%
% Selectors:
%   runtime             - focused 1200-FE five-method/BO study
%   comparison          - full 6000-FE four-method comparison
%   baseline            - GA AO/AR, observer-count, and duration study
%   objective_screening - GA objective-component and screening study
%   all                 - all four processors (default)

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
fprintf('Save figures:     %s\n\n',string(saveFigures));

for study = studies
    started = tic;
    fprintf('\n>>> Processing %s\n',upper(strrep(study,'_',' ')));
    switch study
        case "runtime"
            reports.runtime = run_reviewer2_runtime_pipeline(saveFigures);
        case "comparison"
            reports.comparison = run_reviewer2_comparison_pipeline(saveFigures);
        case "baseline"
            reports.baseline = run_reviewer2_baseline_pipeline(saveFigures);
        case "objective_screening"
            reports.objective_screening = ...
                run_reviewer2_objective_screening_pipeline(saveFigures);
    end
    fprintf('<<< %s complete in %.1f s\n',upper(strrep(study,'_',' ')),toc(started));
end

fprintf('\nAll selected result processors completed successfully.\n');
end
