function manifest = make_reviewer2_final_figures(reports,saveFigures)
%MAKE_REVIEWER2_FINAL_FIGURES Stable entry point for curated final figures.
%
% The focused 1200-FE runtime figures use a dedicated renderer because the
% flat-colored categorical bar charts have one MATLAB Bar handle but five
% optimizer categories. Their optimizer identities are carried by x-axis
% labels, while the only bar-chart legend entry is the Baseline AO reference.
% All remaining Reviewer-2 figures are produced by the curated renderer.

if nargin < 2 || isempty(saveFigures), saveFigures = true; end
saveFigures = logical(saveFigures);
assert(isstruct(reports),'reports must come from run_reviewer2_results.');

manifest = table(strings(0,1),strings(0,1),strings(0,1), ...
    'VariableNames',{'Study','FigureStem','Purpose'});
remaining = reports;

if isfield(reports,'runtime')
    baselineResults = table();
    if isfield(reports,'baseline') && isfield(reports.baseline,'results')
        baselineResults = reports.baseline.results;
    end
    runtimeManifest = make_reviewer2_runtime_figures( ...
        reports.runtime,baselineResults,saveFigures);
    manifest = [manifest;runtimeManifest]; %#ok<AGROW>
    remaining = rmfield(remaining,'runtime');
end

if ~isempty(fieldnames(remaining))
    otherManifest = make_reviewer2_curated_figures(remaining,saveFigures);
    manifest = [manifest;otherManifest]; %#ok<AGROW>
end
end
