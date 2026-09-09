function manifest = make_reviewer2_final_figures(reports,saveFigures)
%MAKE_REVIEWER2_FINAL_FIGURES Compatibility entry point for final figures.
%
% The manuscript renderer is intentionally curated in
% make_reviewer2_curated_figures.m so only figures that communicate core
% Reviewer-2 results are emitted. Keep this public entry point stable for the
% results runner and existing user workflows.

if nargin < 2 || isempty(saveFigures), saveFigures = true; end
manifest = make_reviewer2_curated_figures(reports,saveFigures);
end
