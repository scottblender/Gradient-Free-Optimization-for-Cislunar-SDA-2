function details = plot_reviewer2_constellation_geometry(selection,figureDir,stemPrefix,saveFigures)
%PLOT_REVIEWER2_CONSTELLATION_GEOMETRY Backward-compatible geometry exporter.
%
% Final Reviewer 2 constellation figures are rendered by
% plot_reviewer2_geometry_grid so comparison and baseline pipelines share a
% compact journal layout with common per-mission limits, a centered 3-D
% camera, solid orbit lines, and 12-point minimum text.

if nargin < 4 || isempty(saveFigures), saveFigures = true; end
details = plot_reviewer2_geometry_grid( ...
    selection,figureDir,stemPrefix,saveFigures);
end
