function details = plot_reviewer2_constellation_geometry(selection,figureDir,stemPrefix,saveFigures)
%PLOT_REVIEWER2_CONSTELLATION_GEOMETRY Backward-compatible geometry exporter.
%
% Final manuscript geometry is rendered centrally by
% make_reviewer2_paper_figures using representative realizations nearest the
% 20-run group means. Individual processing pipelines call this function with
% saveFigures=false, so data-only processing must not create legacy preview
% figures or spend time propagating representative observer orbits.

if nargin < 4 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);

if ~saveFigures
    details = empty_geometry_details();
    return;
end

details = plot_reviewer2_geometry_grid( ...
    selection,figureDir,stemPrefix,saveFigures);
end


function details = empty_geometry_details()
details = table(strings(0,1),strings(0,1),strings(0,1), ...
    nan(0,1),nan(0,1),nan(0,1),nan(0,1),strings(0,1), ...
    nan(0,1),strings(0,1),strings(0,1),strings(0,1), ...
    strings(0,1),strings(0,1), ...
    'VariableNames',{'Mission','PanelKey','PanelLabel', ...
    'RepresentativeObjective','GroupMeanObjective','GroupStdObjective', ...
    'RepresentativeSeed','RunFile','NumObservers','OrbitFamilies', ...
    'OrbitIndices','SlotIndices','FigureStem','GridFigureStem'});
end
