function style = reviewer2_paper_style()
%REVIEWER2_PAPER_STYLE Shared journal-figure styling for Reviewer 2 results.
%
% Manuscript formatting is applied while figures are constructed. Export
% functions must not resize/reflow completed figures or modify their axes,
% legends, cameras, clipping, or paper geometry.

style.fontName = 'Times New Roman';
style.fontWeight = 'bold';

% Manuscript readability. These values intentionally reproduce the larger
% visual scale of the earlier readable figures while leaving room for ticks,
% labels, and legends inside the exported canvas.
style.manuscriptPanelWidth = 3.3;
style.metricPlotPosition = [0.18 0.23 0.76 0.52];
style.legendMaxColumns = 8;  % start row-oriented; wrap only when space requires it
style.groupedBarWidth = 0.64;
style.categoryLabelAngle = 25;
style.fontSize = 22;
style.labelFontSize = 24;
style.lineWidth = 1.8;
style.axisLineWidth = 1.35;
style.markerSize = 5.5;
style.capSize = 7;
style.alphaBand = 0.16;
style.exportDpi = 300;

% -------------------------------------------------------------------------
% Export-size classes. Figures intended to line up in LaTeX share exactly
% the same outer paper size within each class.
% -------------------------------------------------------------------------
style.metricFigureWidth = 6.5;
style.metricFigureHeight = 5.8;
style.figureWidth = style.metricFigureWidth;
style.figureHeight = style.metricFigureHeight;
style.panelFigureHeight = 6.2; % retained only for backward compatibility
style.convergenceFigureWidth = style.metricFigureWidth;
style.convergenceFigureHeight = style.metricFigureHeight;

style.geometryFigureWidth = 7.6;
style.geometryFigureHeight = 7.0;
style.orbitFamilyFigureWidth = style.geometryFigureWidth;
style.orbitFamilyFigureHeight = style.geometryFigureHeight;
style.slotPhaseFigureWidth = style.geometryFigureWidth;
style.slotPhaseFigureHeight = style.geometryFigureHeight;

% Keep the occlusion/keepout schematic at its established size. The
% definition renderer also pins that schematic to its existing 12 pt text.
style.visibilityFigureWidth = 7.2;
style.visibilityFigureHeight = 5.1;

% RA and Dec deliberately share the same outer canvas and axes geometry.
style.measurementFigureWidth = style.metricFigureWidth;
style.measurementFigureHeight = style.metricFigureHeight;
style.monteCarloFigureWidth = style.metricFigureWidth;
style.monteCarloFigureHeight = style.metricFigureHeight;

% Shared 3-D layout and fallback camera.
style.geometryPlotPosition = [0.14 0.23 0.72 0.54];
style.geometryLegendGap = 0.012;
style.geometryAzimuth = -37.5;
style.geometryElevation = 30;
style.geometryProjection = 'perspective';
style.geometryXPadding = 0.08;
style.geometryYPadding = 0.10;
style.geometryZPadding = 0.10;

% Orbit-family study-definition cameras.
style.orbitFamilyViews.northern_halo = [-37.5 30];
style.orbitFamilyViews.southern_halo = [-37.5 30];
style.orbitFamilyViews.northern_rectilinear = [-37.5 30];
style.orbitFamilyViews.southern_rectilinear = [-37.5 30];
style.orbitFamilyViews.dro_family = [0 90];

style.orbitFamilyProjections.northern_halo = 'perspective';
style.orbitFamilyProjections.southern_halo = 'perspective';
style.orbitFamilyProjections.northern_rectilinear = 'perspective';
style.orbitFamilyProjections.southern_rectilinear = 'perspective';
style.orbitFamilyProjections.dro_family = 'orthographic';

% Target/maneuver trajectory cameras.
style.maneuverViews.LUNAR_GATEWAY = [-37.5 30];
style.maneuverViews.LOW_THRUST_TRANSFER = [-37.5 35];
style.maneuverViews.GATEWAY_IMPULSE = [-37.5 30];

style.maneuverProjections.LUNAR_GATEWAY = 'perspective';
style.maneuverProjections.LOW_THRUST_TRANSFER = 'perspective';
style.maneuverProjections.GATEWAY_IMPULSE = 'perspective';

% Fixed qualitative palette so optimizer identity never changes by figure.
style.optimizerOrder = ["GA","PSO","BAYESIAN","ABC","ACO"];
style.optimizerColors = [ ...
    0.0000 0.4470 0.7410; ...
    0.8500 0.3250 0.0980; ...
    0.9290 0.6940 0.1250; ...
    0.4940 0.1840 0.5560; ...
    0.4660 0.6740 0.1880];

style.measurementOrder = ["ANGLES_ONLY","ANGLES_RANGE"];
style.measurementColors = style.optimizerColors(1:2,:);

style.configurationOrder = [ ...
    "combined_on","combined_off","j1_only","j2_only","j3_only"];
style.configurationColors = style.optimizerColors;

style.missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
style.missionColors = [ ...
    reviewer2_target_color("LUNAR_GATEWAY"); ...
    reviewer2_target_color("LOW_THRUST_TRANSFER"); ...
    reviewer2_target_color("GATEWAY_IMPULSE")];
end
