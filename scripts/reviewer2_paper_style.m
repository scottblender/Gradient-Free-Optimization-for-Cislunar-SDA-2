function style = reviewer2_paper_style()
%REVIEWER2_PAPER_STYLE Shared journal-figure styling for Reviewer 2 results.
%
% This file is the single source of manuscript figure style constants.
% export_manuscript_figure applies these settings once, immediately before
% writing EPS/PNG output. Individual plotters should define data and camera
% geometry, but should not perform a second export-time layout pass.

style.fontName = 'Times New Roman';
style.fontWeight = 'bold';
% A 6.5-inch EPS placed at 3 inches retains approximately 10-point text.
style.manuscriptPanelWidth = 3.0;
style.minimumPrintedFontSize = 10;
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
style.metricFigureHeight = 5.2;
style.figureWidth = style.metricFigureWidth;
style.figureHeight = style.metricFigureHeight;
style.panelFigureHeight = 6.2; % retained only for backward compatibility
style.convergenceFigureWidth = style.metricFigureWidth;
style.convergenceFigureHeight = style.metricFigureHeight;

style.geometryFigureWidth = style.metricFigureWidth;
style.geometryFigureHeight = style.metricFigureHeight;
style.orbitFamilyFigureWidth = style.geometryFigureWidth;
style.orbitFamilyFigureHeight = style.geometryFigureHeight;
style.slotPhaseFigureWidth = style.geometryFigureWidth;
style.slotPhaseFigureHeight = style.geometryFigureHeight;
% The keep-out schematic needs extra vertical room so the equal-axis
% geometry and callouts use more of the EPS at the same manuscript width.
style.visibilityFigureWidth = style.metricFigureWidth;
style.visibilityFigureHeight = 6.4;
style.measurementFigureWidth = style.metricFigureWidth;
style.measurementFigureHeight = style.metricFigureHeight;
style.monteCarloFigureWidth = style.metricFigureWidth;
style.monteCarloFigureHeight = style.metricFigureHeight;

% Stable runner-era axes layouts. Export does not camera-zoom or rescale
% trajectory geometry after the plotter has chosen its data limits/camera.
style.geometryPlotPosition = [0.15 0.18 0.72 0.60];
style.metricPlotPosition = [0.18 0.22 0.77 0.55];
style.schematicPlotPosition = [0.08 0.10 0.84 0.80];
style.measurementXLim = [-0.95 5.10];
style.measurementYLim = [-0.80 4.20];
style.measurementAxisLength = [4.25 3.45];

% Legend policy. MATLAB first computes northoutside; the master formatter
% nudges it slightly downward, preserves a safe axes gap, and freezes it.
style.legendMaxColumns = 6;
style.legendMaxRows = 2;
style.legendWidthLimit = 0.94;
style.legendNorthOutsideYOffset = -0.015;
style.legendMinimumGap = 0.008;
style.geometryLegendGap = 0.012; % live-figure compatibility only

% Non-trajectory numeric plots need enough ticks for quantitative reading.
style.max2DXTicks = 8;
style.max2DYTicks = 7;

% Slot-definition visibility. Selected/adjacent slots keep their distinct
% colors; neutral candidate slots use this filled marker color in export.
style.slotCandidateFillColor = [0.72 0.72 0.72];

style.geometryAzimuth = -37.5;
style.geometryElevation = 30;
style.geometryProjection = 'perspective';
style.geometryXPadding = 0.08;
style.geometryYPadding = 0.10;
style.geometryZPadding = 0.10;

% -------------------------------------------------------------------------
% Camera controls: orbit-family study-definition figures.
% -------------------------------------------------------------------------
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

% -------------------------------------------------------------------------
% Camera controls: target/maneuver trajectory figures.
% -------------------------------------------------------------------------
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
