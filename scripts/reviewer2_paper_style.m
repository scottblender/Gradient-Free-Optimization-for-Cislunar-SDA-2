function style = reviewer2_paper_style()
%REVIEWER2_PAPER_STYLE Shared journal-figure styling for Reviewer 2 results.
%
% This file is the single source of manuscript styling constants. Plotters
% apply these values while constructing figures; the EPS exporter must not
% resize/reflow figures after generation.

style.fontName = 'Times New Roman';
style.fontWeight = 'bold';
% At the intended paired-panel placement, these export sizes retain
% approximately 10-point manuscript text.
style.manuscriptPanelWidth = 3.3;
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
% Export-size classes.
% -------------------------------------------------------------------------
% Quantitative figures get enough vertical room for labels and an outside
% legend without crowding the axes.
style.metricFigureWidth = 7.25;
style.metricFigureHeight = 5.8;
style.figureWidth = style.metricFigureWidth;
style.figureHeight = style.metricFigureHeight;
style.panelFigureHeight = 6.8; % compatibility alias
style.convergenceFigureWidth = style.metricFigureWidth;
style.convergenceFigureHeight = style.metricFigureHeight;

% Restore the larger pre-runner 3-D canvas that exported correctly before
% the shared-export rewrite. This also gives the north-outside legend room
% to remain horizontal rather than wrapping into multiple rows.
style.geometryFigureWidth = 7.6;
style.geometryFigureHeight = 7.0;
style.orbitFamilyFigureWidth = style.geometryFigureWidth;
style.orbitFamilyFigureHeight = style.geometryFigureHeight;
style.slotPhaseFigureWidth = style.geometryFigureWidth;
style.slotPhaseFigureHeight = style.geometryFigureHeight;

% The keep-out schematic needs additional room so the geometry itself can be
% larger while retaining all callouts inside the exported page.
style.visibilityFigureWidth = 7.6;
style.visibilityFigureHeight = 6.4;
style.measurementFigureWidth = style.metricFigureWidth;
style.measurementFigureHeight = style.metricFigureHeight;
style.monteCarloFigureWidth = style.metricFigureWidth;
style.monteCarloFigureHeight = style.metricFigureHeight;

% -------------------------------------------------------------------------
% Shared layouts.
% -------------------------------------------------------------------------
style.geometryPlotPosition = [0.12 0.18 0.76 0.64];
style.metricPlotPosition = [0.16 0.18 0.79 0.60];
style.schematicPlotPosition = [0.05 0.06 0.90 0.88];
style.measurementXLim = [-0.95 5.10];
style.measurementYLim = [-0.80 4.20];
style.measurementAxisLength = [4.25 3.45];

% Legends are created at northoutside by each plotter, then moved only by
% the plotter's own centered-layout helper. Five/six-item geometry legends
% should remain in one row whenever the canvas permits it.
style.legendMaxColumns = 6;
style.geometryLegendGap = 0.008;

% Desired tick density. Non-trajectory quantitative renderers may use these
% limits; 3-D geometry plots should remain sparse to prevent projected-label
% collisions.
style.max2DXTicks = 8;
style.max2DYTicks = 7;
style.max3DXTicks = 3;
style.max3DYTicks = 2;
style.max3DZTicks = 3;

% Slot-definition visibility. The neutral candidate markers are intended to
% be filled with this gray while the excluded endpoint remains hollow.
style.slotCandidateFillColor = [0.72 0.72 0.72];

% Shared 3-D fallback camera and padding.
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
