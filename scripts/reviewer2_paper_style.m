function style = reviewer2_paper_style()
%REVIEWER2_PAPER_STYLE Shared journal-figure styling for Reviewer 2 results.
%
% The figure geometry follows the last known-good manuscript layout. The
% manuscript pass increases readability with moderate fonts, compact labels,
% tighter 3-D framing, and explicit margins rather than export-time reflow.

style.fontName = 'Times New Roman';
style.fontWeight = 'bold';

% Typography: larger than the original 12/14 pt manuscript figures, but
% compact enough that tick labels and axis labels remain inside the EPS canvas.
style.fontSize = 16;
style.labelFontSize = 18;
style.legendFontSize = 14;
style.geometryLegendFontSize = 15; % slightly larger only for 3-D geometry legends
style.lineWidth = 1.8;
style.axisLineWidth = 1.35;
style.markerSize = 5.5;
style.capSize = 7;
style.alphaBand = 0.16;
style.exportDpi = 300;

% Dense categorical figures retain enough margin for labels while using more
% of the available canvas than the oversized-font layout.
style.manuscriptPanelWidth = 3.3;
style.metricPlotPosition = [0.16 0.18 0.80 0.62];
style.legendMaxColumns = 8;  % prefer a single row; wrap only when necessary
style.groupedBarWidth = 0.64;
style.categoryLabelAngle = 30;

% -------------------------------------------------------------------------
% Export-size classes. These retain the working original proportions.
% -------------------------------------------------------------------------
style.metricFigureWidth = 6.5;
style.metricFigureHeight = 4.6;
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

% Keep the occlusion/keepout schematic exactly on its established canvas.
style.visibilityFigureWidth = 7.2;
style.visibilityFigureHeight = 5.1;

% RA and Dec remain matched and use a compact manuscript panel.
style.measurementFigureWidth = 5.2;
style.measurementFigureHeight = 4.8;
style.monteCarloFigureWidth = 4.8;
style.monteCarloFigureHeight = 4.2;

% Shared 3-D layout. The x-position and width are exactly symmetric about
% the exported canvas center (0.09 + 0.82/2 = 0.50). Tight axis padding keeps
% the trajectories large without changing camera geometry or EPS export.
style.geometryPlotPosition = [0.09 0.16 0.82 0.70];
style.geometryLegendGap = 0.006;
style.geometryAzimuth = -37.5;
style.geometryElevation = 30;
style.geometryProjection = 'perspective';
style.geometryXPadding = 0.04;
style.geometryYPadding = 0.05;
style.geometryZPadding = 0.05;

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
