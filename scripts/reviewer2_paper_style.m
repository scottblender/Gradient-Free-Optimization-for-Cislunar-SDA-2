function style = reviewer2_paper_style()
%REVIEWER2_PAPER_STYLE Shared journal-figure styling for Reviewer 2 results.
%
% The figure geometry intentionally follows the last known-good manuscript
% layout. Readability is improved with a moderate font increase and shorter
% labels rather than oversized text or export-time reformatting.

style.fontName = 'Times New Roman';
style.fontWeight = 'bold';

% Typography: larger than the original 12/14 pt manuscript figures, but
% small enough that tick labels and axis labels remain inside the EPS canvas.
style.fontSize = 16;
style.labelFontSize = 18;
style.legendFontSize = 15;
style.lineWidth = 1.8;
style.axisLineWidth = 1.35;
style.markerSize = 5.5;
style.capSize = 7;
style.alphaBand = 0.16;
style.exportDpi = 300;

% Dense categorical figures retain enough margin for labels while using more
% of the available canvas than the previous oversized-font layout.
style.manuscriptPanelWidth = 3.3;
style.metricPlotPosition = [0.16 0.18 0.80 0.62];
style.legendMaxColumns = 8;  % prefer a single row; wrap only when necessary
style.groupedBarWidth = 0.64;
style.categoryLabelAngle = 18;

% -------------------------------------------------------------------------
% Export-size classes. These return to the working original proportions.
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

% RA and Dec remain matched, but use a compact manuscript panel rather than
% the oversized metric canvas introduced during the failed formatting pass.
style.measurementFigureWidth = 5.2;
style.measurementFigureHeight = 4.8;
style.monteCarloFigureWidth = 4.8;
style.monteCarloFigureHeight = 4.2;

% Shared 3-D layout restored from the known-good manuscript geometry.
style.geometryPlotPosition = [0.12 0.20 0.76 0.64];
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
