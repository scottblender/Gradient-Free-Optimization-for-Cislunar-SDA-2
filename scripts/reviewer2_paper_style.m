function style = reviewer2_paper_style()
%REVIEWER2_PAPER_STYLE Single source of truth for manuscript figure styling.
%
% All manuscript renderers should take figure size, axes geometry, typography,
% legend placement, and 3-D padding from this file. Plot-specific code may
% choose data, colors, cameras, and labels, but should not redefine layout.

style.fontName = 'Times New Roman';
style.fontWeight = 'bold';

% Typography.
style.fontSize = 16;
style.labelFontSize = 18;
style.legendFontSize = 14;
style.geometryLegendFontSize = 15;
style.legendMinFontSize = 12;
style.lineWidth = 1.8;
style.axisLineWidth = 1.35;
style.markerSize = 5.5;
style.capSize = 7;
style.alphaBand = 0.16;
style.exportDpi = 300;

% -------------------------------------------------------------------------
% One standard EPS/PNG canvas for every manuscript figure.
% Keeping the outer paper rectangle identical gives every exported figure the
% same crop in LaTeX. Specialized figure-size aliases remain for compatibility
% with existing renderers, but all resolve to this one canvas.
% -------------------------------------------------------------------------
style.figureWidth = 6.5;
style.figureHeight = 5.2;
style.metricFigureWidth = style.figureWidth;
style.metricFigureHeight = style.figureHeight;
style.convergenceFigureWidth = style.figureWidth;
style.convergenceFigureHeight = style.figureHeight;
style.geometryFigureWidth = style.figureWidth;
style.geometryFigureHeight = style.figureHeight;
style.orbitFamilyFigureWidth = style.figureWidth;
style.orbitFamilyFigureHeight = style.figureHeight;
style.slotPhaseFigureWidth = style.figureWidth;
style.slotPhaseFigureHeight = style.figureHeight;
style.visibilityFigureWidth = style.figureWidth;
style.visibilityFigureHeight = style.figureHeight;
style.measurementFigureWidth = style.figureWidth;
style.measurementFigureHeight = style.figureHeight;
style.monteCarloFigureWidth = style.figureWidth;
style.monteCarloFigureHeight = style.figureHeight;
style.panelFigureHeight = style.figureHeight; % backward compatibility
style.manuscriptPanelWidth = 3.3;

% -------------------------------------------------------------------------
% Standard 2-D layout.
% The plot top is deliberately below the legend row so long vertical y-axis
% labels cannot collide with the legend. format_manuscript_legend restores
% this rectangle after MATLAB creates the legend.
% -------------------------------------------------------------------------
style.metricPlotPosition = [0.15 0.17 0.82 0.56];
style.legend2DGap = 0.08;
style.legendMaxColumns = 8;
style.legendMaxWidth = 0.96;
style.legendItemTokenSize = [18 8];
style.groupedBarWidth = 0.64;
style.categoryLabelAngle = 30;

% -------------------------------------------------------------------------
% Standard 3-D layout.
% The axes are exactly centered horizontally. The legend intentionally sits
% inside the otherwise-unused top portion of the 3-D axes box so it stays
% visually close to the trajectories without reducing the data region.
% -------------------------------------------------------------------------
style.geometryPlotPosition = [0.10 0.14 0.80 0.70];
style.geometryLegendGap = -0.04;
style.geometryLegendItemTokenSize = [16 8];
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
