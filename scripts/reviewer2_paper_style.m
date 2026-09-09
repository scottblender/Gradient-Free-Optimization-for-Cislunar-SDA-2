function style = reviewer2_paper_style()
%REVIEWER2_PAPER_STYLE Shared journal-figure styling for Reviewer 2 results.
%
% The 3-D result figures intentionally match the tracking-case figures in
% plot_study_definition_figures.m: 7.6 x 7.0 inch canvas, centered inner
% axes box, manuscript camera, and Times New Roman with 12-point minimum
% text. Two-dimensional result figures are standalone 6.5-inch-wide paper
% figures so LaTeX can assemble them with subfigure/subcaption as needed.

style.fontName = 'Times New Roman';
style.fontSize = 12;
style.labelFontSize = 14;
style.lineWidth = 1.8;
style.axisLineWidth = 1.35;
style.markerSize = 5.5;
style.capSize = 7;
style.alphaBand = 0.16;
style.exportDpi = 300;

% Standalone journal-sized 2-D figures.
style.figureWidth = 6.5;
style.figureHeight = 4.6;
style.panelFigureHeight = 6.2; % retained only for backward compatibility
style.convergenceFigureWidth = 6.5;
style.convergenceFigureHeight = 4.6;

% Shared 3-D layout used by the manuscript trajectory/result figures.
% The low-thrust study-definition panel has a small local camera override
% in plot_study_definition_figures.m; all other 3-D figures use this view.
style.geometryFigureWidth = 7.6;
style.geometryFigureHeight = 7.0;
style.geometryPlotPosition = [0.12 0.20 0.76 0.64];
style.geometryLegendGap = 0.012;
style.geometryAzimuth = -37.5;
style.geometryElevation = 30;
style.geometryXPadding = 0.08;
style.geometryYPadding = 0.10;
style.geometryZPadding = 0.10;

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
