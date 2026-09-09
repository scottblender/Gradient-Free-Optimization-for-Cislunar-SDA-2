function style = reviewer2_paper_style()
%REVIEWER2_PAPER_STYLE Shared journal-figure styling for Reviewer 2 results.
%
% Final Reviewer 2 figures are sized for a journal column/page workflow and
% use Times New Roman with 12-point minimum text. The geometry figures use a
% fixed 6.5 x 6.5 inch canvas so every 3-D comparison has identical export
% dimensions and enough physical margin for labels.

style.fontName = 'Times New Roman';
style.fontSize = 12;
style.labelFontSize = 14;
style.lineWidth = 1.8;
style.axisLineWidth = 1.0;
style.markerSize = 5.5;
style.capSize = 7;
style.alphaBand = 0.16;
style.exportDpi = 300;

% Journal-sized 2-D figures.
style.figureWidth = 6.5;
style.figureHeight = 4.6;
style.panelFigureHeight = 6.2;
style.convergenceFigureWidth = 6.5;
style.convergenceFigureHeight = 6.2;

% Fixed geometry export requested for the manuscript.
style.geometryFigureWidth = 6.5;
style.geometryFigureHeight = 6.5;
style.exportPaddingIn = 0.25;
style.geometryAzimuth = -37.5;
style.geometryElevation = 30;

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
