function style = reviewer2_paper_style()
%REVIEWER2_PAPER_STYLE Shared journal-figure styling for Reviewer 2 results.
%
% All final Reviewer 2 figures use Times New Roman, 12-point minimum text,
% 14-point axis labels, consistent optimizer/measurement colors, and the
% target-case colors used by the study-definition figures.

style.fontName = 'Times New Roman';
style.fontSize = 12;
style.labelFontSize = 14;
style.lineWidth = 1.8;
style.axisLineWidth = 1.0;
style.markerSize = 6;
style.capSize = 7;
style.alphaBand = 0.14;
style.figureWidth = 7.2;
style.figureHeight = 4.8;
style.panelFigureHeight = 6.4;
style.geometryFigureWidth = 7.2;
style.geometryPanelHeight = 3.0;
style.exportDpi = 300;

% MATLAB default qualitative palette, fixed here so every study uses the
% same algorithm colors.
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
