function manifest = plot_screening_velocity_metrics(report,saveFigures)
%PLOT_SCREENING_VELOCITY_METRICS Add velocity diagnostics for screening ON/OFF.
%
% These plots use the saved per-run validation metrics, so no optimization or
% tracking rerun is required. They complement the position-domain screening
% panels and make the low-thrust objective difference directly interpretable.

if nargin < 2 || isempty(saveFigures), saveFigures = true; end
saveFigures = logical(saveFigures);
assert(isstruct(report) && isfield(report,'results') && isfield(report,'runMetrics'), ...
    'Expected the objective_screening report from run_reviewer2_results.');

R = report.results;
M = report.runMetrics;
requiredRunMetrics = ["comparison_key","rmse_vel_kms", ...
    "mean_effective_sigma_vel_kms"];
assert(all(ismember(requiredRunMetrics,string(M.Properties.VariableNames))), ...
    'Saved objective/screening run metrics do not contain velocity diagnostics.');

missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
configs = ["combined_on","combined_off"];
labels = ["Screening ON","Screening OFF"];
style = reviewer2_paper_style();
out = string(report.analysisDirectory);
assert(isfolder(out),'Objective/screening analysis directory does not exist.');

rmseMean = nan(3,2); rmseStd = nan(3,2);
sigmaMean = nan(3,2); sigmaStd = nan(3,2);
rowsOut = table(strings(0,1),strings(0,1),nan(0,1),nan(0,1),nan(0,1),nan(0,1), ...
    'VariableNames',{'Mission','Configuration','RMSEVelMean_kmps','RMSEVelStd_kmps', ...
    'EffectiveSigmaVelMean_kmps','EffectiveSigmaVelStd_kmps'});

for m = 1:numel(missions)
    for c = 1:numel(configs)
        resultRow = R(R.Mission == missions(m) & ...
            string(R.Configuration) == configs(c),:);
        assert(height(resultRow) == 1, ...
            'Expected one objective/screening result for %s/%s.',missions(m),configs(c));
        runs = M(M.comparison_key == resultRow.ComparisonKey,:);
        assert(height(runs) == 20, ...
            'Expected 20 saved runs for %s/%s.',missions(m),configs(c));

        [rmseMean(m,c),rmseStd(m,c)] = sample_stats(runs.rmse_vel_kms);
        [sigmaMean(m,c),sigmaStd(m,c)] = ...
            sample_stats(runs.mean_effective_sigma_vel_kms);
        rowsOut(end+1,:) = {missions(m),configs(c),rmseMean(m,c),rmseStd(m,c), ...
            sigmaMean(m,c),sigmaStd(m,c)};
    end
end

writetable(rowsOut,fullfile(char(out),'ga_screening_velocity_metrics.csv'));

plot_metric(rmseMean,rmseStd,'Velocity RMSE (km/s)', ...
    "ga_screening_velocity_rmse",out,saveFigures,style);
plot_metric(sigmaMean,sigmaStd,'Effective velocity sigma (km/s)', ...
    "ga_screening_effective_sigma_velocity",out,saveFigures,style);

manifest = table( ...
    repmat("objective_screening",2,1), ...
    ["ga_screening_velocity_rmse";"ga_screening_effective_sigma_velocity"], ...
    ["Screening ON/OFF velocity RMSE across all target cases."; ...
     "Screening ON/OFF effective velocity uncertainty across all target cases."], ...
    'VariableNames',{'Study','FigureStem','Purpose'});
end


function [mu,sigma] = sample_stats(values)
values = double(values(:));
assert(numel(values) == 20 && all(isfinite(values)), ...
    'Velocity metric group must contain 20 finite values.');
mu = mean(values);
sigma = std(values,0);
end


function plot_metric(values,errors,yLabel,stem,out,saveFigures,style)
if ~saveFigures, return; end
fig = figure('Color','w','Units','inches', ...
    'Position',[1 1 style.metricFigureWidth style.metricFigureHeight], ...
    'PaperUnits','inches','PaperSize',[style.metricFigureWidth style.metricFigureHeight], ...
    'PaperPosition',[0 0 style.metricFigureWidth style.metricFigureHeight], ...
    'PaperPositionMode','manual','Renderer','painters','InvertHardcopy','off');
ax = axes(fig,'Units','normalized','Position',style.metricPlotPositionNoLegend);
hold(ax,'on'); box(ax,'off'); grid(ax,'off');

bars = bar(ax,1:3,values,'grouped','BarWidth',style.groupedBarWidth);
drawnow;
for c = 1:2
    bars(c).FaceColor = style.configurationColors(c,:);
    errorbar(ax,bars(c).XEndPoints,values(:,c),errors(:,c),'k.', ...
        'LineWidth',0.9,'CapSize',style.capSize,'HandleVisibility','off');
end

ax.XTick = 1:3;
ax.XTickLabel = {'LG','LT','GI'};
xlabel(ax,'Target case','FontWeight','bold');
ylabel(ax,yLabel,'FontWeight','bold');
set(ax,'FontName',style.fontName,'FontSize',style.fontSize, ...
    'FontWeight','bold','LineWidth',style.axisLineWidth,'TickDir','out', ...
    'Layer','top','Box','off','XGrid','off','YGrid','off','ZGrid','off');
ax.XLabel.FontSize = style.labelFontSize;
ax.YLabel.FontSize = style.labelFontSize;
space_manuscript_bars(ax,style);

base = fullfile(char(out),char(stem));
finalize_manuscript_figure(fig);
print(fig,[base '.eps'],'-depsc2','-painters','-r600','-loose');
exportgraphics(fig,[base '.png'],'Resolution',style.exportDpi);
close(fig);
end
