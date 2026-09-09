function stem = plot_reviewer2_optimizer_ranking(report,saveFigures)
%PLOT_REVIEWER2_OPTIMIZER_RANKING Summarize full-comparison optimizer ranking.
%
% The left panel shows the mean mission-wise objective rank (lower is better)
% and the right panel shows the number of target cases won. The figure is a
% compact visual companion to comparison_6000_overall_ranking.csv and
% comparison_6000_best_by_mission.csv.

if nargin < 2 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);
assert(isstruct(report) && isfield(report,'overallRanking'), ...
    'A completed comparison report is required.');
style = reviewer2_paper_style();
R = report.overallRanking;
R = sortrows(R,'OverallRank','ascend');

fig = figure('Color','w','Units','inches','Position',[1 1 7.2 4.8], ...
    'PaperUnits','inches','PaperSize',[7.2 4.8], ...
    'PaperPosition',[0 0 7.2 4.8],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
movegui(fig,'center');
t = tiledlayout(fig,1,2,'Padding','loose','TileSpacing','compact');
colors = zeros(height(R),3);
for k = 1:height(R)
    idx = find(style.optimizerOrder == upper(string(R.Optimizer(k))),1);
    colors(k,:) = style.optimizerColors(idx,:);
end
labels = optimizer_labels(R.Optimizer);

ax1 = nexttile(t); hold(ax1,'on'); box(ax1,'on'); grid(ax1,'on');
b = bar(ax1,1:height(R),R.MeanObjectiveRank,0.72,'FaceColor','flat');
b.CData = colors;
ax1.XTick = 1:height(R); ax1.XTickLabel = cellstr(labels);
ylabel(ax1,'Mean objective rank','FontWeight','bold');
title(ax1,'(a) Overall solution-quality rank','FontName',style.fontName, ...
    'FontSize',style.fontSize,'FontWeight','bold');
style_axis(ax1,style);

ax2 = nexttile(t); hold(ax2,'on'); box(ax2,'on'); grid(ax2,'on');
b = bar(ax2,1:height(R),R.MissionWins,0.72,'FaceColor','flat');
b.CData = colors;
ax2.XTick = 1:height(R); ax2.XTickLabel = cellstr(labels);
ylabel(ax2,'Target cases won','FontWeight','bold');
yticks(ax2,0:max(3,max(R.MissionWins)));
title(ax2,'(b) Case-wise objective wins','FontName',style.fontName, ...
    'FontSize',style.fontSize,'FontWeight','bold');
style_axis(ax2,style);

stem = "comparison_6000_optimizer_ranking";
if saveFigures
    out = string(fullfile(char(report.analysisDirectory),'paper_final'));
    if ~isfolder(out), mkdir(out); end
    base = fullfile(char(out),char(stem));
    drawnow;
    print(fig,[base '.eps'],'-depsc','-painters');
    exportgraphics(fig,[base '.png'],'Resolution',style.exportDpi);
    close(fig);
end
end


function style_axis(ax,style)
set(ax,'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold', ...
    'LineWidth',style.axisLineWidth,'TickDir','out','Layer','top');
ax.YLabel.FontSize = style.labelFontSize;
end

function labels = optimizer_labels(values)
values = string(values(:)); labels = strings(size(values));
for k = 1:numel(values)
    if upper(values(k)) == "BAYESIAN", labels(k) = "BO";
    else, labels(k) = upper(values(k)); end
end
end
