function stem = plot_reviewer2_runtime_summary(report,saveFigures)
%PLOT_REVIEWER2_RUNTIME_SUMMARY Equal-FE BO cost/benefit summary.
%
% The two panels make the reviewer-facing conclusion explicit: all five
% optimizers receive exactly 1200 admitted FE, then final objective and
% equal-budget runtime are compared across the 20 independent seeds.

if nargin < 2 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);
assert(isstruct(report) && isfield(report,'runtimeResults'), ...
    'A completed runtime-study report is required.');
style = reviewer2_paper_style();
R = report.runtimeResults;
order = style.optimizerOrder(ismember(style.optimizerOrder,R.Optimizer));
idx = nan(numel(order),1);
for k = 1:numel(order), idx(k) = find(R.Optimizer == order(k),1); end
R = R(idx,:);
colors = zeros(height(R),3);
for k = 1:height(R)
    colors(k,:) = style.optimizerColors(style.optimizerOrder == R.Optimizer(k),:);
end
labels = optimizer_labels(R.Optimizer);

fig = figure('Color','w','Units','inches','Position',[1 1 7.2 4.8], ...
    'PaperUnits','inches','PaperSize',[7.2 4.8], ...
    'PaperPosition',[0 0 7.2 4.8],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
movegui(fig,'center');
t = tiledlayout(fig,1,2,'Padding','loose','TileSpacing','compact');

ax1 = nexttile(t); hold(ax1,'on'); box(ax1,'on'); grid(ax1,'on');
b = bar(ax1,1:height(R),R.BestJMean,0.72,'FaceColor','flat'); b.CData = colors;
errorbar(ax1,1:height(R),R.BestJMean,R.BestJStd,'k.','LineWidth',0.9, ...
    'CapSize',style.capSize,'HandleVisibility','off');
format_axis(ax1,labels,'Final best objective',style);
title(ax1,'(a) Equal-FE solution quality','FontName',style.fontName, ...
    'FontSize',style.fontSize,'FontWeight','bold');

idxBO = find(R.Optimizer == "BAYESIAN",1);
nonBO = R(R.Optimizer ~= "BAYESIAN",:);
[bestNonBO,~] = min(nonBO.BestJMean);
gap = 100*(R.BestJMean(idxBO)-bestNonBO)/max(abs(bestNonBO),eps);
if gap >= 0
    note = sprintf('BO: +%.1f%% vs best mean',gap);
else
    note = sprintf('BO: %.1f%% vs best mean',gap);
end
text(ax1,0.04,0.96,note,'Units','normalized','VerticalAlignment','top', ...
    'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold');

ax2 = nexttile(t); hold(ax2,'on'); box(ax2,'on'); grid(ax2,'on');
b = bar(ax2,1:height(R),R.BudgetRuntimeMean_s,0.72,'FaceColor','flat'); b.CData = colors;
errorbar(ax2,1:height(R),R.BudgetRuntimeMean_s,R.BudgetRuntimeStd_s,'k.', ...
    'LineWidth',0.9,'CapSize',style.capSize,'HandleVisibility','off');
format_axis(ax2,labels,'Runtime to 1200 FE (s)',style);
title(ax2,'(b) Equal-FE computational cost','FontName',style.fontName, ...
    'FontSize',style.fontSize,'FontWeight','bold');
ratio = R.BudgetRuntimeMean_s(idxBO)/min(nonBO.BudgetRuntimeMean_s);
text(ax2,0.04,0.96,sprintf('BO: %.1fx slower than fastest',ratio), ...
    'Units','normalized','VerticalAlignment','top','FontName',style.fontName, ...
    'FontSize',style.fontSize,'FontWeight','bold');

stem = "runtime_1200_summary";
if saveFigures
    out = string(fullfile(char(report.analysisDirectory),'paper_final'));
    if ~isfolder(out), mkdir(out); end
    base = fullfile(char(out),char(stem)); drawnow;
    print(fig,[base '.eps'],'-depsc','-painters');
    exportgraphics(fig,[base '.png'],'Resolution',style.exportDpi);
    close(fig);
end
end


function format_axis(ax,labels,yLabel,style)
ax.XTick = 1:numel(labels); ax.XTickLabel = cellstr(labels);
ax.XTickLabelRotation = 18;
ylabel(ax,yLabel,'FontWeight','bold');
set(ax,'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold', ...
    'LineWidth',style.axisLineWidth,'TickDir','out','Layer','top');
ax.YLabel.FontSize = style.labelFontSize;
end

function labels = optimizer_labels(values)
values = string(values(:)); labels = strings(size(values));
for k = 1:numel(values)
    if values(k) == "BAYESIAN", labels(k) = "BO";
    else, labels(k) = upper(values(k)); end
end
end
