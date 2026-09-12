function manifest = make_reviewer2_runtime_figures(r,baselineResults,saveFigures)
%MAKE_REVIEWER2_RUNTIME_FIGURES Render the focused 1200-FE paper figures.
%
% The optimizer identity is already explicit on the x-axis of the two bar
% charts, so those charts intentionally do not create one legend entry per
% optimizer. MATLAB's bar() returns one Bar object for this flat-colored
% categorical chart; pairing that single handle with five optimizer labels
% produces the "Ignoring extra legend entries" warning. The objective chart
% therefore uses a legend only for the dashed 6000-FE GA reference.
% The convergence chart uses one graphics handle per optimizer curve and
% intentionally shows only the 20-run mean best-so-far history. Run-to-run
% variability is reported in the metric summaries/tables rather than as
% terminal error bars on the convergence figure.
% All runtime metric and convergence figures use the shared manuscript
% metric export size, contain no MATLAB titles, and use no grid/axes box.

if nargin < 2 || isempty(baselineResults), baselineResults = table(); end
if nargin < 3 || isempty(saveFigures), saveFigures = true; end
saveFigures = logical(saveFigures);
style = reviewer2_paper_style();
out = string(r.analysisDirectory);
assert(isfolder(out),'Runtime analysis directory does not exist: %s',out);

if saveFigures
    clear_runtime_figures(out);
end

baseline = matched_baseline(baselineResults,"LUNAR_GATEWAY",3,1);
plot_runtime_metric(r,'BestJMean','BestJStd','Mean final best objective', ...
    "runtime_1200_objective",out,saveFigures,style,false,baseline);
plot_runtime_metric(r,'BudgetRuntimeMean_s','BudgetRuntimeStd_s', ...
    'Mean runtime to 1200 FE (s)',"runtime_1200_runtime",out,saveFigures,style,true,table());
plot_runtime_convergence(r,out,saveFigures,style);

manifest = table( ...
    repmat("runtime",3,1), ...
    ["runtime_1200_objective";"runtime_1200_runtime";"runtime_1200_convergence"], ...
    ["Equal-1200-FE mean final-best objective with matched 6000-FE GA reference."; ...
     "Equal-1200-FE mean computational cost showing BO scaling penalty."; ...
     "Five-method equal-FE mean convergence comparison."], ...
    'VariableNames',{'Study','FigureStem','Purpose'});
writetable(manifest,fullfile(char(out),'paper_figure_manifest.csv'));
end


function plot_runtime_metric(r,valueField,stdField,yLabel,stem,out,saveFigures,style,annotateBO,baseline)
R = r.runtimeResults;
order = style.optimizerOrder(ismember(style.optimizerOrder,R.Optimizer));
R = sort_to_order(R,'Optimizer',order);
colors = colors_for_optimizers(R.Optimizer,style);

fig = paper_figure(style.metricFigureWidth,style.metricFigureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'off'); grid(ax,'off');
values = R.(valueField); errors = R.(stdField);
b = bar(ax,1:height(R),values,style.groupedBarWidth,'FaceColor','flat');
b.CData = colors;
errorbar(ax,1:height(R),values,errors,'k.','LineWidth',1.0, ...
    'CapSize',style.capSize,'HandleVisibility','off');

ax.XTick = 1:height(R);
ax.XTickLabel = cellstr(optimizer_labels(R.Optimizer));
ax.XTickLabelRotation = 18;
xlabel(ax,'Optimizer','FontWeight','bold');
ylabel(ax,yLabel,'FontWeight','bold');
style_axes(ax,style);

% Optimizers are identified by the x-axis labels. The dashed reference is
% the matched 6000-FE GA result; its run-to-run spread remains in the table
% rather than being drawn as an error bar on a horizontal reference line.
if ~isempty(baseline)
    hBase = plot(ax,[0.55 height(R)+0.45],[baseline.Mean baseline.Mean],'--', ...
        'Color',[0.30 0.30 0.30],'LineWidth',1.5, ...
        'DisplayName','6000-FE GA reference');
    lgd = legend(ax,hBase,{'6000-FE GA reference'},'Location','northoutside', ...
        'Orientation','horizontal','Box','off');
    style_legend(lgd,ax,style);
end

% The runtime bars and their standard-deviation error bars communicate the
% BO cost directly. Do not add ratio callouts such as "x fastest" above BO.
if annotateBO
    % Retained as an input for compatibility with the curated call pattern.
end

export_figure(fig,out,stem,saveFigures,style);
end


function plot_runtime_convergence(r,out,saveFigures,style)
files = dir(fullfile(char(r.analysisDirectory),'convergence_*.mat'));
assert(numel(files) == 1,'Expected one runtime convergence file.');
S = load(fullfile(files(1).folder,files(1).name),'curves');
optimizers = style.optimizerOrder(ismember(style.optimizerOrder, ...
    upper(string({S.curves.optimizer}))));
curves = cell(numel(optimizers),1);
for k = 1:numel(optimizers)
    idx = find(upper(string({S.curves.optimizer})) == optimizers(k),1);
    assert(~isempty(idx),'Missing runtime convergence curve for %s.',optimizers(k));
    curves{k} = S.curves(idx);
end
plot_curve_overlay(curves,optimizer_labels(optimizers), ...
    colors_for_optimizers(optimizers,style),r.budget,out, ...
    "runtime_1200_convergence",saveFigures,style);
end


function plot_curve_overlay(curves,labels,colors,budget,out,stem,saveFigures,style)
fig = paper_figure(style.metricFigureWidth,style.metricFigureHeight,style);
ax = axes(fig); hold(ax,'on'); box(ax,'off'); grid(ax,'off');
handles = gobjects(numel(curves),1);
allY = zeros(0,1);
for k = 1:numel(curves)
    c = curves{k};
    valid = c.fe >= 60 & isfinite(c.mean);
    assert(any(valid),'Runtime convergence curve contains no valid FE >= 60.');
    x = double(c.fe(valid));
    y = double(c.mean(valid));
    handles(k) = stairs(ax,x,y,'Color',colors(k,:), ...
        'LineWidth',style.lineWidth,'DisplayName',string(labels(k)));
    allY = [allY;y];
end
allY = allY(isfinite(allY));
lo = min(allY); hi = max(allY);
span = max(hi-lo,0.05*max(1,abs(hi)));
ylim(ax,[lo-0.06*span hi+0.08*span]);
xlim(ax,[60 budget]);
xlabel(ax,'Function evaluations','FontWeight','bold');
ylabel(ax,'Mean best-so-far objective','FontWeight','bold');
style_axes(ax,style);
lgd = legend(ax,handles,'Location','northoutside','Orientation','horizontal', ...
    'NumColumns',min(numel(handles),5),'Box','off');
style_legend(lgd,ax,style);
export_figure(fig,out,stem,saveFigures,style);
end


function ref = matched_baseline(B,mission,numObservers,nPeriods)
ref = table();
if isempty(B), return; end
rows = B(B.Mission == mission & B.Measurement == "ANGLES_ONLY" & ...
    B.NumObservers == numObservers & B.NPeriods == nPeriods,:);
if height(rows) ~= 1, return; end
ref = table(string(mission),rows.BestJMean,rows.BestJStd, ...
    'VariableNames',{'Mission','Mean','Std'});
end


function R = sort_to_order(R,field,order)
idx = nan(numel(order),1);
for k = 1:numel(order)
    idx(k) = find(string(R.(field)) == order(k),1);
end
R = R(idx,:);
end


function colors = colors_for_optimizers(optimizers,style)
colors = zeros(numel(optimizers),3);
for k = 1:numel(optimizers)
    idx = find(style.optimizerOrder == upper(string(optimizers(k))),1);
    assert(~isempty(idx),'Unknown optimizer color: %s',optimizers(k));
    colors(k,:) = style.optimizerColors(idx,:);
end
end


function labels = optimizer_labels(values)
values = upper(string(values(:)));
labels = values;
labels(values == "BAYESIAN") = "BO";
end


function fig = paper_figure(widthIn,heightIn,style)
fig = figure('Color','w','Units','inches','Position',[1 1 widthIn heightIn], ...
    'PaperUnits','inches','PaperSize',[widthIn heightIn], ...
    'PaperPosition',[0 0 widthIn heightIn],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
movegui(fig,'center');
set(fig,'DefaultAxesFontName',style.fontName,'DefaultAxesFontSize',style.fontSize);
end


function style_axes(ax,style)
set(ax,'Units','normalized','Position',style.metricPlotPosition);
set(ax,'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold', ...
    'LineWidth',style.axisLineWidth,'TickDir','out','Layer','top', ...
    'Box','off','XGrid','off','YGrid','off','ZGrid','off');
ax.XLabel.FontSize = style.labelFontSize;
ax.YLabel.FontSize = style.labelFontSize;
wrap_manuscript_label(ax.XLabel); wrap_manuscript_label(ax.YLabel);
space_manuscript_bars(ax,style);
end


function style_legend(lgd,ax,style)
lgd.FontName = style.fontName;
lgd.FontSize = style.fontSize;
lgd.FontWeight = 'bold';
format_manuscript_legend(ax,lgd,style,style.metricPlotPosition);
end


function export_figure(fig,out,stem,saveFigures,style)
drawnow;
if ~saveFigures, return; end
base = fullfile(char(out),char(stem));
finalize_manuscript_figure(fig);
print(fig,[base '.eps'],'-depsc2','-painters','-r600','-loose');
exportgraphics(fig,[base '.png'],'Resolution',style.exportDpi);
close(fig);
end


function enforce_minimum_font_size(fig,minFontSize)
objects = findall(fig,'-property','FontSize');
for k = 1:numel(objects)
    try
        if objects(k).FontSize < minFontSize
            objects(k).FontSize = minFontSize;
        end
    catch
    end
end
end


function clear_runtime_figures(out)
for pattern = ["runtime_1200_*.eps","runtime_1200_*.png"]
    files = dir(fullfile(char(out),char(pattern)));
    for k = 1:numel(files)
        delete(fullfile(files(k).folder,files(k).name));
    end
end
end
