function details = plot_reviewer2_baseline_monte_carlo(samples,summary,outDir,saveFigures)
%PLOT_REVIEWER2_BASELINE_MONTE_CARLO Export separate local-MC boxplots.
%
% Each baseline configuration is written as its own EPS/PNG so the paper can
% assemble panels with subfigure/subcaption. The boxplot contains all local
% design samples, including sample 1 (the optimized GA reference), and the
% optimized objective is shown as a red horizontal line. All Monte Carlo
% figures use the shared manuscript export size and contain no figure titles.
% Figure files are stored together in the run's figures/ subdirectory while
% CSV/data products remain at the Monte Carlo analysis root. Final axes use
% clear labels, no grid lines, and no surrounding axes box.

if nargin < 4 || isempty(saveFigures), saveFigures = true; end
saveFigures = logical(saveFigures);
assert(istable(samples) && istable(summary));
requiredSamples = ["Mission","Measurement","NumObservers","NPeriods", ...
    "Sample","IsReference","TotalCost"];
requiredSummary = ["Mission","Measurement","NumObservers","NPeriods", ...
    "ReferenceObjective"];
assert(all(ismember(requiredSamples,string(samples.Properties.VariableNames))));
assert(all(ismember(requiredSummary,string(summary.Properties.VariableNames))));

style = reviewer2_paper_style();
outDir = string(outDir);
figureDir = string(fullfile(char(outDir),'figures'));
if saveFigures && ~isfolder(figureDir), mkdir(figureDir); end

figureStem = strings(height(summary),1);
for k = 1:height(summary)
    s = summary(k,:);
    rows = samples(samples.Mission == s.Mission & ...
        samples.Measurement == s.Measurement & ...
        samples.NumObservers == s.NumObservers & ...
        samples.NPeriods == s.NPeriods,:);
    assert(~isempty(rows),'Missing Monte Carlo sample rows for plotted case.');

    widthIn = style.monteCarloFigureWidth;
    heightIn = style.monteCarloFigureHeight;
    fig = figure('Color','w','Units','inches','Position',[1 1 widthIn heightIn], ...
        'PaperUnits','inches','PaperSize',[widthIn heightIn], ...
        'PaperPosition',[0 0 widthIn heightIn],'PaperPositionMode','manual', ...
        'Renderer','painters','InvertHardcopy','off');
    movegui(fig,'center');
    ax = axes(fig,'Units','normalized','Position',[0.18 0.18 0.76 0.74]);
    hold(ax,'on'); box(ax,'off'); grid(ax,'off');

    hBox = boxchart(ax,ones(height(rows),1),rows.TotalCost, ...
        'BoxFaceColor',style.optimizerColors(1,:), ...
        'MarkerStyle','.');
    hRef = yline(ax,s.ReferenceObjective,'-','Color',[1.00 0.30 0.30], ...
        'LineWidth',1.5);

    % Keep the optimized-reference line visibly separated from the x axis.
    % MATLAB's automatic limits can place the minimum reference exactly on the
    % lower axes boundary when it is the smallest plotted value. Reserve a
    % small data-relative margin below and above every MC distribution so the
    % reference line remains distinct in both EPS and PNG exports.
    plotValues = [double(rows.TotalCost(:));double(s.ReferenceObjective)];
    plotValues = plotValues(isfinite(plotValues));
    assert(~isempty(plotValues),'Monte Carlo plot contains no finite objective values.');
    plotMin = min(plotValues);
    plotMax = max(plotValues);
    plotSpan = max(plotMax-plotMin,0.05*max(1,max(abs(plotValues))));
    yPadding = 0.08*plotSpan;
    ylim(ax,[plotMin-yPadding,plotMax+yPadding]);

    xlim(ax,[0.55 1.45]);
    xticks(ax,[]);
    xlabel(ax,'Monte Carlo samples','FontWeight','bold');
    ylabel(ax,'Objective value','FontWeight','bold');
    set(ax,'FontName',style.fontName,'FontSize',style.fontSize, ...
        'FontWeight','bold','LineWidth',style.axisLineWidth, ...
        'TickDir','out','Layer','top','Box','off', ...
        'XGrid','off','YGrid','off','ZGrid','off');
    ax.XLabel.FontSize = style.labelFontSize;
    ax.YLabel.FontSize = style.labelFontSize;

    lgd = legend(ax,[hBox hRef],{'Monte Carlo samples','Optimized reference'}, ...
        'Location','northoutside','Orientation','horizontal','Box','off');
    lgd.FontName = style.fontName;
    lgd.FontSize = style.fontSize;
    lgd.FontWeight = 'bold';
    format_manuscript_legend(ax,lgd,style,style.metricPlotPosition);

    stem = "baseline_mc_"+mission_code(s.Mission)+"_"+ ...
        measurement_code(s.Measurement)+"_o"+string(s.NumObservers);
    if s.Mission == "LUNAR_GATEWAY"
        stem = stem+"_p"+string(s.NPeriods);
    end
    figureStem(k) = stem;

    drawnow;
    if saveFigures
        base = fullfile(char(figureDir),char(stem));
        print(fig,[base '.eps'],'-depsc2','-painters','-r600','-loose');
        exportgraphics(fig,[base '.png'],'Resolution',style.exportDpi);
        close(fig);
    end
end

details = summary(:,intersect(summary.Properties.VariableNames, ...
    {'Mission','Measurement','NumObservers','NPeriods','ReferenceSeed', ...
    'ReferenceObjective','MedianObjective','FractionNeighborsAtOrAboveReference', ...
    'ImprovedNeighborCount','StrictLocalMinimumPass'},'stable'));
details.FigureStem = figureStem;
details.FigureDirectory = repmat(figureDir,height(details),1);
end


function enforce_minimum_font_size(fig,minFontSize)
objects = findall(fig,'-property','FontSize');
for k = 1:numel(objects)
    try
        if objects(k).FontSize < minFontSize, objects(k).FontSize = minFontSize; end
    catch
    end
end
end


function code = mission_code(mission)
switch string(mission)
    case "LUNAR_GATEWAY", code = "lg";
    case "LOW_THRUST_TRANSFER", code = "lt";
    case "GATEWAY_IMPULSE", code = "gi";
    otherwise, code = lower(string(mission));
end
end


function code = measurement_code(measurement)
if string(measurement) == "ANGLES_ONLY", code = "ao"; else, code = "ar"; end
end
