function details = plot_reviewer2_baseline_monte_carlo(samples,summary,outDir,saveFigures)
%PLOT_REVIEWER2_BASELINE_MONTE_CARLO Export separate local-MC boxplots.
%
% Each baseline configuration is written as its own EPS/PNG so the paper can
% assemble 3/5/7/10-observer panels with subfigure/subcaption. The boxplot
% contains all local design samples, including sample 1 (the optimized GA
% reference), and the optimized objective is shown as a red horizontal line.

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
if saveFigures && ~isfolder(outDir), mkdir(outDir); end

figureStem = strings(height(summary),1);
for k = 1:height(summary)
    s = summary(k,:);
    rows = samples(samples.Mission == s.Mission & ...
        samples.Measurement == s.Measurement & ...
        samples.NumObservers == s.NumObservers & ...
        samples.NPeriods == s.NPeriods,:);
    assert(~isempty(rows),'Missing Monte Carlo sample rows for plotted case.');

    fig = figure('Color','w','Units','inches','Position',[1 1 4.8 4.2], ...
        'PaperUnits','inches','PaperSize',[4.8 4.2], ...
        'PaperPosition',[0 0 4.8 4.2],'PaperPositionMode','manual', ...
        'Renderer','painters','InvertHardcopy','off');
    movegui(fig,'center');
    ax = axes(fig,'Units','normalized','Position',[0.18 0.18 0.76 0.74]);
    hold(ax,'on'); box(ax,'on'); grid(ax,'on');

    hBox = boxchart(ax,ones(height(rows),1),rows.TotalCost, ...
        'BoxFaceColor',style.optimizerColors(1,:), ...
        'MarkerStyle','.');
    hRef = yline(ax,s.ReferenceObjective,'-','Color',[1.00 0.30 0.30], ...
        'LineWidth',1.5);

    xlim(ax,[0.55 1.45]);
    xticks(ax,[]);
    xlabel(ax,'MC samples','FontWeight','bold');
    ylabel(ax,'Total cost','FontWeight','bold');
    set(ax,'FontName',style.fontName,'FontSize',style.fontSize, ...
        'FontWeight','bold','LineWidth',style.axisLineWidth, ...
        'TickDir','out','Layer','top');
    ax.XLabel.FontSize = style.labelFontSize;
    ax.YLabel.FontSize = style.labelFontSize;

    lgd = legend(ax,[hBox hRef],{'MC samples','Optimized reference'}, ...
        'Location','northoutside','Orientation','horizontal','Box','off');
    lgd.FontName = style.fontName;
    lgd.FontSize = style.fontSize;
    lgd.FontWeight = 'bold';

    stem = "baseline_mc_"+mission_code(s.Mission)+"_"+ ...
        measurement_code(s.Measurement)+"_o"+string(s.NumObservers);
    if s.Mission == "LUNAR_GATEWAY"
        stem = stem+"_p"+string(s.NPeriods);
    end
    figureStem(k) = stem;

    enforce_minimum_font_size(fig,12);
    drawnow;
    if saveFigures
        base = fullfile(char(outDir),char(stem));
        print(fig,[base '.eps'],'-depsc2','-painters','-r600');
        exportgraphics(fig,[base '.png'],'Resolution',style.exportDpi);
        close(fig);
    end
end

details = summary(:,intersect(summary.Properties.VariableNames, ...
    {'Mission','Measurement','NumObservers','NPeriods','ReferenceSeed', ...
    'ReferenceObjective','MedianObjective','FractionNeighborsAtOrAboveReference', ...
    'ImprovedNeighborCount','StrictLocalMinimumPass'},'stable'));
details.FigureStem = figureStem;
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
