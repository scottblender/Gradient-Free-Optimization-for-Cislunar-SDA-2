function stem = plot_reviewer2_baseline_duration_matrix(report,saveFigures)
%PLOT_REVIEWER2_BASELINE_DURATION_MATRIX Gateway duration/observer interaction.
%
% Four panels show how objective and RMSE change over one, three, and five
% Gateway periods for all four constellation sizes. AO and AR are separated
% so the observer-count and tracking-duration effects remain readable.

if nargin < 2 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);
assert(isstruct(report) && isfield(report,'results'), ...
    'A completed baseline report is required.');
style = reviewer2_paper_style();
R = report.results;
counts = [3 5 7 10];
periods = [1 3 5];
measurements = ["ANGLES_ONLY","ANGLES_RANGE"];
colors = lines(numel(counts));

fig = figure('Color','w','Units','inches','Position',[1 1 7.2 6.2], ...
    'PaperUnits','inches','PaperSize',[7.2 6.2], ...
    'PaperPosition',[0 0 7.2 6.2],'PaperPositionMode','manual', ...
    'Renderer','painters','InvertHardcopy','off');
movegui(fig,'center');
t = tiledlayout(fig,2,2,'Padding','loose','TileSpacing','compact');
legendHandles = gobjects(numel(counts),1);

for m = 1:2
    for metric = 1:2
        ax = nexttile(t,(metric-1)*2+m);
        hold(ax,'on'); box(ax,'on'); grid(ax,'on');
        for c = 1:numel(counts)
            values = nan(size(periods)); errors = values;
            for p = 1:numel(periods)
                row = R(R.Mission == "LUNAR_GATEWAY" & ...
                    R.Measurement == measurements(m) & ...
                    R.NumObservers == counts(c) & R.NPeriods == periods(p),:);
                assert(height(row) == 1,'Missing Gateway duration/observer result.');
                if metric == 1
                    values(p) = row.BestJMean; errors(p) = row.BestJStd;
                else
                    values(p) = row.RMSEPosMean_km; errors(p) = row.RMSEPosStd_km;
                end
            end
            h = errorbar(ax,periods,values,errors,'-o','Color',colors(c,:), ...
                'LineWidth',style.lineWidth,'MarkerSize',style.markerSize, ...
                'MarkerFaceColor',colors(c,:),'CapSize',style.capSize, ...
                'DisplayName',sprintf('%d observers',counts(c)));
            if m == 1 && metric == 1, legendHandles(c) = h; end
        end
        ax.XTick = periods;
        xlabel(ax,'Gateway tracking periods','FontWeight','bold');
        if metric == 1, ylabel(ax,'Final best objective','FontWeight','bold');
        else, ylabel(ax,'Position RMSE (km)','FontWeight','bold'); end
        if m == 1, measLabel = 'AO'; else, measLabel = 'AR'; end
        if metric == 1, panelLetter = char('a'+m-1); else, panelLetter = char('c'+m-1); end
        title(ax,sprintf('(%c) %s',panelLetter,measLabel), ...
            'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold');
        style_axis(ax,style);
    end
end

lgd = legend(legendHandles,'Orientation','horizontal','NumColumns',2,'Box','off');
lgd.FontName = style.fontName; lgd.FontSize = style.fontSize;
lgd.FontWeight = 'bold'; lgd.Layout.Tile = 'north';

stem = "baseline_gateway_duration_by_observers";
if saveFigures
    out = string(fullfile(char(report.analysisDirectory),'paper_final'));
    if ~isfolder(out), mkdir(out); end
    base = fullfile(char(out),char(stem)); drawnow;
    print(fig,[base '.eps'],'-depsc','-painters');
    exportgraphics(fig,[base '.png'],'Resolution',style.exportDpi);
    close(fig);
end
end

function style_axis(ax,style)
set(ax,'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold', ...
    'LineWidth',style.axisLineWidth,'TickDir','out','Layer','top');
ax.XLabel.FontSize = style.labelFontSize;
ax.YLabel.FontSize = style.labelFontSize;
end
