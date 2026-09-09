function stems = plot_reviewer2_screening_summary(report,saveFigures)
%PLOT_REVIEWER2_SCREENING_SUMMARY Matched J111 screening ON/OFF summary.
%
% Mirrors the useful metric-panel structure of the original manuscript while
% using the new GA-only equal-FE sensitivity study. Total objective is valid
% here because screening ON and OFF use the identical J111 objective.
%
% screening_count is labeled as a visibility-rule violation count because
% cr3bp_ekf evaluates visibility in both cases; violations are only rejected
% from the EKF measurement update when screening is enabled.

if nargin < 2 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);
assert(isstruct(report) && isfield(report,'results') && isfield(report,'runMetrics'), ...
    'A completed objective/screening report is required.');
style = reviewer2_paper_style();
missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
configs = ["combined_on","combined_off"];
stems = strings(numel(missions),1);

for m = 1:numel(missions)
    mission = missions(m);
    values = nan(2,4); errors = nan(2,4);
    for k = 1:2
        row = result_row(report.results,mission,configs(k));
        values(k,1) = row.BestJMean; errors(k,1) = row.BestJStd;
        values(k,2) = row.RMSEPosMean_km; errors(k,2) = row.RMSEPosStd_km;
        values(k,4) = row.ScreeningMean; errors(k,4) = row.ScreeningStd;

        runs = report.runMetrics(report.runMetrics.comparison_key == ...
            string(row.ComparisonKey),:);
        assert(height(runs) == 20,'Expected 20 runs for screening summary.');
        values(k,3) = mean(double(runs.budget_runtime_s));
        errors(k,3) = std(double(runs.budget_runtime_s));
    end

    fig = figure('Color','w','Units','inches','Position',[1 1 7.2 6.2], ...
        'PaperUnits','inches','PaperSize',[7.2 6.2], ...
        'PaperPosition',[0 0 7.2 6.2],'PaperPositionMode','manual', ...
        'Renderer','painters','InvertHardcopy','off');
    movegui(fig,'center');
    t = tiledlayout(fig,2,2,'Padding','loose','TileSpacing','compact');
    labels = {'Final best objective','Position RMSE (km)', ...
        'Runtime to 6000 FE (s)','Visibility-rule violations'};
    panelLabels = {'(a)','(b)','(c)','(d)'};
    colors = style.configurationColors(1:2,:);
    handles = gobjects(2,1);

    for q = 1:4
        ax = nexttile(t); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
        b = bar(ax,1:2,values(:,q),0.68,'FaceColor','flat');
        b.CData = colors;
        errorbar(ax,1:2,values(:,q),errors(:,q),'k.','LineWidth',0.9, ...
            'CapSize',style.capSize,'HandleVisibility','off');
        ax.XTick = 1:2;
        ax.XTickLabel = {'Screening ON','Screening OFF'};
        ylabel(ax,labels{q},'FontWeight','bold');
        title(ax,panelLabels{q},'FontName',style.fontName, ...
            'FontSize',style.fontSize,'FontWeight','bold');
        style_axis(ax,style);
        if q == 1
            handles(1) = plot(ax,nan,nan,'s','MarkerFaceColor',colors(1,:), ...
                'MarkerEdgeColor','none','MarkerSize',7,'DisplayName','Screening ON');
            handles(2) = plot(ax,nan,nan,'s','MarkerFaceColor',colors(2,:), ...
                'MarkerEdgeColor','none','MarkerSize',7,'DisplayName','Screening OFF');
        end
    end

    lgd = legend(handles,{'Screening ON','Screening OFF'}, ...
        'Orientation','horizontal','Box','off');
    lgd.FontName = style.fontName; lgd.FontSize = style.fontSize;
    lgd.FontWeight = 'bold'; lgd.Layout.Tile = 'north';

    stem = "ga_screening_summary_"+mission_code(mission);
    stems(m) = stem;
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
end


function row = result_row(results,mission,configuration)
row = results(results.Mission == mission & ...
    string(results.Configuration) == configuration,:);
assert(height(row) == 1,'Missing screening result for %s/%s.',mission,configuration);
end

function style_axis(ax,style)
set(ax,'FontName',style.fontName,'FontSize',style.fontSize,'FontWeight','bold', ...
    'LineWidth',style.axisLineWidth,'TickDir','out','Layer','top');
ax.YLabel.FontSize = style.labelFontSize;
end

function code = mission_code(mission)
switch upper(string(mission))
    case "LUNAR_GATEWAY", code = "lg";
    case "LOW_THRUST_TRANSFER", code = "lt";
    case "GATEWAY_IMPULSE", code = "gi";
    otherwise, code = lower(string(mission));
end
end
