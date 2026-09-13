function files = plot_parallel_speed(sourceFile,outputDirectory)
%PLOT_PARALLEL_SPEED Export saved LG GA convergence; never run optimizations.
% plot_parallel_speed                       % latest completed LG benchmark
% plot_parallel_speed(sourceFile,folder)    % selected benchmark and destination
paths=setup_project();
if nargin<2 || isempty(outputDirectory)
    outputDirectory=fullfile(paths.root,'MANUSCRIPT_OUTPUT');
end
if ~isfolder(outputDirectory), mkdir(outputDirectory); end
if nargin<1 || isempty(sourceFile)
    candidates=dir(fullfile(paths.root,'MANUSCRIPT_OUTPUT', ...
        'parallel_speed_lunar_gateway_*','parallel_speed_convergence.mat'));
    [~,order]=sort([candidates.datenum],'descend'); sourceFile='';
    for idx=order
        file=fullfile(candidates(idx).folder,candidates(idx).name);
        saved=load(file,'benchmark');
        if saved.benchmark.complete && saved.benchmark.budget==6000
            sourceFile=file; break;
        end
    end
    assert(~isempty(sourceFile),'Run test_parallel_speed first: no complete LG benchmark found.');
end
S=load(sourceFile,'benchmark'); B=S.benchmark;
assert(B.complete && B.mission=="LUNAR_GATEWAY" && B.budget==6000, ...
    'A completed LG 6000-FE benchmark is required. Rerun test_parallel_speed.');
R=B.results; modes=["Serial","Parallel"];
assert(height(R)==2*B.nRepeats && all(R.SearchFE==6000),'Incomplete benchmark.');
style=reviewer2_paper_style(); files=strings(3,1);
for kind=1:2
    fig=figure('Visible','off','Color','w','Units','inches', ...
        'Position',[1 1 style.figureWidth style.figureHeight], ...
        'PaperUnits','inches','PaperSize',[style.figureWidth style.figureHeight], ...
        'PaperPosition',[0 0 style.figureWidth style.figureHeight], ...
        'PaperPositionMode','manual','Renderer','painters','InvertHardcopy','off');
    cleanup=onCleanup(@() close(fig));
    ax=axes(fig,'Units','normalized','Position',style.metricPlotPositionNoLegend); hold(ax,'on');
    set(ax,'FontName',style.fontName,'FontSize',style.fontSize, ...
        'LineWidth',style.axisLineWidth,'Box','off','FontWeight','bold', ...
        'TickDir','out','Layer','top','XGrid','off','YGrid','off','ZGrid','off');
    completionX=nan(2,1); completionY=nan(2,1);
    for m=1:2
        indices=find(R.Mode==modes(m));
        assert(numel(indices)==B.nRepeats && numel(unique(R.Repeat(indices)))==B.nRepeats, ...
            'Missing/duplicate timing repetitions.');
        H=B.histories(indices); x=cell(size(H)); y=x;
        for j=1:numel(H)
            h=H{j};
            assert(all(ismember({'fe','bestJ','elapsed_s'},h.Properties.VariableNames)), ...
                'No elapsed-time history. Rerun the updated benchmark.');
            assert(all(isfinite(h.bestJ)) && all(diff(h.fe)>0) && h.fe(end)==6000 ...
                && all(isfinite(h.elapsed_s)) && all(diff(h.elapsed_s)>0), ...
                'Invalid convergence history.');
            if kind==1, x{j}=h.fe; else, x{j}=h.elapsed_s; end
            y{j}=h.bestJ;
        end
        first=max(cellfun(@(v) v(1),x)); last=min(cellfun(@(v) v(end),x));
        assert(last>first,'Insufficient common history.');
        grid=unique(vertcat(x{:})); grid=grid(grid>=first & grid<=last);
        values=zeros(numel(grid),numel(x));
        for j=1:numel(x), values(:,j)=interp1(x{j},y{j},grid,'previous'); end
        lineStyle='-'; if m==2, lineStyle='--'; end
        stairs(ax,grid,mean(values,2),'LineWidth',style.lineWidth, ...
            'LineStyle',lineStyle,'Color',style.optimizerColors(m,:), ...
            'DisplayName',char(modes(m)));
        if kind==2
            completionX(m)=mean(R.OptimizationRuntime_s(indices));
            completionY(m)=mean(R.BestJ(indices));
        end
    end
    if kind==1
        xlabel(ax,'Function evaluations','FontWeight','bold','FontSize',style.labelFontSize); xlim(ax,[0 6000]);
        stem='parallel_speed_lg_convergence_fe';
    else
        markerSymbols={'o','s'};
        for m=1:2
            plot(ax,completionX(m),completionY(m),markerSymbols{m}, ...
                'MarkerSize',8,'MarkerFaceColor',style.optimizerColors(m,:), ...
                'MarkerEdgeColor',style.optimizerColors(m,:), ...
                'LineWidth',1.1,'HandleVisibility','off');
        end
        xlim(ax,[0 1.12*max(completionX)]);
        xl=xlim(ax); yl=ylim(ax);
        xSpan=diff(xl); ySpan=diff(yl);
        serialLabelX=completionX(1)-0.18*xSpan;
        serialLabelY=completionY(1)+0.12*ySpan;
        parallelLabelX=completionX(2)+0.18*xSpan;
        parallelLabelY=completionY(2)+0.28*ySpan;
        arrowStartX=[serialLabelX;parallelLabelX];
        arrowStartY=[serialLabelY-0.012*ySpan;parallelLabelY-0.012*ySpan];

        runtimeLabels = compose('Runtime: %.1f s',completionX);
        text(ax,serialLabelX,serialLabelY,runtimeLabels(1), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom', ...
            'FontName',style.fontName,'FontSize',style.legendFontSize, ...
            'FontWeight','bold','Color',style.optimizerColors(1,:));
        text(ax,parallelLabelX,parallelLabelY,runtimeLabels(2), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom', ...
            'FontName',style.fontName,'FontSize',style.legendFontSize, ...
            'FontWeight','bold','Color',style.optimizerColors(2,:));

        xlabel(ax,'Optimization elapsed time (s)','FontWeight','bold','FontSize',style.labelFontSize);
        stem='parallel_speed_lg_convergence_time';
    end
    ylabel(ax,'Best-so-far objective','FontWeight','bold','FontSize',style.labelFontSize);
    files(kind)=string(fullfile(outputDirectory,[stem '.eps']));
    drawnow;
    finalize_manuscript_figure(fig);
    drawnow;
    if kind==2
        for m=1:2
            [xStart,yStart]=data_to_figure_normalized(ax,arrowStartX(m),arrowStartY(m));
            [xEnd,yEnd]=data_to_figure_normalized(ax,completionX(m),completionY(m));
            dx=xEnd-xStart; dy=yEnd-yStart;
            arrowLength=hypot(dx,dy);
            gap=0.015;
            if arrowLength>gap
                xTip=xEnd-gap*dx/arrowLength;
                yTip=yEnd-gap*dy/arrowLength;
            else
                xTip=xEnd; yTip=yEnd;
            end
            annotation(fig,'arrow',[xStart xTip],[yStart yTip], ...
                'Color',style.optimizerColors(m,:), ...
                'LineWidth',1.3,'HeadLength',9,'HeadWidth',9);
        end
    end

    print(fig,char(files(kind)),'-depsc2','-painters','-r600','-loose');
    print(fig,char(replace(files(kind),'.eps','.png')),'-dpng', ...
        sprintf('-r%d',style.exportDpi));
    clear cleanup;
end
legendFiles=export_shared_result_legend(outputDirectory,"parallel_speed_lg_legend", ...
    ["Serial";"Parallel"],style.optimizerColors(1:2,:),style, ...
    'LineStyles',["-";"--"],'NumColumns',2);
files(3)=legendFiles(1);
writetable(R,fullfile(outputDirectory,'parallel_speed_results.csv'));
sourceDir=fileparts(sourceFile);
summaryFile=fullfile(sourceDir,'parallel_speed_summary.txt');
if isfile(summaryFile) && ~strcmp(string(sourceDir),string(outputDirectory))
    copyfile(summaryFile,fullfile(outputDirectory,'parallel_speed_summary.txt'));
end
end

function [xFigure,yFigure] = data_to_figure_normalized(ax,xData,yData)
oldUnits=ax.Units;
ax.Units='normalized';
position=ax.Position;
ax.Units=oldUnits;
xl=xlim(ax); yl=ylim(ax);
xFigure=position(1)+(xData-xl(1))/diff(xl)*position(3);
yFigure=position(2)+(yData-yl(1))/diff(yl)*position(4);
end
