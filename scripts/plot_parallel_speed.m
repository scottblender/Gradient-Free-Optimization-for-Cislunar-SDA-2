function files = plot_parallel_speed(sourceFile,outputDirectory)
%PLOT_PARALLEL_SPEED Export saved LG GA convergence; never run optimizations.
S=load(sourceFile,'benchmark'); B=S.benchmark;
assert(B.complete && B.mission=="LUNAR_GATEWAY" && B.budget==6000, ...
    'A completed LG 6000-FE benchmark is required. Rerun test_parallel_speed.');
R=B.results; modes=["Serial","Parallel"];
assert(height(R)==2*B.nRepeats && all(R.SearchFE==6000),'Incomplete benchmark.');
style=reviewer2_paper_style(); files=strings(2,1);
for kind=1:2
    fig=figure('Visible','off','Color','w','Units','inches', ...
        'Position',[1 1 style.metricFigureWidth style.metricFigureHeight], ...
        'PaperUnits','inches','PaperSize',[style.metricFigureWidth style.metricFigureHeight]);
    cleanup=onCleanup(@() close(fig)); %#ok<NASGU>
    ax=axes(fig); hold(ax,'on');
    set(ax,'FontName',style.fontName,'FontSize',style.fontSize, ...
        'LineWidth',style.axisLineWidth,'Box','off');
    setappdata(ax,'ManuscriptAxesPosition',style.metricPlotPosition);
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
        % Compare repeats only where all have recorded observations. Do not
        % fabricate time-zero objectives or extrapolate completed runs.
        first=max(cellfun(@(v) v(1),x)); last=min(cellfun(@(v) v(end),x));
        assert(last>first,'Insufficient common history.');
        grid=unique(vertcat(x{:})); grid=grid(grid>=first & grid<=last);
        values=zeros(numel(grid),numel(x));
        for j=1:numel(x), values(:,j)=interp1(x{j},y{j},grid,'previous'); end
        lineStyle='-'; if m==2, lineStyle='--'; end
        stairs(ax,grid,mean(values,2),'LineWidth',style.lineWidth, ...
            'LineStyle',lineStyle,'Color',style.optimizerColors(m,:), ...
            'DisplayName',char(modes(m)));
    end
    if kind==1
        xlabel(ax,'Function evaluations'); xlim(ax,[0 6000]);
        stem='parallel_speed_lg_convergence_fe';
    else
        xlabel(ax,'Optimization elapsed time (s)');
        stem='parallel_speed_lg_convergence_time';
    end
    ylabel(ax,'Mean best-so-far objective');
    legend(ax,'Location','northoutside','Orientation','horizontal','Box','off');
    files(kind)=string(fullfile(outputDirectory,[stem '.eps']));
    prepare_manuscript_figure(fig);
    export_manuscript_figure(fig,files(kind));
    clear cleanup;
end
% Keep the numeric printout beside the final figures as well as in the raw run.
writetable(R,fullfile(outputDirectory,'parallel_speed_results.csv'));
sourceDir=fileparts(sourceFile);
summaryFile=fullfile(sourceDir,'parallel_speed_summary.txt');
if isfile(summaryFile), copyfile(summaryFile,fullfile(outputDirectory,'parallel_speed_summary.txt')); end
end
