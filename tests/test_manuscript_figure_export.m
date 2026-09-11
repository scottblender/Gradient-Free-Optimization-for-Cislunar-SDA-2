function test_manuscript_figure_export()
% Data-free regression for paired export canvases and manuscript typography.
paths = setup_project(); %#ok<NASGU>
style = reviewer2_paper_style();
folder = tempname; mkdir(folder);
cleanup = onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
boxes = strings(2,1); imageSizes = zeros(2,2);
for k = 1:2
    fig = figure('Visible','off','Units','inches', ...
        'Position',[1 1 style.metricFigureWidth style.metricFigureHeight], ...
        'PaperUnits','inches','PaperSize',[style.metricFigureWidth style.metricFigureHeight]);
    closeFigure = onCleanup(@() close(fig));
    ax = axes(fig); plot(ax,1:10,k*(1:10)); xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)');
    if k==2
        hold(ax,'on');
        for j=2:5, plot(ax,1:10,j*(1:10)); end
        ylabel(ax,'Mean effective position uncertainty (km)');
        legend(ax,{'GA','PSO','BO','ABCO','ACO'},'Location','northoutside');
        setappdata(ax,'ManuscriptAxesPosition',style.metricPlotPosition);
    end
    text(ax,5,5*k,repmat('label ',1,k),'FontSize',8);
    file = fullfile(folder,sprintf('panel%d.eps',k));
    meta = export_manuscript_figure(fig,file);
    if k==2
        lp=ax.Legend.Position; ap=ax.Position;
        assert(lp(2)>ap(2)+ap(4),'Legend overlaps the plot rectangle.');
        assert(lp(1)>=0 && lp(1)+lp(3)<=1,'Legend extends outside canvas.');
    end
    epsText = fileread(file);
    boxes(k) = string(regexp(epsText,'(?m)^%%BoundingBox:[^\r\n]*','match','once'));
    assert(contains(epsText,'%%HiResBoundingBox: 0 0 468.000000 374.400000'));
    assert(meta.minimumPrintedFontPoints>=style.minimumPrintedFontSize);
    png = imfinfo(strrep(file,'.eps','.png')); imageSizes(k,:)=[png.Width png.Height];
    objects=findall(fig,'-property','FontSize');
    assert(all(arrayfun(@(obj) obj.FontSize>=style.fontSize,objects)));
    weightObjects=findall(fig,'-property','FontWeight');
    assert(all(arrayfun(@(obj) strcmpi(obj.FontWeight,style.fontWeight),weightObjects)), ...
        'All manuscript figure text must be bold.');
    clear closeFigure;
end
assert(boxes(1)==boxes(2),'Paired EPS canvases differ.');
assert(isequal(imageSizes(1,:),imageSizes(2,:)),'Paired PNG canvases differ.');
assert(style.geometryFigureWidth==style.measurementFigureWidth && ...
    style.geometryFigureHeight==style.measurementFigureHeight);
fprintf('Manuscript EPS canvas and font checks passed.\n');
end
