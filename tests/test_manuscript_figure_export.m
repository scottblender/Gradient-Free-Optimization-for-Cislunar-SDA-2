function test_manuscript_figure_export()
% Data-free regression for paired export canvases and manuscript typography.
paths = setup_project(); %#ok<NASGU>
style = reviewer2_paper_style();
folder = tempname; mkdir(folder);
cleanup = onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
boxes = strings(3,1); imageSizes = zeros(3,2);
for k = 1:3
    fig = figure('Visible','off','Units','inches', ...
        'Position',[1 1 style.metricFigureWidth style.metricFigureHeight], ...
        'PaperUnits','inches','PaperSize',[style.metricFigureWidth style.metricFigureHeight]);
    closeFigure = onCleanup(@() close(fig));
    ax = axes(fig); plot(ax,1:10,k*(1:10)); xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)');
    if k==2
        hold(ax,'on');
        for j=2:6, plot(ax,1:10,j*(1:10)); end
        ylabel(ax,'Mean effective position uncertainty (km)');
        lgd=legend(ax,{'Nominal Gateway','Post-impulse','10 m/s burn', ...
            'Moon','L1','L2'},'Location','northoutside', ...
            'Orientation','horizontal','NumColumns',3);
        lgd.ItemTokenSize=[16 9];
        setappdata(ax,'ManuscriptAxesPosition',style.metricPlotPosition);
    elseif k==3
        hold(ax,'on');
        plot(ax,1:10,2*(1:10));
        legend(ax,{'DRO','Moon'},'Location','northeast', ...
            'Orientation','vertical','NumColumns',1);
        setappdata(ax,'ManuscriptAxesPosition',style.metricPlotPosition);
    end
    text(ax,5,5*k,repmat('label ',1,k),'FontSize',8);
    file = fullfile(folder,sprintf('panel%d.eps',k));
    meta = export_manuscript_figure(fig,file);
    if k>=2
        lgd=ax.Legend; lp=lgd.Position; ap=ax.Position;
        assert(lp(2)>ap(2)+ap(4),'Legend overlaps the plot rectangle.');
        assert(lp(1)>=0 && lp(1)+lp(3)<=1.01,'Legend extends outside canvas.');
        assert(abs((lp(1)+0.5*lp(3))-0.5)<0.02,'Legend is not centered above the plot.');
        assert(strcmpi(lgd.Orientation,'horizontal'),'Legend must remain horizontal.');
        rows=ceil(numel(lgd.String)/lgd.NumColumns);
        assert(rows<=style.legendMaxRows,'Legend must use at most two rows.');
        assert(lgd.FontSize>=style.fontSize,'Legend font size changed during export.');
        if k==2
            assert(lgd.NumColumns==3,'Impulse-style legend should remain two balanced rows.');
        else
            assert(lgd.NumColumns==2 && rows==1, ...
                'Two-entry legends such as DRO/Moon must export as one row.');
        end
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
assert(all(boxes==boxes(1)),'Paired EPS canvases differ.');
assert(all(all(imageSizes==imageSizes(1,:))),'Paired PNG canvases differ.');
assert(style.geometryFigureWidth==style.measurementFigureWidth && ...
    style.geometryFigureHeight==style.measurementFigureHeight);
assert(style.visibilityFigureWidth==style.metricFigureWidth && ...
    style.visibilityFigureHeight>style.metricFigureHeight, ...
    'Keep-out schematic should retain manuscript width but use a taller canvas.');
fprintf('Manuscript EPS canvas, legend, and font checks passed.\n');
end
