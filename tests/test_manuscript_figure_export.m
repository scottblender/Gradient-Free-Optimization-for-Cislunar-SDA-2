function test_manuscript_figure_export()
% Data-free regression for paired export canvases and manuscript typography.
paths = setup_project(); %#ok<NASGU>
style = reviewer2_paper_style();
folder = tempname; mkdir(folder);
cleanup = onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
boxes = strings(4,1); imageSizes = zeros(4,2);
cameraViewAngleBefore = NaN;
for k = 1:4
    fig = figure('Visible','off','Units','inches', ...
        'Position',[1 1 style.metricFigureWidth style.metricFigureHeight], ...
        'PaperUnits','inches','PaperSize',[style.metricFigureWidth style.metricFigureHeight]);
    closeFigure = onCleanup(@() close(fig));
    ax = axes(fig);
    if k==1
        t=linspace(0,2*pi,200);
        plot3(ax,cos(t),sin(t),0.35*sin(2*t),'LineWidth',1.5);
        xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)'); zlabel(ax,'z (LU)');
        axis(ax,'equal'); view(ax,-37.5,30);
        % Reproduce the real failure mode: a plotter-generated/manual y tick
        % set that projects into one crowded screen-space corner.
        ax.YTick=[-1 -0.5 0 0.5 1];
        cameraViewAngleBefore=ax.CameraViewAngle;
    elseif k==4
        x=[1:4 6:9 11:14];
        V=repmat([20 55 20 3 2],12,1);
        bar(ax,x,V,'stacked','BarWidth',0.82);
        ax.XTick=x;
        ax.XTickLabel=repmat({'GA','PSO','ABC','ACO'},1,3);
        text(ax,mean(x(1:4)),104,'LG','HorizontalAlignment','center');
        text(ax,mean(x(5:8)),104,'LT','HorizontalAlignment','center');
        text(ax,mean(x(9:12)),104,'GI','HorizontalAlignment','center');
        ylim(ax,[0 108]);
        xlabel(ax,'Optimizer'); ylabel(ax,'Observer selections (%)');
        setappdata(ax,'ManuscriptAxesPosition',style.metricPlotPosition);
    else
        plot(ax,1:10,k*(1:10)); xlabel(ax,'x (LU)'); ylabel(ax,'y (LU)');
    end
    if k==2
        hold(ax,'on');
        for j=2:6, plot(ax,1:10,j*(1:10)); end
        ylabel(ax,'Mean effective position uncertainty (km)');
        ax.XTick = 1:3;
        ax.XTickLabel = {'Lunar Gateway','Low-thrust transfer','Gateway impulse'};
        lgd=legend(ax,{'Nominal Gateway','Target trajectory','Observer orbits', ...
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
    if k==1
        text(ax,0,0,0,'label','FontSize',8);
    elseif k~=4
        text(ax,5,5*k,repmat('label ',1,k),'FontSize',8);
    end
    file = fullfile(folder,sprintf('panel%d.eps',k));
    meta = export_manuscript_figure(fig,file);
    if k==1
        assert(abs(ax.CameraViewAngle-cameraViewAngleBefore) <= ...
            100*eps(max(1,cameraViewAngleBefore)), ...
            '3-D export should not change camera zoom.');
        ap=ax.Position;
        assert(ap(1)>=0 && ap(2)>=0 && ap(1)+ap(3)<=1 && ap(2)+ap(4)<=1, ...
            '3-D axes rectangle extends outside the export canvas.');
        assert(numel(ax.XTick)<=style.max3DXTicks && ...
            numel(ax.YTick)<=style.max3DYTicks && ...
            numel(ax.ZTick)<=style.max3DZTicks, ...
            '3-D tick labels were not reduced enough for manuscript export.');
        assert(numel(ax.YTick)==2, ...
            'Manual 3-D y ticks should be reduced to endpoint labels.');
    elseif k==4
        ticks=ax.XTick;
        within=diff(ticks(1:4));
        assert(all(within>1.15), ...
            'Repeated optimizer labels were not given visible within-group spacing.');
        assert(ticks(5)-ticks(4)>max(within), ...
            'Mission-block gap should remain larger than optimizer spacing.');
        assert(isappdata(ax,'ManuscriptOptimizerSpacingApplied'), ...
            'Optimizer-family spacing routine did not run.');
        bars=findall(ax,'-property','BarWidth');
        assert(~isempty(bars), ...
            'Family-selection stacked bars were not detected during export.');
        for b=1:numel(bars)
            if isprop(bars(b),'XData') && numel(bars(b).XData)==numel(ticks)
                assert(isequal(double(bars(b).XData(:).'),double(ticks(:).')), ...
                    'Family-selection bars did not move with optimizer tick labels.');
            end
        end
    end
    if k==2 || k==3
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
            legendText = string(lgd.String);
            assert(any(legendText=="Nominal LG") && any(legendText=="Target") && ...
                any(legendText=="Obs. orbits"), ...
                'Compact geometry abbreviations were not applied to the legend.');
            assert(isequal(string(ax.XTickLabel(:)),["LG";"LT";"GI"]), ...
                'Mission abbreviations were not applied to categorical ticks.');
        else
            assert(lgd.NumColumns==2 && rows==1, ...
                'Two-entry legends such as DRO/Moon must export as one row.');
        end
    end
    epsText = fileread(file);
    boxes(k) = string(regexp(epsText,'(?m)^%%BoundingBox:[^\r\n]*','match','once'));
    expectedHires = sprintf('%%%%HiResBoundingBox: 0 0 %.6f %.6f', ...
        72*style.metricFigureWidth,72*style.metricFigureHeight);
    assert(contains(epsText,expectedHires));
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
assert(style.metricFigureWidth>6.5 && style.metricFigureHeight>5.2, ...
    'Manuscript export canvas should be larger than the legacy dimensions.');
assert(style.geometryPlotPosition(3)>=0.80 && style.geometryPlotPosition(4)>=0.65, ...
    '3-D manuscript axes should remain larger than the legacy plot box.');
assert(~isfield(style,'geometryCameraZoom'), ...
    'Camera zoom should not be used for fixed-canvas manuscript exports.');
assert(style.geometryLegendGap<=0.005 && style.legendTopPadding<=0.020, ...
    'Legend spacing should remain compact to maximize manuscript plot area.');
assert(style.max3DXTicks<=3 && style.max3DYTicks<=2 && style.max3DZTicks<=3, ...
    '3-D manuscript tick density is too high for perspective projection.');
assert(style.familyOptimizerSpacingFactor>1.15, ...
    'Optimizer-family spacing factor should produce a visible separation.');
assert(style.geometryFigureWidth==style.measurementFigureWidth && ...
    style.geometryFigureHeight==style.measurementFigureHeight);
assert(style.visibilityFigureWidth==style.metricFigureWidth && ...
    style.visibilityFigureHeight>style.metricFigureHeight, ...
    'Keep-out schematic should retain manuscript width but use a taller canvas.');
fprintf(['Manuscript EPS canvas, contained 3-D geometry, manual/automatic sparse ' ...
    '3-D ticks, compact legend spacing, optimizer spacing, abbreviation, legend, ' ...
    'and font checks passed.\n']);
end
