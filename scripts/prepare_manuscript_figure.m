function prepare_manuscript_figure(fig)
%PREPARE_MANUSCRIPT_FIGURE Finish typography and layout before inspection/export.
% The EPS writer does not alter this completed scene. Do not change trajectory
% coordinates, data limits, camera direction/projection, or data aspect ratio.
style=reviewer2_paper_style();
objects=findall(fig,'-property','FontSize');
for k=1:numel(objects)
    obj=objects(k);
    if isprop(obj,'FontUnits'), obj.FontUnits='points'; end
    obj.FontSize=max(obj.FontSize,style.fontSize);
    if isprop(obj,'FontName'), obj.FontName=style.fontName; end
    if isprop(obj,'FontWeight'), obj.FontWeight=style.fontWeight; end
end
axesObjects=findall(fig,'Type','axes');
for k=1:numel(axesObjects)
    ax=axesObjects(k);
    if strcmp(ax.Visible,'off'), continue; end % explicitly constructed schematics
    is3D=abs(ax.View(2)-90)>1e-6;
    ax.Units='normalized';
    if is3D
        position=style.geometryPlotPosition;
        % Perspective line projection can be clipped by painters even when
        % all data are inside the data limits. Reserve page margins instead.
        ax.Clipping='off';
        lines=findall(ax,'Type','line');
        set(lines,'Clipping','off');
        names={'X','Y','Z'}; counts=[style.max3DXTicks style.max3DYTicks style.max3DZTicks];
        for j=1:3
            ticks=ax.([names{j} 'Tick']);
            if numel(ticks)>counts(j)
                ax.([names{j} 'Tick'])=ticks(unique(round(linspace(1,numel(ticks),counts(j)))));
            end
        end
    elseif isappdata(ax,'ManuscriptAxesPosition')
        position=style.metricPlotPosition;
    else
        % Keep the equal-sized geometry canvas for DRO/slot-planar panels.
        position=ax.Position;
    end
    lgd=ax.Legend;
    if ~isempty(lgd) && isvalid(lgd)
        lgd.Units='normalized'; lgd.Box='off'; lgd.FontWeight=style.fontWeight;
        lgd.Location='northoutside'; lgd.Orientation='horizontal';
        columns=min(numel(lgd.String),style.legendMaxColumns);
        lgd.NumColumns=columns; drawnow;
        while lgd.Position(3)>0.90 && columns>1
            columns=columns-1; lgd.NumColumns=columns; drawnow;
        end
        lp=lgd.Position;
        if lp(3)>0.94
            error('Manuscript:LegendTooWide','Legend is wider than the page; split its long labels.');
        end
        lp(1)=(1-lp(3))/2; lp(2)=0.96-lp(4);
        lgd.Position=lp; lgd.AutoUpdate='off';
        top=min(position(2)+position(4),lp(2)-0.045);
        position(4)=top-position(2);
    end
    ax.PositionConstraint='innerposition'; ax.Position=position;
    drawnow;
    % Only shrink to reserve genuinely missing label space. Do not expand
    % a 3-D axes to fill the page: this made projected labels cross the edge.
    inset=ax.TightInset;
    left=max(position(1),inset(1)+0.035);
    bottom=max(position(2),inset(2)+0.04);
    right=min(position(1)+position(3),0.965-inset(3));
    top=min(position(2)+position(4),0.96-inset(4));
    assert(right-left>0.3 && top-bottom>0.3,'Manuscript:LabelSpace', ...
        'Insufficient label space; split long labels before exporting.');
    ax.Position=[left bottom right-left top-bottom];
end
drawnow;
end
