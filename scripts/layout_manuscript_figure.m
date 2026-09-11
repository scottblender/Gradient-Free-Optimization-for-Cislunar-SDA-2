function layout_manuscript_figure(fig,style)
%LAYOUT_MANUSCRIPT_FIGURE Reserve measured label/legend space at export fonts.
% Preserve schematic data limits and equal physical canvas dimensions.
axesObjects=findall(fig,'Type','axes');
for k=1:numel(axesObjects)
    ax=axesObjects(k);
    if strcmp(ax.Visible,'off'), continue; end % hand-positioned schematics
    ax.Units='normalized';
    wrap_label(ax.XLabel,30); wrap_label(ax.YLabel,25);
    % Automatic numeric axes need fewer ticks at manuscript font sizes.
    for axisName=["X","Y","Z"]
        ticks=ax.(axisName+"Tick");
        if strcmp(ax.(axisName+"TickMode"),'auto') && numel(ticks)>6
            ax.(axisName+"Tick")=ticks(unique(round(linspace(1,numel(ticks),5))));
        end
    end
    labels=string(ax.XTickLabel);
    if numel(labels)>4 && any(isnan(str2double(labels)))
        ax.XTickLabelRotation=35;
    end
    lgd=ax.Legend; legendHeight=0;
    if ~isempty(lgd) && isvalid(lgd)
        lgd.Units='normalized'; lgd.Box='off';
        lgd.FontWeight=style.fontWeight;
        lgd.Orientation='horizontal';
        lgd.Location='northoutside';

        % Preserve the plotter's intended top-centered legend arrangement.
        % Start from an explicit NumColumns value when one was requested;
        % otherwise prefer one row. If that is too wide, wrap once only so
        % the final legend occupies at most two rows. Do not search through
        % alternative column counts during export because that makes the
        % legend geometry change unexpectedly between preview and EPS.
        count=numel(lgd.String);
        if isprop(lgd,'NumColumnsMode') && strcmp(lgd.NumColumnsMode,'manual')
            columns=max(1,min(count,lgd.NumColumns));
        else
            columns=count;
        end
        minimumTwoRowColumns=max(1,ceil(count/style.legendMaxRows));
        columns=max(columns,minimumTwoRowColumns);
        lgd.NumColumns=columns;
        drawnow;
        pos=lgd.Position;
        if pos(3)>style.legendWidthLimit && columns>minimumTwoRowColumns
            columns=minimumTwoRowColumns;
            lgd.NumColumns=columns;
            drawnow;
            pos=lgd.Position;
        end
        rows=ceil(count/columns);
        assert(rows<=style.legendMaxRows,'Manuscript:LegendRows', ...
            'Legend requires more than %d rows; shorten legend text.',style.legendMaxRows);
        assert(pos(3)<=0.96,'Manuscript:LegendWidth', ...
            'Two-row legend exceeds canvas width; shorten legend text before export.');
        legendHeight=pos(4)+0.045;
        pos(1)=(1-pos(3))/2;
        pos(2)=0.975-pos(4);
        lgd.Position=pos;
    end
    if isappdata(ax,'ManuscriptAxesPosition')
        base=getappdata(ax,'ManuscriptAxesPosition');
    else
        base=style.geometryPlotPosition;
    end
    % TightInset includes the tick labels and axis labels, including rotated
    % categories. Iterate because MATLAB updates extents after repositioning.
    ax.PositionConstraint='innerposition'; ax.Position=base;
    for pass=1:3
        drawnow; inset=ax.TightInset;
        left=max(base(1),inset(1)+0.035);
        bottom=max(base(2),inset(2)+0.04);
        right=max(0.04,inset(3)+0.025);
        top=max(0.045,inset(4)+0.025)+legendHeight;
        width=1-left-right; height=1-bottom-top;
        assert(width>0.3 && height>0.25,'Manuscript:LayoutSpace', ...
            'Labels leave insufficient plotting space; shorten labels before export.');
        ax.Position=[left bottom width height];
    end
end
drawnow;
end

function wrap_label(label,limit)
value=label.String;
if ~(ischar(value) || (isstring(value) && isscalar(value))), return; end
value=char(value);
% Leave explicit multiline labels and mathematical LaTeX expressions alone.
if numel(value)<=limit || contains(value,'$') || contains(value,newline), return; end
words=strsplit(value); rows={}; line='';
for k=1:numel(words)
    if ~isempty(line) && numel(line)+1+numel(words{k})>limit
        rows{end+1}=line; line=words{k}; %#ok<AGROW>
    elseif isempty(line), line=words{k};
    else, line=[line ' ' words{k}]; %#ok<AGROW>
    end
end
rows{end+1}=line; label.String=rows;
end
