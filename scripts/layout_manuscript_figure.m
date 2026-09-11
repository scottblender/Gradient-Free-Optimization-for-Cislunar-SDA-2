function layout_manuscript_figure(fig,style)
%LAYOUT_MANUSCRIPT_FIGURE Reserve measured label/legend space at export fonts.
% Preserve schematic data limits and equal physical canvas dimensions.
axesObjects=findall(fig,'Type','axes');
for k=1:numel(axesObjects)
    ax=axesObjects(k);
    if strcmp(ax.Visible,'off'), continue; end % hand-positioned schematics
    ax.Units='normalized';
    wrap_label(ax.XLabel,30); wrap_label(ax.YLabel,25);

    % Perspective 3-D axes need substantially fewer numeric ticks than 2-D
    % plots because projected labels can collapse into the same screen-space
    % corner. Apply explicit per-axis limits to both automatic and manual
    % numeric ticks; the y axis uses only two endpoints for the paper camera.
    % In contrast, 2-D metric/convergence plots are actively given readable
    % nice-number ticks when MATLAB's automatic selection is too sparse.
    viewAngles=view(ax);
    isThreeDimensional=abs(viewAngles(1))>1e-9 || abs(viewAngles(2)-90)>1e-9;
    for axisName=["X","Y","Z"]
        tickProperty=axisName+"Tick";
        labelProperty=axisName+"TickLabel";
        labelModeProperty=axisName+"TickLabelMode";
        ticks=ax.(tickProperty);
        if isThreeDimensional
            switch axisName
                case "X"
                    maxTicks=style.max3DXTicks;
                case "Y"
                    maxTicks=style.max3DYTicks;
                otherwise
                    maxTicks=style.max3DZTicks;
            end
            if numel(ticks)>maxTicks
                keep=unique(round(linspace(1,numel(ticks),maxTicks)));
                preserveManualLabels=false;
                manualLabels=string.empty(0,1);
                if isprop(ax,char(labelModeProperty)) && ...
                        strcmp(ax.(labelModeProperty),'manual')
                    labelsBefore=string(ax.(labelProperty));
                    if numel(labelsBefore)==numel(ticks)
                        manualLabels=labelsBefore(keep);
                        preserveManualLabels=true;
                    end
                end
                ax.(tickProperty)=ticks(keep);
                if preserveManualLabels
                    ax.(labelProperty)=cellstr(manualLabels);
                end
            end
        elseif axisName~="Z" && strcmp(ax.(axisName+"Scale"),'linear') && ...
                strcmp(ax.(axisName+"TickMode"),'auto')
            if axisName=="X"
                maxTicks=style.max2DXTicks;
            else
                maxTicks=style.max2DYTicks;
            end
            limits=ax.(axisName+"Lim");
            niceTicks=nice_linear_ticks(limits,maxTicks);
            % Never make an already-readable automatic axis sparser. Only
            % replace it when the nice-number set adds useful resolution or
            % when MATLAB produced more labels than the manuscript limit.
            if ~isempty(niceTicks) && ...
                    (numel(niceTicks)>numel(ticks) || numel(ticks)>maxTicks)
                ax.(tickProperty)=niceTicks;
            end
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

        % Keep legends centered above the axes and preserve the plotter's
        % intended one-row/two-row organization. Two-entry legends (e.g.
        % DRO + Moon) are always one row. Longer legends may wrap once, but
        % export never changes the font size or creates a third row.
        count=numel(lgd.String);
        if count<=2
            columns=count;
        elseif isprop(lgd,'NumColumnsMode') && strcmp(lgd.NumColumnsMode,'manual')
            columns=max(1,min(count,lgd.NumColumns));
        else
            columns=count;
        end
        minimumTwoRowColumns=max(1,ceil(count/style.legendMaxRows));
        columns=max(columns,minimumTwoRowColumns);
        lgd.NumColumns=columns;
        drawnow;
        pos=lgd.Position;

        % If the requested one-row layout is too wide, use the balanced
        % two-row arrangement. This changes only wrapping, not typography.
        if pos(3)>style.legendWidthLimit && columns>minimumTwoRowColumns
            columns=minimumTwoRowColumns;
            lgd.NumColumns=columns;
            drawnow;
            pos=lgd.Position;
        end

        % Long two-row legends can still be a few percent too wide at the
        % final 22-point export font. Reduce only the sample swatch width;
        % keep font size/weight and the two-row structure unchanged.
        if pos(3)>style.legendWidthLimit && isprop(lgd,'ItemTokenSize')
            token=lgd.ItemTokenSize;
            minimumTokenWidth=8;
            while pos(3)>style.legendWidthLimit && token(1)>minimumTokenWidth
                token(1)=max(minimumTokenWidth,token(1)-2);
                lgd.ItemTokenSize=token;
                drawnow;
                pos=lgd.Position;
            end
        end

        rows=ceil(count/columns);
        assert(rows<=style.legendMaxRows,'Manuscript:LegendRows', ...
            'Legend requires more than %d rows; shorten legend text.',style.legendMaxRows);
        if pos(3)>0.995
            warning('Manuscript:LegendWidth', ...
                ['Legend remains wider than the preferred canvas width after ' ...
                 'two-row wrapping and compact swatches; exporting centered.']);
        end
        legendHeight=pos(4)+style.legendTopPadding;
        pos(1)=max(0.002,(1-pos(3))/2);
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

    % Final legend placement is measured directly from the axes rectangle.
    % TightInset is already accounted for when sizing the axes above; adding
    % it again here made the legend appear unchanged even when the requested
    % gap was reduced. Use one explicit axes-to-legend gap instead.
    if ~isempty(lgd) && isvalid(lgd)
        drawnow;
        pos=lgd.Position;
        desiredBottom=ax.Position(2)+ax.Position(4)+style.legendAxesGap;
        maximumBottom=0.99-pos(4);
        pos(1)=max(0.002,(1-pos(3))/2);
        pos(2)=min(desiredBottom,maximumBottom);
        lgd.Position=pos;
        setappdata(ax,'ManuscriptFinalLegendGap', ...
            lgd.Position(2)-(ax.Position(2)+ax.Position(4)));
    end

    % Do not camera-zoom 3-D plots at export time. MATLAB already frames the
    % current data limits inside the final axes box; an additional camzoom can
    % push projected trajectories or markers outside the fixed EPS rectangle.
    % Figure size is increased through the shared axes rectangle instead.
end
drawnow;
end

function ticks=nice_linear_ticks(limits,maxTicks)
%NICE_LINEAR_TICKS Dense readable 1/2/2.5/5-decade ticks within fixed limits.
limits=double(limits(:).');
ticks=[];
if numel(limits)~=2 || any(~isfinite(limits)) || limits(2)<=limits(1) || maxTicks<2
    return;
end
span=limits(2)-limits(1);
roughStep=span/max(2,maxTicks-1);
if ~isfinite(roughStep) || roughStep<=0, return; end
power=10^floor(log10(roughStep));
steps=power*[1 2 2.5 5 10];
tolerance=1e-10*max(1,max(abs(limits)));
for step=steps
    first=ceil((limits(1)-tolerance)/step)*step;
    last=floor((limits(2)+tolerance)/step)*step;
    candidate=first:step:last;
    if numel(candidate)>=2 && numel(candidate)<=maxTicks
        candidate(abs(candidate)<100*eps(max(1,max(abs(candidate)))))=0;
        ticks=candidate;
        return;
    end
end
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
