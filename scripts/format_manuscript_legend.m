function format_manuscript_legend(ax,lgd,style,plotPosition)
%FORMAT_MANUSCRIPT_LEGEND Finalize a horizontal manuscript legend.
% Formatting occurs during plot construction only. Start with one row and
% add rows only when the legend would exceed the available figure width.

labels = abbreviate_manuscript_text(string(lgd.String));
lgd.String = cellstr(labels);
lgd.FontName = style.fontName;
lgd.FontSize = style.fontSize;
lgd.FontWeight = 'bold';
lgd.Box = 'off';
lgd.Orientation = 'horizontal';
lgd.Location = 'northoutside';
lgd.Units = 'normalized';

columns = min(numel(labels),style.legendMaxColumns);
lgd.NumColumns = max(columns,1);
drawnow;
while lgd.Position(3) > 0.92 && columns > 1
    columns = columns-1;
    lgd.NumColumns = columns;
    drawnow;
end

lp = lgd.Position;
lp(1) = max(0.02,0.5-lp(3)/2);
lp(2) = 0.97-lp(4);
lgd.Position = lp;
lgd.AutoUpdate = 'off';

plotPosition(4) = min(plotPosition(4),lp(2)-0.04-plotPosition(2));
assert(plotPosition(4) > 0.30 && lp(1) >= 0,'Manuscript:LegendSpace', ...
    'Legend text is too large for this canvas; shorten labels in the plotter.');
ax.Units = 'normalized';
ax.PositionConstraint = 'innerposition';
ax.Position = plotPosition;
drawnow;

% Remove only the crowded central zero from symmetric three-tick 3-D axes.
% Data limits and two-dimensional zero baselines are unchanged.
format_manuscript_ticks(ax);
end
