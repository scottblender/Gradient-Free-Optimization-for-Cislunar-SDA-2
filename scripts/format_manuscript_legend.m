function format_manuscript_legend(ax,lgd,style,plotPosition)
%FORMAT_MANUSCRIPT_LEGEND Construct the final legend before preview/export.
% Called by plotters while building the scene, never by an export function.
lgd.FontName=style.fontName; lgd.FontSize=style.fontSize; lgd.FontWeight='bold';
lgd.Box='off'; lgd.Orientation='horizontal'; lgd.Location='northoutside';
lgd.Units='normalized';
columns=min(numel(lgd.String),style.legendMaxColumns);
lgd.NumColumns=columns; drawnow;
while lgd.Position(3)>0.92 && columns>1
    columns=columns-1; lgd.NumColumns=columns; drawnow;
format_manuscript_ticks(ax);
end
lp=lgd.Position; lp(1)=(1-lp(3))/2;
% A fixed top anchor keeps labels off the trajectory and inside the page.
lp(2)=0.97-lp(4); lgd.Position=lp; lgd.AutoUpdate='off';
plotPosition(4)=min(plotPosition(4),lp(2)-0.04-plotPosition(2));
assert(plotPosition(4)>0.30 && lp(1)>=0,'Manuscript:LegendSpace', ...
    'Legend text is too large for this canvas; shorten labels in the plotter.');
ax.Units='normalized'; ax.PositionConstraint='innerposition'; ax.Position=plotPosition;
drawnow;
format_manuscript_ticks(ax);
end
