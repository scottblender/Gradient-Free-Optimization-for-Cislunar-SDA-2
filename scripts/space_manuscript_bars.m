function space_manuscript_bars(ax,style)
%SPACE_MANUSCRIPT_BARS Set consistent gaps and readable categorical labels.
% Called during axes construction after bars/error bars have been created.
bars=findall(ax,'Type','bar');
if isempty(bars), return; end
% Preserve the bar centers and error-bar coordinates. Only widths, outside
% padding, and label angle change; stacked percentages remain stacked.
x=unique(vertcat(bars.XData)); x=x(:); x=sort(unique(x));
if numel(x)>1 && isnumeric(x)
    gap=min(diff(x)); xlim(ax,[x(1)-0.65*gap,x(end)+0.65*gap]);
end
labels=string(ax.XTickLabel);
if any(strlength(labels)>10), ax.XTickLabelRotation=style.categoryLabelAngle; end
end
