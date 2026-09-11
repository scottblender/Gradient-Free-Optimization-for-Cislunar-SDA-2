function format_manuscript_ticks(ax)
%FORMAT_MANUSCRIPT_TICKS Omit crowded central zero on symmetric 3-D axes.
% Construction-time only; retain limits, data, and 2-D bar-chart baselines.
if abs(ax.View(2)-90)<1e-8, return; end
for name=["X","Y","Z"]
    ticks=ax.(name+"Tick");
    if numel(ticks)~=3, continue; end
    tolerance=1e-10*max(abs(ticks));
    if ticks(1)<0 && ticks(3)>0 && abs(ticks(2))<=tolerance ...
            && abs(ticks(1)+ticks(3))<=tolerance
        labels=ax.(name+"TickLabel"); manual=strcmp(ax.(name+"TickLabelMode"),'manual');
        ax.(name+"Tick")=ticks([1 3]);
        if manual
            labels=cellstr(labels);
            if numel(labels)==3, ax.(name+"TickLabel")=labels([1 3]); end
        end
    end
end
end
