function center_manuscript_content(fig,ax,lgd)
%CENTER_MANUSCRIPT_CONTENT Fit and center visible content on a fixed canvas.
% Larger paper fonts are accommodated by shrinking only the inner axes box.
% The outer EPS/PNG canvas is never resized. The same pass is used for 2-D
% and 3-D manuscript panels.

if nargin < 3, lgd = []; end
if isempty(fig) || ~isgraphics(fig) || isempty(ax) || ~isgraphics(ax), return; end

style = reviewer2_paper_style();
figUnits = fig.Units; axUnits = ax.Units;
hasLegend = ~isempty(lgd) && isgraphics(lgd);
if hasLegend, legendUnits = lgd.Units; else, legendUnits = ''; end
fig.Units = 'pixels'; ax.Units = 'pixels';
if hasLegend, lgd.Units = 'pixels'; end
drawnow;

fp = fig.Position; fw = fp(3); fh = fp(4);
availW = fw*(1-2*style.fitMarginFraction);
availH = fh*(1-2*style.fitMarginFraction);
ap = ax.Position; minW = style.fitMinimumAxesScale*ap(3); minH = style.fitMinimumAxesScale*ap(4);

if hasLegend
    lp = lgd.Position; lp(1) = 0.5*(fw-lp(3)); lgd.Position = lp; drawnow;
    ap = ax.Position; lp = lgd.Position; gap = lp(2)-(ap(2)+ap(4));
else
    gap = 0;
end

for k = 1:style.fitMaxIterations
    if hasLegend
        ap = ax.Position; lp = lgd.Position;
        lp(1) = 0.5*(fw-lp(3)); lp(2) = ap(2)+ap(4)+gap; lgd.Position = lp;
    end
    drawnow;
    b = content_bounds(ax,lgd,hasLegend);
    if b(2)-b(1) <= availW && b(4)-b(3) <= availH, break; end
    ap = ax.Position; nw = ap(3); nh = ap(4);
    if b(2)-b(1) > availW, nw = max(minW,nw*style.fitShrinkFactor); end
    if b(4)-b(3) > availH, nh = max(minH,nh*style.fitShrinkFactor); end
    if abs(nw-ap(3)) < 0.5 && abs(nh-ap(4)) < 0.5, break; end
    cx = ap(1)+0.5*ap(3); cy = ap(2)+0.5*ap(4);
    ap = [cx-0.5*nw cy-0.5*nh nw nh]; ax.Position = ap;
end
drawnow;

if hasLegend
    lp = lgd.Position; lp(1) = 0.5*(fw-lp(3)); lgd.Position = lp;
    legendLeft = lp(1); legendRight = lp(1)+lp(3);
else
    legendLeft = inf; legendRight = -inf;
end
drawnow;

ap = ax.Position; inset = ax.TightInset;
axesLeft = ap(1)-inset(1); axesRight = ap(1)+ap(3)+inset(3);
imbalance = @(dx) min(axesLeft+dx,legendLeft) - (fw-max(axesRight+dx,legendRight));
lo = -fw; hi = fw;
for k = 1:48
    mid = 0.5*(lo+hi);
    if imbalance(mid) < 0, lo = mid; else, hi = mid; end
end
ap(1) = ap(1)+0.5*(lo+hi); ax.Position = ap; drawnow;

b = content_bounds(ax,lgd,hasLegend); dy = 0.5*fh-0.5*(b(3)+b(4));
ap = ax.Position; ap(2) = ap(2)+dy; ax.Position = ap;
if hasLegend, lp = lgd.Position; lp(2) = lp(2)+dy; lgd.Position = lp; end
drawnow;

fig.Units = figUnits; ax.Units = axUnits;
if hasLegend, lgd.Units = legendUnits; end
end

function b = content_bounds(ax,lgd,hasLegend)
ap = ax.Position; ti = ax.TightInset;
b = [ap(1)-ti(1), ap(1)+ap(3)+ti(3), ap(2)-ti(2), ap(2)+ap(4)+ti(4)];
if hasLegend
    lp = lgd.Position;
    b = [min(b(1),lp(1)), max(b(2),lp(1)+lp(3)), min(b(3),lp(2)), max(b(4),lp(2)+lp(4))];
end
end
