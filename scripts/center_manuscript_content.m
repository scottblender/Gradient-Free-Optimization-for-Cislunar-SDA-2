function center_manuscript_content(fig,ax,lgd)
%CENTER_MANUSCRIPT_CONTENT Fit and center visible content on a fixed canvas.
% Larger paper fonts are accommodated by shrinking only the inner axes box.
% The outer EPS/PNG canvas is never resized. The same pass is used for 2-D
% and 3-D manuscript panels. A final explicit bounds check guarantees that
% tick labels, axis labels, and any in-panel legend remain inside the safe
% export margins after centering.

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
ap = ax.Position;
minW = style.fitMinimumAxesScale*ap(3);
minH = style.fitMinimumAxesScale*ap(4);

if hasLegend
    lp = lgd.Position;
    lp(1) = 0.5*(fw-lp(3));
    lgd.Position = lp;
    drawnow;
    ap = ax.Position;
    lp = lgd.Position;
    gap = lp(2)-(ap(2)+ap(4));
else
    gap = 0;
end

% First fit the total visible span to the safe canvas. This pass preserves the
% configured font sizes and changes only the axes rectangle when necessary.
for k = 1:style.fitMaxIterations
    if hasLegend
        ap = ax.Position;
        lp = lgd.Position;
        lp(1) = 0.5*(fw-lp(3));
        lp(2) = ap(2)+ap(4)+gap;
        lgd.Position = lp;
    end
    drawnow;
    b = content_bounds(ax,lgd,hasLegend);
    if b(2)-b(1) <= availW && b(4)-b(3) <= availH, break; end

    ap = ax.Position;
    nw = ap(3);
    nh = ap(4);
    if b(2)-b(1) > availW, nw = max(minW,nw*style.fitShrinkFactor); end
    if b(4)-b(3) > availH, nh = max(minH,nh*style.fitShrinkFactor); end
    if abs(nw-ap(3)) < 0.5 && abs(nh-ap(4)) < 0.5, break; end

    cx = ap(1)+0.5*ap(3);
    cy = ap(2)+0.5*ap(4);
    ax.Position = [cx-0.5*nw cy-0.5*nh nw nh];
end
drawnow;

% Center horizontally using the complete visible axes extent, not just the
% inner axes rectangle. The legend remains centered on the figure itself.
if hasLegend
    lp = lgd.Position;
    lp(1) = 0.5*(fw-lp(3));
    lgd.Position = lp;
    legendLeft = lp(1);
    legendRight = lp(1)+lp(3);
else
    legendLeft = inf;
    legendRight = -inf;
end
drawnow;

ap = ax.Position;
inset = ax.TightInset;
axesLeft = ap(1)-inset(1);
axesRight = ap(1)+ap(3)+inset(3);
imbalance = @(dx) min(axesLeft+dx,legendLeft) - ...
    (fw-max(axesRight+dx,legendRight));
lo = -fw;
hi = fw;
for k = 1:48
    mid = 0.5*(lo+hi);
    if imbalance(mid) < 0, lo = mid; else, hi = mid; end
end
ap(1) = ap(1)+0.5*(lo+hi);
ax.Position = ap;
drawnow;

% Center the complete axes/legend block vertically.
b = content_bounds(ax,lgd,hasLegend);
dy = 0.5*fh-0.5*(b(3)+b(4));
ap = ax.Position;
ap(2) = ap(2)+dy;
ax.Position = ap;
if hasLegend
    lp = lgd.Position;
    lp(2) = lp(2)+dy;
    lgd.Position = lp;
end
drawnow;

% Centering can move a long axis label or rotated tick slightly outside the
% requested safe margin even when the total span fits. Run a final explicit
% edge check and translate/shrink only the axes box until every visible bound
% lies inside the protected canvas.
enforce_safe_bounds(ax,lgd,hasLegend,fw,fh,style,minW,minH,gap);
drawnow;

fig.Units = figUnits;
ax.Units = axUnits;
if hasLegend, lgd.Units = legendUnits; end
end

function enforce_safe_bounds(ax,lgd,hasLegend,fw,fh,style,minW,minH,gap)
marginX = style.fitMarginFraction*fw;
marginY = style.fitMarginFraction*fh;
safe = [marginX,fw-marginX,marginY,fh-marginY];

for k = 1:style.fitMaxIterations
    drawnow;
    b = content_bounds(ax,lgd,hasLegend);
    if b(1) >= safe(1) && b(2) <= safe(2) && ...
            b(3) >= safe(3) && b(4) <= safe(4)
        return;
    end

    % Translate first whenever the visible span already fits within the safe
    % width/height. This preserves the largest possible data rectangle.
    dx = 0;
    dy = 0;
    if b(2)-b(1) <= safe(2)-safe(1)
        if b(1) < safe(1), dx = safe(1)-b(1); end
        if b(2)+dx > safe(2), dx = dx-(b(2)+dx-safe(2)); end
    end
    if b(4)-b(3) <= safe(4)-safe(3)
        if b(3) < safe(3), dy = safe(3)-b(3); end
        if b(4)+dy > safe(4), dy = dy-(b(4)+dy-safe(4)); end
    end

    if abs(dx) > 0.25 || abs(dy) > 0.25
        ap = ax.Position;
        ap(1:2) = ap(1:2)+[dx dy];
        ax.Position = ap;
        if hasLegend
            lp = lgd.Position;
            lp(1:2) = lp(1:2)+[dx dy];
            lgd.Position = lp;
        end
        drawnow;
        b = content_bounds(ax,lgd,hasLegend);
        if b(1) >= safe(1) && b(2) <= safe(2) && ...
                b(3) >= safe(3) && b(4) <= safe(4)
            return;
        end
    end

    % If the visible span itself is still too large, contract only the inner
    % axes box. Font sizes and the fixed EPS canvas remain unchanged.
    ap = ax.Position;
    nw = ap(3);
    nh = ap(4);
    if b(1) < safe(1) || b(2) > safe(2)
        nw = max(minW,nw*style.fitShrinkFactor);
    end
    if b(3) < safe(3) || b(4) > safe(4)
        nh = max(minH,nh*style.fitShrinkFactor);
    end
    if abs(nw-ap(3)) < 0.5 && abs(nh-ap(4)) < 0.5
        break;
    end

    cx = ap(1)+0.5*ap(3);
    cy = ap(2)+0.5*ap(4);
    ap = [cx-0.5*nw cy-0.5*nh nw nh];
    ax.Position = ap;
    if hasLegend
        lp = lgd.Position;
        lp(1) = 0.5*(fw-lp(3));
        lp(2) = ap(2)+ap(4)+gap;
        lgd.Position = lp;
    end
end

% Final translation guards against sub-pixel rounding at the EPS boundary.
drawnow;
b = content_bounds(ax,lgd,hasLegend);
dx = max(0,safe(1)-b(1)) - max(0,b(2)-safe(2));
dy = max(0,safe(3)-b(3)) - max(0,b(4)-safe(4));
ap = ax.Position;
ap(1:2) = ap(1:2)+[dx dy];
ax.Position = ap;
if hasLegend
    lp = lgd.Position;
    lp(1:2) = lp(1:2)+[dx dy];
    lgd.Position = lp;
end
end

function b = content_bounds(ax,lgd,hasLegend)
ap = ax.Position;
ti = ax.TightInset;
b = [ap(1)-ti(1), ap(1)+ap(3)+ti(3), ...
    ap(2)-ti(2), ap(2)+ap(4)+ti(4)];
if hasLegend
    lp = lgd.Position;
    b = [min(b(1),lp(1)), max(b(2),lp(1)+lp(3)), ...
        min(b(3),lp(2)), max(b(4),lp(2)+lp(4))];
end
end
