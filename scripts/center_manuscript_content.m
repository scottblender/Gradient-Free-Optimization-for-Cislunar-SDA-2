function center_manuscript_content(fig,ax,lgd)
%CENTER_MANUSCRIPT_CONTENT Center visible manuscript content horizontally.
%
% The legend is centered independently on the full figure bounding box. The
% axes are then shifted horizontally so the union of the axes/tick/label
% extent (from TightInset) and the legend has equal left/right margins. This
% is a construction-time layout pass; the EPS/PNG writers remain non-mutating.
%
% lgd is optional so the same helper can also center axes-only figures.

if nargin < 3
    lgd = [];
end
if isempty(fig) || ~isgraphics(fig) || isempty(ax) || ~isgraphics(ax)
    return;
end

figUnits = fig.Units;
axUnits = ax.Units;
hasLegend = ~isempty(lgd) && isgraphics(lgd);
if hasLegend
    legendUnits = lgd.Units;
else
    legendUnits = '';
end
cleanup = onCleanup(@() restore_units(fig,ax,lgd,figUnits,axUnits,legendUnits)); %#ok<NASGU>

fig.Units = 'pixels';
ax.Units = 'pixels';
if hasLegend
    lgd.Units = 'pixels';
end
drawnow;

figPosition = fig.Position;
figureWidth = figPosition(3);

% Always center the finished legend itself on the full figure canvas. This
% keeps short trajectory legends and long multi-entry rows visually centered
% regardless of the axes position or MATLAB's original legend anchor.
if hasLegend
    legendPosition = lgd.Position;
    legendPosition(1) = 0.5*(figureWidth-legendPosition(3));
    lgd.Position = legendPosition;
    drawnow;
    legendLeft = legendPosition(1);
    legendRight = legendPosition(1)+legendPosition(3);
else
    legendLeft = inf;
    legendRight = -inf;
end

% TightInset includes the rendered tick labels and x/y/z axis labels. Its
% left/right extents therefore represent the visible plot content rather than
% only the nominal axes rectangle.
axesPosition = ax.Position;
inset = ax.TightInset;
axesLeft = axesPosition(1)-inset(1);
axesRight = axesPosition(1)+axesPosition(3)+inset(3);

% Solve for the horizontal axes shift that makes the complete visible-content
% bounding box symmetric about the figure center while the legend remains
% fixed at the figure center. The margin imbalance is monotone in dx, so a
% short bisection is stable even when the outermost feature switches between
% an axis label/tick and a legend endpoint.
imbalance = @(dx) content_imbalance( ...
    axesLeft+dx,axesRight+dx,legendLeft,legendRight,figureWidth);

lo = -figureWidth;
hi = figureWidth;
fLo = imbalance(lo);
fHi = imbalance(hi);
if fLo <= 0 && fHi >= 0
    for k = 1:48
        mid = 0.5*(lo+hi);
        fMid = imbalance(mid);
        if fMid < 0
            lo = mid;
        else
            hi = mid;
        end
    end
    dx = 0.5*(lo+hi);
else
    % This should only occur if rendered content is wider than the canvas.
    % Fall back to centering the axes-visible extent itself.
    dx = 0.5*figureWidth-0.5*(axesLeft+axesRight);
end

axesPosition(1) = axesPosition(1)+dx;
ax.Position = axesPosition;
drawnow;
end


function value = content_imbalance( ...
    axesLeft,axesRight,legendLeft,legendRight,figureWidth)
contentLeft = min(axesLeft,legendLeft);
contentRight = max(axesRight,legendRight);
leftMargin = contentLeft;
rightMargin = figureWidth-contentRight;
value = leftMargin-rightMargin;
end


function restore_units(fig,ax,lgd,figUnits,axUnits,legendUnits)
if isgraphics(fig), fig.Units = figUnits; end
if isgraphics(ax), ax.Units = axUnits; end
if ~isempty(lgd) && isgraphics(lgd), lgd.Units = legendUnits; end
end
