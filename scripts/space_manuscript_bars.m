function space_manuscript_bars(ax,style)
%SPACE_MANUSCRIPT_BARS Improve dense categorical-axis readability.
% Called during axes construction after bars/error bars have been created.
% Bar centers and error-bar coordinates are preserved.

bars = findall(ax,'Type','bar');
if isempty(bars), return; end

x = unique(vertcat(bars.XData));
x = sort(x(:));
if numel(x) > 1 && isnumeric(x)
    gap = min(diff(x));
    xlim(ax,[x(1)-0.65*gap,x(end)+0.65*gap]);
end

labels = abbreviate_manuscript_text(string(ax.XTickLabel));
ax.XTickLabel = cellstr(labels);

% Dense family-comparison figures contain repeated optimizer names. A larger
% rotation keeps GA/PSO/ABC/ACO distinct without shrinking the manuscript font.
if numel(labels) >= 8 || any(strlength(labels) > 10)
    ax.XTickLabelRotation = style.categoryLabelAngle;
end

% Abbreviate repeated case names used as in-axes annotations and long axis
% labels. Mission-group labels are nudged upward slightly above the 100% bars.
textObjects = findall(ax,'Type','text');
yRange = diff(ylim(ax));
for k = 1:numel(textObjects)
    try
        original = string(textObjects(k).String);
        shortened = abbreviate_manuscript_text(original);
        if ~isequal(original,shortened)
            if isscalar(shortened)
                textObjects(k).String = char(shortened);
            else
                textObjects(k).String = cellstr(shortened);
            end
        end
        if isscalar(shortened) && any(shortened == ["LG","LT","GI"])
            p = textObjects(k).Position;
            p(2) = p(2) + 0.02*yRange;
            textObjects(k).Position = p;
        end
    catch
        % Ignore graphics proxy objects that do not expose writable strings.
    end
end
end
