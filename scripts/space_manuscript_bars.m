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
    xlim(ax,[x(1)-0.60*gap,x(end)+0.60*gap]);
end

labels = abbreviate_manuscript_text(string(ax.XTickLabel));
ax.XTickLabel = cellstr(labels);

% Family-comparison figures can contain many short optimizer labels. Rotate
% those labels slightly even though the strings themselves are short.
if numel(labels) >= 8 || any(strlength(labels) > 10)
    ax.XTickLabelRotation = style.categoryLabelAngle;
end

% Abbreviate repeated case names used as in-axes annotations and long axis
% labels. This improves readability without changing any saved result data.
textObjects = findall(ax,'Type','text');
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
    catch
        % Ignore graphics proxy objects that do not expose writable strings.
    end
end
end
