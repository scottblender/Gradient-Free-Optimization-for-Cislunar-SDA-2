function nCols = manuscript_legend_columns(lgdOrLabels,style)
%MANUSCRIPT_LEGEND_COLUMNS Choose the shared manuscript legend column count.
% Legends with three or more entries are arranged in two centered rows when
% possible: 4 entries -> 2 columns, 8 entries -> 4 columns, etc. One- and
% two-entry legends remain on a single row.

if nargin < 2 || isempty(style)
    style = reviewer2_paper_style();
end

if isscalar(lgdOrLabels) && isgraphics(lgdOrLabels) && isprop(lgdOrLabels,'String')
    labels = string(lgdOrLabels.String);
else
    labels = string(lgdOrLabels);
end
labels = labels(strlength(labels) > 0);
nItems = numel(labels);

if nItems <= 1
    nCols = 1;
elseif nItems == 2
    nCols = 2;
else
    nCols = ceil(nItems/2);
end

nCols = max(1,min(nCols,style.legendMaxColumns));
end
