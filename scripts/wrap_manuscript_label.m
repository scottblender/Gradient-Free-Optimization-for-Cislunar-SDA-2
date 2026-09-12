function wrap_manuscript_label(label)
%WRAP_MANUSCRIPT_LABEL Construct a compact multiline plain-text axis label.
% Abbreviate first so labels such as Tracking duration (LG periods) stay on
% one line and do not extend below the EPS canvas.
s = label.String;
if ~(ischar(s) && isrow(s)) || contains(s,'$'), return; end
s = char(abbreviate_manuscript_text(string(s)));
label.String = s;
maxChars = 32;
if numel(s)<=maxChars, return; end
words = strsplit(s); lines = {}; current = '';
for k = 1:numel(words)
    if ~isempty(current) && numel(current)+1+numel(words{k})>maxChars
        lines{end+1} = current;
        current = words{k};
    elseif isempty(current)
        current = words{k};
    else
        current = [current ' ' words{k}];
    end
end
lines{end+1} = current;
label.String = lines;
end
