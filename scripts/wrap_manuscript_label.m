function wrap_manuscript_label(label)
%WRAP_MANUSCRIPT_LABEL Construct a short multiline plain-text axis label.
s=label.String;
if ~(ischar(s) && isrow(s)) || numel(s)<=25 || contains(s,'$'), return; end
words=strsplit(s); lines={}; current='';
for k=1:numel(words)
    if ~isempty(current) && numel(current)+1+numel(words{k})>25
        lines{end+1}=current; current=words{k}; %#ok<AGROW>
    elseif isempty(current), current=words{k};
    else, current=[current ' ' words{k}]; %#ok<AGROW>
    end
end
lines{end+1}=current; label.String=lines;
end
