function value = abbreviate_manuscript_text(value)
%ABBREVIATE_MANUSCRIPT_TEXT Shorten repeated case names in dense figures.
% Keep abbreviations limited to manuscript graphics so scientific data and
% saved result labels retain their full canonical mission names.
value = string(value);
value = replace(value,"Lunar Gateway","LG");
value = replace(value,"Low-thrust transfer","LT");
value = replace(value,"Low-thrust","LT");
value = replace(value,"Gateway impulse","GI");
value = replace(value,"Gateway periods","LG periods");
end
