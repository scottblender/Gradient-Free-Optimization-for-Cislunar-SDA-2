function value = abbreviate_manuscript_text(value)
%ABBREVIATE_MANUSCRIPT_TEXT Shorten repeated labels in manuscript graphics.
% Scientific data and saved result labels retain their canonical names.
value = string(value);
value = replace(value,"Lunar Gateway","LG");
value = replace(value,"Low-thrust transfer","LT");
value = replace(value,"Low-thrust","LT");
value = replace(value,"Gateway impulse","GI");
value = replace(value,"Gateway periods","LG periods");
value = replace(value,"Mean observer stability index","Mean stability index");
value = replace(value,"Mean runtime to 1200 FE (s)","Runtime to 1200 FE (s)");
value = replace(value,"Mean final best objective","Final best objective");
end
