function outputs = plot_study_definition_figures_for_manuscript( ...
    inspectFigures,sections,outputDirectory)
%PLOT_STUDY_DEFINITION_FIGURES_FOR_MANUSCRIPT Route definitions to one output.
% Manuscript-specific figures are generated in their final styled state before
% export. Legacy study-definition products are still supported for catalog,
% measurement, and case sections, but no persistent copy is left in the old
% results/study_definition_figures directory.

if nargin<1 || isempty(inspectFigures), inspectFigures = false; end
if nargin<2 || isempty(sections), sections = "all"; end

paths = setup_project();
if nargin<3 || strlength(string(outputDirectory))==0
    outputDirectory = fullfile(paths.root,'MANUSCRIPT_OUTPUT');
end
outputDirectory = char(string(outputDirectory));
if ~isfolder(outputDirectory), mkdir(outputDirectory); end

requested = lower(string(sections(:)'));
available = ["catalog","slots","visibility","measurement","cases"];
if isequal(requested,"all"), requested = available; end
assert(all(ismember(requested,available)),'Unknown definition figure section.');
requested = unique(requested,'stable');

legacyDirectory = fullfile(paths.results,'study_definition_figures');
if isfolder(legacyDirectory), rmdir(legacyDirectory,'s'); end
outputs = struct();

% Slot and visibility figures have manuscript-specific generators so marker
% fill and geometry sizing are completed during generation, never at export.
legacySections = requested(~ismember(requested,["slots","visibility"]));
if ~isempty(legacySections)
    transcript = evalc('legacyOutputs = plot_study_definition_figures(inspectFigures,legacySections);'); %#ok<NASGU>
    legacyOutputs = relocate_output_paths(legacyOutputs,legacyDirectory,outputDirectory);
    outputs = merge_struct(outputs,legacyOutputs);

    if isfolder(legacyDirectory)
        listing = dir(fullfile(legacyDirectory,'**','*'));
        listing = listing(~[listing.isdir]);
        for k = 1:numel(listing)
            move_one_file(fullfile(listing(k).folder,listing(k).name),outputDirectory);
        end
    end
end

if ismember("slots",requested)
    outputs.slots = plot_slot_definition_manuscript(inspectFigures,outputDirectory);
end
if ismember("visibility",requested)
    outputs.visibilityGeometry = plot_visibility_keepout_manuscript( ...
        inspectFigures,outputDirectory);
end

if isfolder(legacyDirectory), rmdir(legacyDirectory,'s'); end
assert(~isfolder(legacyDirectory), ...
    'Legacy study-definition output directory should not remain after manuscript generation.');

fprintf('Study-definition manuscript files: %s\n',outputDirectory);
end


function target = merge_struct(target,source)
fields = fieldnames(source);
for k = 1:numel(fields)
    target.(fields{k}) = source.(fields{k});
end
end


function value = relocate_output_paths(value,legacyDirectory,outputDirectory)
if isstruct(value)
    fields = fieldnames(value);
    for k = 1:numel(value)
        for f = 1:numel(fields)
            value(k).(fields{f}) = relocate_output_paths( ...
                value(k).(fields{f}),legacyDirectory,outputDirectory);
        end
    end
    return;
end
if iscell(value)
    for k = 1:numel(value)
        value{k} = relocate_output_paths(value{k},legacyDirectory,outputDirectory);
    end
    return;
end
if isstring(value)
    for k = 1:numel(value)
        candidate = value(k);
        if strlength(candidate)>0 && isfile(candidate) && ...
                startsWith(candidate,string(legacyDirectory),'IgnoreCase',true)
            value(k) = string(move_one_file(char(candidate),outputDirectory));
        end
    end
    return;
end
if ischar(value) && isrow(value) && isfile(value) && ...
        startsWith(string(value),string(legacyDirectory),'IgnoreCase',true)
    value = move_one_file(value,outputDirectory);
end
end


function destination = move_one_file(source,outputDirectory)
[~,name,extension] = fileparts(source);
destination = fullfile(outputDirectory,[name extension]);
if ~strcmpi(source,destination)
    if isfile(destination), delete(destination); end
    [ok,message] = movefile(source,destination);
    assert(ok,'Could not move manuscript definition file %s: %s',source,message);
end

if strcmpi(extension,'.eps')
    sourcePng = fullfile(fileparts(source),[name '.png']);
    destinationPng = fullfile(outputDirectory,[name '.png']);
    if isfile(sourcePng) && ~strcmpi(sourcePng,destinationPng)
        if isfile(destinationPng), delete(destinationPng); end
        [ok,message] = movefile(sourcePng,destinationPng);
        assert(ok,'Could not move manuscript PNG preview %s: %s',sourcePng,message);
    end
end
end
