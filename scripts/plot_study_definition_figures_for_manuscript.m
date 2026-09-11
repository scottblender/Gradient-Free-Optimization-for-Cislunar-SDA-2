function outputs = plot_study_definition_figures_for_manuscript( ...
    inspectFigures,sections,outputDirectory)
%PLOT_STUDY_DEFINITION_FIGURES_FOR_MANUSCRIPT Route definitions to one output.
% The legacy generator still uses its historical staging path internally,
% but manuscript runs never leave a second copy there. The staging folder is
% cleared before generation, every generated product is moved to the selected
% MANUSCRIPT_OUTPUT directory, and the staging folder is removed afterward.

if nargin<1 || isempty(inspectFigures), inspectFigures = false; end
if nargin<2 || isempty(sections), sections = "all"; end

paths = setup_project();
if nargin<3 || strlength(string(outputDirectory))==0
    outputDirectory = fullfile(paths.root,'MANUSCRIPT_OUTPUT');
end
outputDirectory = char(string(outputDirectory));
if ~isfolder(outputDirectory), mkdir(outputDirectory); end

legacyDirectory = fullfile(paths.results,'study_definition_figures');

% This directory is obsolete for the manuscript workflow. Remove stale
% products before generation so a manuscript run cannot leave old duplicates.
if isfolder(legacyDirectory)
    rmdir(legacyDirectory,'s');
end

% Suppress legacy path messages while the underlying generator runs.
transcript = evalc('outputs = plot_study_definition_figures(inspectFigures,sections);'); %#ok<NASGU>
outputs = relocate_output_paths(outputs,legacyDirectory,outputDirectory);

% Move any generated side products not explicitly referenced in the returned
% structure, then remove the staging directory completely.
if isfolder(legacyDirectory)
    listing = dir(fullfile(legacyDirectory,'**','*'));
    listing = listing(~[listing.isdir]);
    for k = 1:numel(listing)
        move_one_file(fullfile(listing(k).folder,listing(k).name),outputDirectory);
    end
    if isfolder(legacyDirectory), rmdir(legacyDirectory,'s'); end
end
assert(~isfolder(legacyDirectory), ...
    'Legacy study-definition output directory should not remain after manuscript generation.');

fprintf('Study-definition manuscript files: %s\n',outputDirectory);
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

% EPS export creates a matching PNG preview. Move the sidecar too when it is
% still present beside the EPS.
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
