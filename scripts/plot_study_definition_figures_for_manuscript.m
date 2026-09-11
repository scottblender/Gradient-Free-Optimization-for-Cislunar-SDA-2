function outputs = plot_study_definition_figures_for_manuscript( ...
    inspectFigures,sections,outputDirectory)
%PLOT_STUDY_DEFINITION_FIGURES_FOR_MANUSCRIPT Use one manuscript output root.
% The legacy definition generator is retained for standalone/backward-
% compatible use. The manuscript runner calls this adapter so the selected
% definition products end in the same MANUSCRIPT_OUTPUT directory as the
% final result figures, manifest, and table printouts.

if nargin<1 || isempty(inspectFigures), inspectFigures = false; end
if nargin<2 || isempty(sections), sections = "all"; end

paths = setup_project();
if nargin<3 || strlength(string(outputDirectory))==0
    outputDirectory = fullfile(paths.root,'MANUSCRIPT_OUTPUT');
end
outputDirectory = char(string(outputDirectory));
if ~isfolder(outputDirectory), mkdir(outputDirectory); end

legacyDirectory = fullfile(paths.results,'study_definition_figures');

% Suppress legacy path messages in the manuscript workflow. The underlying
% generator remains unchanged for direct standalone calls, then every file
% referenced by the returned structure is relocated into the selected
% manuscript output directory before control returns to the master runner.
transcript = evalc('outputs = plot_study_definition_figures(inspectFigures,sections);'); %#ok<NASGU>
outputs = relocate_output_paths(outputs,legacyDirectory,outputDirectory);

% Remove the legacy staging directory when this manuscript invocation left
% it empty. Pre-existing unrelated/stale files are deliberately not deleted.
if isfolder(legacyDirectory)
    listing = dir(legacyDirectory);
    listing = listing(~ismember({listing.name},{'.','..'}));
    if isempty(listing)
        rmdir(legacyDirectory);
    end
end

fprintf('Study-definition manuscript files: %s\n',outputDirectory);
end


function value = relocate_output_paths(value,legacyDirectory,outputDirectory)
% Recursively relocate file references returned by the legacy generator.

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
        value{k} = relocate_output_paths( ...
            value{k},legacyDirectory,outputDirectory);
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

% EPS export always creates a matching PNG preview. Move that sidecar even
% though the returned definition structure normally references only the EPS.
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
