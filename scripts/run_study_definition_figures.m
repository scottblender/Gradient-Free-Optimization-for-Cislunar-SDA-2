function outputs = run_study_definition_figures(inspectFigures,clearDirectory)
%RUN_STUDY_DEFINITION_FIGURES Generate study-definition figures with cleanup.
%
% outputs = run_study_definition_figures(inspectFigures,clearDirectory)
%
% Inputs may be logical values or 1/0 numeric flags:
%   inspectFigures  - 1 previews each figure before export; 0 skips pauses.
%   clearDirectory - 1 removes results/study_definition_figures before the
%                    run; 0 preserves the existing directory contents.
%
% Examples:
%   run_study_definition_figures(0,1)  % regenerate from a clean directory
%   run_study_definition_figures(0,0)  % preserve existing outputs
%
% The cleanup is intentionally limited to the study-definition output
% directory and never removes raw optimization results or compiled Reviewer
% 2 result directories.

if nargin < 1 || isempty(inspectFigures), inspectFigures = true; end
if nargin < 2 || isempty(clearDirectory), clearDirectory = false; end

validateattributes(inspectFigures,{'logical','numeric'},{'scalar'});
validateattributes(clearDirectory,{'logical','numeric'},{'scalar'});
inspectFigures = logical(inspectFigures);
clearDirectory = logical(clearDirectory);

paths = setup_project();
outputDir = fullfile(paths.results,'study_definition_figures');

if clearDirectory && isfolder(outputDir)
    fprintf('Clearing study-definition output directory:\n  %s\n',outputDir);
    [ok,msg] = rmdir(outputDir,'s');
    assert(ok,'Could not clear study-definition output directory: %s',msg);
end

if ~isfolder(outputDir)
    mkdir(outputDir);
end

outputs = plot_study_definition_figures(inspectFigures);
outputs.outputDirectory = string(outputDir);
outputs.clearedDirectory = clearDirectory;
end
