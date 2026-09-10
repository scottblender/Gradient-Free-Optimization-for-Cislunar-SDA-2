function manifest = make_reviewer2_final_figures(reports,saveFigures)
%MAKE_REVIEWER2_FINAL_FIGURES Stable entry point for curated final figures.
%
% Each processed study keeps numerical analysis products in its timestamped
% compiled-results directory and all manuscript EPS/PNG exports in a single
% figures/ subdirectory. This wrapper preserves the existing renderers and
% collects their exports after rendering, so both new and previously loaded
% reports use the same output organization.
%
% The focused 1200-FE runtime figures use a dedicated renderer because the
% flat-colored categorical bar charts have one MATLAB Bar handle but five
% optimizer categories. Their optimizer identities are carried by x-axis
% labels, while the only bar-chart legend entry is the 6000-FE GA reference.
% All remaining Reviewer-2 figures are produced by the curated renderer.

if nargin < 2 || isempty(saveFigures), saveFigures = true; end
saveFigures = logical(saveFigures);
assert(isstruct(reports),'reports must come from run_reviewer2_results.');

reports = prepare_figure_directories(reports,saveFigures);

manifest = table(strings(0,1),strings(0,1),strings(0,1), ...
    'VariableNames',{'Study','FigureStem','Purpose'});
remaining = reports;

if isfield(reports,'runtime')
    baselineResults = table();
    if isfield(reports,'baseline') && isfield(reports.baseline,'results')
        baselineResults = reports.baseline.results;
    end
    runtimeManifest = make_reviewer2_runtime_figures( ...
        reports.runtime,baselineResults,saveFigures);
    manifest = [manifest;runtimeManifest]; %#ok<AGROW>
    remaining = rmfield(remaining,'runtime');
end

if ~isempty(fieldnames(remaining))
    otherManifest = make_reviewer2_curated_figures(remaining,saveFigures);
    manifest = [manifest;otherManifest]; %#ok<AGROW>
end

if saveFigures
    collect_rendered_figures(reports);
end
end


function reports = prepare_figure_directories(reports,saveFigures)
%PREPARE_FIGURE_DIRECTORIES Define one figure folder per compiled study.

fields = ["runtime","comparison","baseline","objective_screening"];
for k = 1:numel(fields)
    field = char(fields(k));
    if ~isfield(reports,field), continue; end

    r = reports.(field);
    if ~isfield(r,'analysisDirectory') || ...
            strlength(string(r.analysisDirectory)) == 0
        continue;
    end

    if ~isfield(r,'figureDirectory') || ...
            strlength(string(r.figureDirectory)) == 0
        r.figureDirectory = string(fullfile( ...
            char(r.analysisDirectory),'figures'));
    else
        r.figureDirectory = string(r.figureDirectory);
    end
    reports.(field) = r;

    if saveFigures
        if ~isfolder(r.figureDirectory), mkdir(r.figureDirectory); end
        clear_figure_files(r.figureDirectory);
    end
end
end


function collect_rendered_figures(reports)
%COLLECT_RENDERED_FIGURES Move EPS/PNG exports into each figures/ folder.
%
% The existing renderers deliberately continue to write beside their input
% analysis files while they run. Collecting afterward avoids changing their
% data lookup paths and leaves only non-figure analysis products at the study
% root.

fields = ["runtime","comparison","baseline","objective_screening"];
for k = 1:numel(fields)
    field = char(fields(k));
    if ~isfield(reports,field), continue; end

    r = reports.(field);
    if ~isfield(r,'analysisDirectory') || ~isfield(r,'figureDirectory')
        continue;
    end

    analysisDir = string(r.analysisDirectory);
    figureDir = string(r.figureDirectory);
    if ~isfolder(figureDir), mkdir(figureDir); end

    for pattern = ["*.eps","*.png"]
        files = dir(fullfile(char(analysisDir),char(pattern)));
        for j = 1:numel(files)
            source = fullfile(files(j).folder,files(j).name);
            destination = fullfile(char(figureDir),files(j).name);
            [ok,msg] = movefile(source,destination,'f');
            assert(ok,'Could not collect figure %s: %s',source,msg);
        end
    end
end
end


function clear_figure_files(figureDir)
%CLEAR_FIGURE_FILES Remove only prior paper figure exports.

for pattern = ["*.eps","*.png"]
    files = dir(fullfile(char(figureDir),char(pattern)));
    for k = 1:numel(files)
        delete(fullfile(files(k).folder,files(k).name));
    end
end
end
