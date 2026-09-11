function output = run_manuscript_figures(sections,varargin)
%RUN_MANUSCRIPT_FIGURES Single entry point for all manuscript figures.
% run_manuscript_figures                         % definitions + all results + saved MC
% run_manuscript_figures("definitions",'DefinitionSections',"measurement")
% run_manuscript_figures(["runtime","comparison"])
% run_manuscript_figures("results",'Reprocess',true)
% run_manuscript_figures("monte_carlo",'MonteCarloDirectory',folder)
% Reprocess=false reuses the last processed reports saved by this runner.
% Monte Carlo only replots saved samples; it never launches new evaluations.
if nargin < 1 || isempty(sections), sections = "all"; end
paths = setup_project();
p = inputParser;
addParameter(p,'OutputDirectory',fullfile(paths.root,'MANUSCRIPT_OUTPUT'));
addParameter(p,'Inspect',false,@(x) isscalar(x) && (islogical(x)||isnumeric(x)));
addParameter(p,'Reprocess',false,@(x) isscalar(x) && (islogical(x)||isnumeric(x)));
addParameter(p,'DefinitionSections',"all");
addParameter(p,'MonteCarloDirectory',"");
parse(p,varargin{:}); opts = p.Results;
sections = lower(string(sections(:)'));
resultSections = ["runtime","comparison","baseline","objective_screening"];
if isequal(sections,"all"), sections = ["definitions",resultSections,"monte_carlo"]; end
if isequal(sections,"results"), sections = resultSections; end
assert(all(ismember(sections,["definitions",resultSections,"monte_carlo"])), ...
    'Unknown manuscript figure section.');
sections = unique(sections,'stable');
compiled = fullfile(paths.root,'COMPILED_REVIEWER_2_RESULTS');
if ~isfolder(compiled), mkdir(compiled); end
output.directory = string(opts.OutputDirectory);
if ~isfolder(output.directory), mkdir(output.directory); end
sources = strings(0,1);
if ismember("definitions",sections)
    output.definitions = plot_study_definition_figures(logical(opts.Inspect),opts.DefinitionSections);
    % Gather only the selected figures, not stale files from other sections.
    sources = [sources;definition_files(output.definitions)];
end
selected = intersect(resultSections,sections,'stable');
if ~isempty(selected)
    cache = fullfile(compiled,'manuscript_reports.mat');
    reports = struct();
    if isfile(cache) && ~opts.Reprocess
        saved = load(cache,'reports'); reports = saved.reports;
    end
    needed = selected(~isfield(reports,cellstr(selected)));
    if opts.Reprocess, needed = selected; end
    if ~isempty(needed)
        fresh = run_reviewer2_results(needed,false);
        for section = needed, reports.(section) = fresh.(section); end
        save(cache,'reports','-v7.3');
    end
    renderReports = struct();
    % Render only the selected studies.
    for section = selected
        assert(isfolder(reports.(section).analysisDirectory), ...
            'Cached analysis is missing. Run with Reprocess=true.');
        renderReports.(section) = reports.(section);
    end
    output.results = make_reviewer2_final_figures(renderReports,true);
    for section = selected
        sourceDir = fullfile(reports.(section).analysisDirectory,'figures');
        files = dir(fullfile(sourceDir,'*.eps'));
        sources = [sources;string(fullfile({files.folder},{files.name}))']; %#ok<AGROW>
    end
end
if ismember("monte_carlo",sections)
    mcDir = string(opts.MonteCarloDirectory);
    if strlength(mcDir)==0
        candidates = dir(fullfile(compiled,'baseline_monte_carlo_*'));
        candidates = candidates([candidates.isdir]);
        if ~isempty(candidates)
            [~,idx] = max([candidates.datenum]);
            mcDir = string(fullfile(candidates(idx).folder,candidates(idx).name));
        end
    end
    if strlength(mcDir)>0
        samples = readtable(fullfile(mcDir,'baseline_monte_carlo_samples.csv'),'TextType','string');
        summary = readtable(fullfile(mcDir,'baseline_monte_carlo_summary.csv'),'TextType','string');
        details = plot_reviewer2_baseline_monte_carlo(samples,summary,mcDir,true);
        sources = [sources;fullfile(details.FigureDirectory,details.FigureStem+".eps")];
    else
        warning('Manuscript:MissingMC','No saved Monte Carlo samples found; skipped that section.');
    end
end
sources = unique(sources,'stable');
stems = strings(numel(sources),1);
for k = 1:numel(sources)
    [folder,stem] = fileparts(sources(k)); stems(k) = stem;
    assert(sum(stems(1:k)==stem)==1,'Duplicate figure name: %s',stem);
    copyfile(sources(k),fullfile(output.directory,stem+".eps"));
    png = fullfile(folder,stem+".png");
    if isfile(png), copyfile(png,fullfile(output.directory,stem+".png")); end
end
style = reviewer2_paper_style();
output.manifest = table(stems,sources,repmat(style.metricFigureWidth,numel(stems),1), ...
    repmat(style.metricFigureHeight,numel(stems),1), ...
    'VariableNames',{'FigureStem','SourceEPS','WidthInches','HeightInches'});
writetable(output.manifest,fullfile(output.directory,'figure_manifest.csv'));
fprintf('\nManuscript EPS/PNG files: %s\n',output.directory);
output.tables = print_manuscript_tables('OutputDirectory',output.directory);
fprintf('Place paired panels at equal widths, approximately %.1f inches each.\n',style.manuscriptPanelWidth);
end

function files = definition_files(value)
files = strings(0,1);
if isstruct(value)
    fields = fieldnames(value);
    for k = 1:numel(fields), files = [files;definition_files(value.(fields{k}))]; end %#ok<AGROW>
elseif isstring(value) || ischar(value)
    candidates = string(value(:));
    if ischar(value), candidates = string(value); end
    files = candidates(endsWith(candidates,'.eps') & isfile(candidates));
end
files = unique(files,'stable');
end
