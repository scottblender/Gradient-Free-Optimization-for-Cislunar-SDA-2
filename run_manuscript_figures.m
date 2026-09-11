function output = run_manuscript_figures(sections,varargin)
%RUN_MANUSCRIPT_FIGURES Single entry point for all manuscript figures.
% run_manuscript_figures(0)                      % keep existing exports
% run_manuscript_figures(1)                      % clear final exports first
% run_manuscript_figures("parallel")             % saved serial/parallel GA plots
% run_manuscript_figures("definitions",'DefinitionSections',"measurement")
% run_manuscript_figures(["runtime","comparison"])
% run_manuscript_figures("results",'Reprocess',true)
% run_manuscript_figures("monte_carlo",'MonteCarloDirectory',folder)
% Reprocess=false reuses the last processed reports saved by this runner.
% Monte Carlo only replots saved samples; it never launches new evaluations.
if nargin < 1 || isempty(sections), sections = "all"; end
clearFirst = false;
if isnumeric(sections) || islogical(sections)
    assert(isscalar(sections) && ismember(sections,[0 1]),'Use 0 to keep exports or 1 to clear exports.');
    clearFirst = logical(sections); sections = "all";
end
paths = setup_project();
p = inputParser;
addParameter(p,'OutputDirectory',fullfile(paths.root,'MANUSCRIPT_OUTPUT'));
addParameter(p,'ClearDirectory',clearFirst,@(x) isscalar(x) && ismember(x,[0 1]));
addParameter(p,'ParallelSpeedDirectory',"");
addParameter(p,'Inspect',false,@(x) isscalar(x) && (islogical(x)||isnumeric(x)));
addParameter(p,'Reprocess',false,@(x) isscalar(x) && (islogical(x)||isnumeric(x)));
addParameter(p,'DefinitionSections',"all");
addParameter(p,'MonteCarloDirectory',"");
parse(p,varargin{:}); opts = p.Results;
sections = lower(string(sections(:)'));
resultSections = ["runtime","comparison","baseline","objective_screening"];
if isequal(sections,"all"), sections = ["definitions",resultSections,"monte_carlo","parallel"]; end
if isequal(sections,"results"), sections = resultSections; end
assert(all(ismember(sections,["definitions",resultSections,"monte_carlo","parallel"])), ...
    'Unknown manuscript figure section.');
sections = unique(sections,'stable');
compiled = fullfile(paths.root,'COMPILED_REVIEWER_2_RESULTS');
if ~isfolder(compiled), mkdir(compiled); end
output.directory = string(opts.OutputDirectory);
if ~isfolder(output.directory), mkdir(output.directory); end
if opts.ClearDirectory
    % Clear final exports only; retain benchmark subdirectories and raw data.
    patterns = ["*.eps","*.png","figure_manifest.csv","manuscript_tables.txt", ...
        "manuscript_tables.tex","parallel_speed_results.csv","parallel_speed_summary.txt"];
    for pattern = patterns
        old = dir(fullfile(output.directory,pattern));
        for k = 1:numel(old)
            if ~old(k).isdir, delete(fullfile(old(k).folder,old(k).name)); end
        end
    end
end
sources = strings(0,1);
if ismember("definitions",sections)
    output.definitions = plot_study_definition_figures_for_manuscript( ...
        logical(opts.Inspect),opts.DefinitionSections,output.directory);
    % Definition products already live in the shared manuscript directory;
    % collect only the selected figure paths for the manifest.
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
if ismember("parallel",sections)
    benchmarkDir = string(opts.ParallelSpeedDirectory);
    if strlength(benchmarkDir)==0
        candidates = dir(fullfile(paths.root,'MANUSCRIPT_OUTPUT', ...
            'parallel_speed_lunar_gateway_*','parallel_speed_convergence.mat'));
        [~,order] = sort([candidates.datenum],'descend');
        for idx = order
            candidate = fullfile(candidates(idx).folder,candidates(idx).name);
            saved = load(candidate,'benchmark');
            if saved.benchmark.complete && saved.benchmark.budget==6000
                benchmarkDir = string(candidates(idx).folder); break;
            end
        end
    end
    if strlength(benchmarkDir)>0
        sources = [sources;plot_parallel_speed( ...
            fullfile(benchmarkDir,'parallel_speed_convergence.mat'),output.directory)];
    else
        warning('Manuscript:MissingParallel', ...
            'No completed LG 6000-FE benchmark. Run test_parallel_speed first; skipped parallel plots.');
    end
end
sources = unique(sources,'stable');
stems = strings(numel(sources),1);
for k = 1:numel(sources)
    [folder,stem] = fileparts(sources(k)); stems(k) = stem;
    assert(sum(stems(1:k)==stem)==1,'Duplicate figure name: %s',stem);
    if string(folder)==output.directory, continue; end
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
