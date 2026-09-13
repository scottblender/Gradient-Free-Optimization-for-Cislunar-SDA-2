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
    manifest = [manifest;runtimeManifest];
    remaining = rmfield(remaining,'runtime');
end

if ~isempty(fieldnames(remaining))
    otherManifest = make_reviewer2_curated_figures(remaining,saveFigures);
    manifest = [manifest;otherManifest];
end

% Add an intuitive map of the evaluated 6000-FE study slices when both the
% optimizer-comparison and objective-component data are available. The cube
% deliberately omits BO because BO belongs only to the separate 1200-FE pilot.
if isfield(reports,'comparison') && isfield(reports,'objective_screening')
    cubeStem = "study_design_cube_6000";
    plot_study_design_cube(reports.comparison.analysisDirectory,cubeStem,saveFigures);
    manifest = [manifest;table("study_design",cubeStem, ...
        "Evaluated target/objective/optimizer combinations at 6000 FE.", ...
        'VariableNames',manifest.Properties.VariableNames)];
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


function plot_study_design_cube(outputDir,stem,saveFigures)
%PLOT_STUDY_DESIGN_CUBE Visualize the evaluated 6000-FE study combinations.

if ~saveFigures, return; end
style = reviewer2_paper_style();
fig = figure('Color','w','Units','inches', ...
    'Position',[1 1 style.metricFigureWidth style.metricFigureHeight], ...
    'PaperUnits','inches', ...
    'PaperSize',[style.metricFigureWidth style.metricFigureHeight], ...
    'PaperPosition',[0 0 style.metricFigureWidth style.metricFigureHeight], ...
    'PaperPositionMode','manual','Renderer','painters','InvertHardcopy','off');
ax = axes(fig,'Units','normalized','Position',[0.17 0.19 0.67 0.67]);
hold(ax,'on'); box(ax,'on'); grid(ax,'on');

% Extended comparison: J111 x all three targets x GA/PSO/ABCO/ACO.
[x1,y1,z1] = ndgrid(1:3,1,1:4);
hBenchmark = scatter3(ax,x1(:),y1(:),z1(:),88,'o','filled');

% Objective-component sweeps: J100/J010/J001 x all three targets x GA.
[x2,y2] = ndgrid(1:3,2:4);
z2 = ones(size(x2));
hComponents = scatter3(ax,x2(:),y2(:),z2(:),92,'d','filled');

xlim(ax,[0.6 3.4]); ylim(ax,[0.6 4.4]); zlim(ax,[0.6 4.4]);
xticks(ax,1:3); xticklabels(ax,{'LG','LT','GI'});
yticks(ax,1:4); yticklabels(ax,{'J_{111}','J_{100}','J_{010}','J_{001}'});
zticks(ax,1:4); zticklabels(ax,{'GA','PSO','ABCO','ACO'});
xlabel(ax,'Target case','FontWeight','bold');
ylabel(ax,'Objective','FontWeight','bold');
zlabel(ax,'Optimizer','FontWeight','bold');
view(ax,38,24);
set(ax,'FontName',style.fontName,'FontSize',style.fontSize, ...
    'FontWeight','bold','LineWidth',style.axisLineWidth,'TickDir','out', ...
    'Layer','top','XGrid','on','YGrid','on','ZGrid','on');
ax.XLabel.FontSize = style.labelFontSize;
ax.YLabel.FontSize = style.labelFontSize;
ax.ZLabel.FontSize = style.labelFontSize;
lgd = legend(ax,[hBenchmark hComponents], ...
    {'Four-method J_{111} benchmark','GA objective-component sweeps'}, ...
    'Location','northoutside','Orientation','horizontal','NumColumns',2,'Box','off');
lgd.FontName = style.fontName;
lgd.FontSize = style.legendFontSize;
lgd.FontWeight = 'bold';

base = fullfile(char(outputDir),char(stem));
finalize_manuscript_figure(fig);
print(fig,[base '.eps'],'-depsc2','-painters','-r600','-loose');
exportgraphics(fig,[base '.png'],'Resolution',style.exportDpi);
close(fig);
end
