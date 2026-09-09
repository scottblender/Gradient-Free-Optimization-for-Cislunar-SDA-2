function test_reviewer2_legend_configuration()
%TEST_REVIEWER2_LEGEND_CONFIGURATION Audit legend coverage in final figures.

projectDir = fileparts(fileparts(mfilename('fullpath')));
runtimeText = string(fileread(fullfile(projectDir,'scripts','make_reviewer2_runtime_figures.m')));
curatedText = string(fileread(fullfile(projectDir,'scripts','make_reviewer2_curated_figures.m')));
geometryText = string(fileread(fullfile(projectDir,'scripts','plot_reviewer2_geometry_grid.m')));
mcText = string(fileread(fullfile(projectDir,'scripts','plot_reviewer2_baseline_monte_carlo.m')));

% Focused runtime study: optimizer identity is explicit on the x-axis for
% flat-colored bar charts. Only the baseline reference needs a bar-chart
% legend; convergence must retain one legend entry per optimizer curve.
assert(contains(runtimeText,'Optimizers are identified by the x-axis labels'), ...
    'Runtime bar-chart legend convention is undocumented.');
assert(contains(runtimeText,'legend(ax,hBase,{''Baseline AO''}'), ...
    'Runtime objective figure is missing the Baseline AO legend.');
assert(contains(runtimeText,'legend(ax,handles,''Location'',''northoutside'''), ...
    'Runtime convergence figure is missing its optimizer legend.');
assert(~contains(runtimeText,'legendHandles = b') && ...
    ~contains(runtimeText,'legendLabels = optimizer_labels'), ...
    'Runtime bar charts must not pair one Bar handle with multiple labels.');

% Main curated figures: each visual encoding with multiple series must have
% an explicit legend. These checks cover comparison, baseline, screening,
% convergence, and five-family selection figures.
requiredCurated = string({ ...
    'legend(ax,legendHandles,cellstr(legendLabels)', ...
    'legend(ax,handles,{''AO'',''AR''}', ...
    'legend(ax,handles,''Location'',''northoutside''', ...
    'legend(ax,b,{''Screening ON'',''Screening OFF''}', ...
    'legend(ax,b,cellstr(families)', ...
    'legend(ax,handles,''Location'',''northoutside'',''Orientation'',''horizontal'''});
assert_tokens(curatedText,requiredCurated,'Curated figure renderer');

% Representative 3-D geometry and Monte Carlo validation use their own
% renderers and must also identify every non-obvious visual encoding.
assert(contains(geometryText,'legend(ax,legendHandles,cellstr(legendLabels)'), ...
    'Trajectory figures are missing their reference legend.');
assert(contains(geometryText,'center_reference_legend'), ...
    'Trajectory legend is not using the centered publication layout.');
assert(contains(mcText, ...
    'legend(ax,[hBox hRef],{''MC samples'',''Optimized reference''}'), ...
    'Monte Carlo plots are missing the MC/reference legend.');

fprintf('Reviewer 2 legend configuration passed.\n');
end


function assert_tokens(textValue,tokens,label)
for token = tokens
    assert(contains(textValue,token), ...
        '%s is missing required legend token: %s',label,token);
end
end
