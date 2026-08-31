%% process_fe_convergence.m
% Build Reviewer 2 convergence plots from the per-run optimization histories.
%
% Every successful SOO run is expected to contain:
%   <run>/data/optimization_history.csv
% with columns:
%   fe, bestJ
%
% The script groups runs by mission/configuration, aligns each stochastic run
% on a common function-evaluation axis using previous-value interpolation,
% and plots mean best-so-far objective with +/- 1 standard deviation.

clear; clc; close all;

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

comparisonRoot = fullfile(projectPaths.runs, 'COMPARISON');
if ~isfolder(comparisonRoot)
    comparisonRoot = projectPaths.runs;
end

outDir = fullfile(comparisonRoot, 'FE_CONVERGENCE_OUTPUT');
if ~exist(outDir,'dir'), mkdir(outDir); end

historyFiles = dir(fullfile(comparisonRoot, '**', 'optimization_history.csv'));
if isempty(historyFiles)
    error('No optimization_history.csv files found under: %s', comparisonRoot);
end

optimizerOrder = ["GA","PSO","BAYESIAN","ABC","ACO"];
records = table();

for k = 1:numel(historyFiles)
    filePath = fullfile(historyFiles(k).folder, historyFiles(k).name);
    runDir = fileparts(historyFiles(k).folder); % remove /data

    info = parse_run_path(runDir);
    if info.optimizer == ""
        warning('Skipping unrecognized run path: %s', runDir);
        continue;
    end

    try
        H = readtable(filePath, 'VariableNamingRule','preserve');
    catch ME
        warning('Could not read %s: %s', filePath, ME.message);
        continue;
    end

    if ~all(ismember({'fe','bestJ'}, H.Properties.VariableNames)) || isempty(H)
        warning('History missing fe/bestJ columns: %s', filePath);
        continue;
    end

    H = H(isfinite(H.fe) & isfinite(H.bestJ), :);
    if isempty(H)
        continue;
    end

    [feUnique, ia] = unique(double(H.fe(:)), 'stable');
    bestUnique = double(H.bestJ(ia));
    bestUnique = cummin(bestUnique);

    rec = table();
    rec.optimizer = info.optimizer;
    rec.measurement = info.measurement;
    rec.mission = info.mission;
    rec.num_observers = info.numObservers;
    rec.periods = info.periods;
    rec.screening = info.screening;
    rec.cost_combo = info.costCombo;
    rec.seed = info.seed;
    rec.max_fe = max(feUnique);
    rec.run_dir = string(runDir);
    rec.fe = {feUnique};
    rec.bestJ = {bestUnique};

    records = [records; rec]; %#ok<AGROW>
end

if isempty(records)
    error('No valid convergence histories were found.');
end

% Only compare groups that share the same mission/configuration.
groupVars = {'measurement','mission','num_observers','periods','screening','cost_combo'};
[G, groupTable] = findgroups(records(:,groupVars));

summaryRows = table();

for g = 1:height(groupTable)
    rows = records(G == g, :);

    availableOpts = optimizerOrder(ismember(optimizerOrder, unique(rows.optimizer)));
    if numel(availableOpts) < 2
        continue;
    end

    commonBudget = min(rows.max_fe);
    if ~isfinite(commonBudget) || commonBudget < 1
        continue;
    end

    feGrid = (1:commonBudget)';

    fig = figure('Visible','off', 'Color','w', ...
        'Units','inches', 'Position',[1 1 9.5 6.2]);
    ax = axes(fig);
    hold(ax,'on'); grid(ax,'on'); box(ax,'on');

    colorOrder = ax.ColorOrder;

    for oi = 1:numel(availableOpts)
        opt = availableOpts(oi);
        optRows = rows(rows.optimizer == opt, :);

        traces = nan(commonBudget, height(optRows));

        for r = 1:height(optRows)
            fe = optRows.fe{r};
            bestJ = optRows.bestJ{r};

            % Ensure the first known best value is extended back to FE=1.
            if fe(1) > 1
                fe = [1; fe];
                bestJ = [bestJ(1); bestJ];
            end

            traces(:,r) = interp1(fe, bestJ, feGrid, 'previous', 'extrap');
        end

        meanTrace = mean(traces, 2, 'omitnan');
        stdTrace = std(traces, 0, 2, 'omitnan');

        c = colorOrder(mod(oi-1,size(colorOrder,1))+1,:);
        xPatch = [feGrid; flipud(feGrid)];
        yPatch = [meanTrace-stdTrace; flipud(meanTrace+stdTrace)];

        fill(ax, xPatch, yPatch, c, ...
            'FaceAlpha',0.12, 'EdgeColor','none', ...
            'HandleVisibility','off');

        plot(ax, feGrid, meanTrace, ...
            'LineWidth',2.4, 'Color',c, 'DisplayName',char(opt));

        s = table();
        s.optimizer = opt;
        s.measurement = groupTable.measurement(g);
        s.mission = groupTable.mission(g);
        s.num_observers = groupTable.num_observers(g);
        s.periods = groupTable.periods(g);
        s.screening = groupTable.screening(g);
        s.cost_combo = groupTable.cost_combo(g);
        s.n_runs = height(optRows);
        s.fe_budget = commonBudget;
        s.final_best_mean = meanTrace(end);
        s.final_best_std = stdTrace(end);
        summaryRows = [summaryRows; s]; %#ok<AGROW>
    end

    xlabel(ax,'Function Evaluations');
    ylabel(ax,'Best Objective Value');
    legend(ax,'Location','best');

    set(ax, ...
        'FontName','Times New Roman', ...
        'FontSize',18, ...
        'FontWeight','bold', ...
        'LineWidth',1.2);

    tag = sprintf('%s_%s_o%d_p%d_s%d_%s', ...
        char(groupTable.mission(g)), ...
        char(groupTable.measurement(g)), ...
        groupTable.num_observers(g), ...
        groupTable.periods(g), ...
        groupTable.screening(g), ...
        char(lower(groupTable.cost_combo(g))));

    tag = regexprep(tag,'[^A-Za-z0-9_\-]','_');

    exportgraphics(fig, fullfile(outDir, ['convergence_' tag '.eps']), ...
        'ContentType','vector');
    exportgraphics(fig, fullfile(outDir, ['convergence_' tag '.png']), ...
        'Resolution',300);
    close(fig);
end

if ~isempty(summaryRows)
    writetable(summaryRows, fullfile(outDir,'FE_Convergence_Summary.csv'));
end

fprintf('FE convergence outputs saved to:\n%s\n', outDir);


function info = parse_run_path(runDir)
    info = struct( ...
        'optimizer',"", ...
        'measurement',"", ...
        'mission',"", ...
        'numObservers',NaN, ...
        'periods',1, ...
        'screening',NaN, ...
        'costCombo',"", ...
        'seed',NaN);

    p = string(runDir);
    parts = split(p, filesep);

    idxAlg = find(startsWith(upper(parts),'RUNS_'),1,'last');
    if ~isempty(idxAlg)
        info.optimizer = erase(upper(parts(idxAlg)), 'RUNS_');
    end

    if idxAlg + 1 <= numel(parts)
        info.measurement = lower(parts(idxAlg+1));
    end
    if idxAlg + 2 <= numel(parts)
        info.mission = lower(parts(idxAlg+2));
    end

    runName = parts(end);

    tok = regexp(runName, '_o(\d+)', 'tokens','once');
    if ~isempty(tok), info.numObservers = str2double(tok{1}); end

    tok = regexp(runName, '_p(\d+)', 'tokens','once');
    if ~isempty(tok), info.periods = str2double(tok{1}); end

    tok = regexp(runName, '_s([01])', 'tokens','once');
    if ~isempty(tok), info.screening = str2double(tok{1}); end

    tok = regexp(runName, '_j([01]{3})', 'tokens','once');
    if ~isempty(tok), info.costCombo = "J" + string(tok{1}); end

    tok = regexp(runName, '_seed(\d+)', 'tokens','once');
    if ~isempty(tok), info.seed = str2double(tok{1}); end
end
