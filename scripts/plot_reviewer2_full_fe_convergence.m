function figureFiles = plot_reviewer2_full_fe_convergence(analysisDir,saveFigures)
%PLOT_REVIEWER2_FULL_FE_CONVERGENCE Preview convergence on every FE.
%
% Population-based optimizers retain their last legitimate incumbent between
% recorded generation/batch checkpoints. Bayesian optimization retains its
% native finer-resolution history. Values before an optimizer's first valid
% checkpoint remain undefined and are not backfilled.
%
% Usage:
%   plot_reviewer2_full_fe_convergence
%   plot_reviewer2_full_fe_convergence(analysisDir)
%   plot_reviewer2_full_fe_convergence(analysisDir,false)
%
% With no analysisDir, the newest FE_DATA_* directory under
% results/COMPARISON_PILOT_1200 is used. Saved figures overwrite the
% corresponding pilot_convergence_*.eps/png files in that analysis
% directory's paper_preview folder.

if nargin < 2 || isempty(saveFigures), saveFigures = true; end
validateattributes(saveFigures,{'logical','numeric'},{'scalar'});
saveFigures = logical(saveFigures);

paths = setup_project();
if nargin < 1 || isempty(analysisDir)
    pilotRoot = fullfile(paths.results,'COMPARISON_PILOT_1200');
    analysisDir = newest_analysis_directory(pilotRoot);
else
    analysisDir = char(string(analysisDir));
end
assert(isfolder(analysisDir),'Analysis directory does not exist: %s',analysisDir);

missions = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
optimizers = ["GA","PSO","BAYESIAN","ABC","ACO"];
files = dir(fullfile(analysisDir,'convergence_*.mat'));
assert(numel(files) >= numel(missions), ...
    'Expected convergence MAT files under %s.',analysisDir);

figureDir = fullfile(analysisDir,'paper_preview');
if saveFigures && ~isfolder(figureDir), mkdir(figureDir); end
figureFiles = strings(numel(missions),2);
colors = lines(numel(optimizers));

for m = 1:numel(missions)
    loaded = struct();
    found = false;
    for k = 1:numel(files)
        candidate = load(fullfile(files(k).folder,files(k).name), ...
            'comparison','curves');
        if string(candidate.comparison.settings.mission.type) == missions(m)
            loaded = candidate;
            found = true;
            break;
        end
    end
    assert(found,'No convergence data found for %s.',missions(m));

    fig = figure('Color','w','Units','inches','Position',[1 1 7.2 4.4], ...
        'PaperUnits','inches','PaperSize',[7.2 4.4], ...
        'PaperPosition',[0 0 7.2 4.4],'PaperPositionMode','manual', ...
        'Renderer','painters','InvertHardcopy','off');
    ax = axes(fig);
    hold(ax,'on');
    box(ax,'on');
    grid(ax,'on');

    curveOptimizers = upper(string({loaded.curves.optimizer}));
    handles = gobjects(numel(optimizers),1);
    firstFE = inf;
    finalFE = 0;

    for a = 1:numel(optimizers)
        idx = find(curveOptimizers == optimizers(a),1);
        assert(~isempty(idx),'Missing %s convergence curve.',optimizers(a));
        curve = loaded.curves(idx);

        fe = double(curve.fe(:));
        meanBest = double(curve.mean(:));
        valid = isfinite(meanBest);
        assert(any(valid),'No finite convergence values for %s.',optimizers(a));

        x = fe(valid);
        y = meanBest(valid);
        firstFE = min(firstFE,x(1));
        finalFE = max(finalFE,x(end));

        handles(a) = stairs(ax,x,y, ...
            'Color',colors(a,:),'LineWidth',2.0, ...
            'DisplayName',optimizers(a));

        % Sparse markers indicate the plotting grid without obscuring the
        % stair-step convergence trace on a 6000-FE study.
        markerStride = max(1,round(numel(x)/12));
        markerIdx = unique([1:markerStride:numel(x),numel(x)]);
        plot(ax,x(markerIdx),y(markerIdx),'o', ...
            'Color',colors(a,:),'MarkerFaceColor',colors(a,:), ...
            'MarkerSize',4.0,'HandleVisibility','off');
    end

    xlim(ax,[firstFE finalFE]);
    xlabel(ax,'Function evaluations','FontWeight','bold');
    ylabel(ax,'Mean best-so-far objective','FontWeight','bold');
    set(ax,'FontName','Times New Roman','FontSize',12, ...
        'FontWeight','bold','LineWidth',1.2,'Layer','top');
    ax.XLabel.FontSize = 14;
    ax.YLabel.FontSize = 14;

    lgd = legend(ax,handles,cellstr(optimizers), ...
        'Location','northoutside','Orientation','horizontal', ...
        'NumColumns',numel(optimizers),'Box','on');
    lgd.FontName = 'Times New Roman';
    lgd.FontSize = 11;
    lgd.FontWeight = 'bold';
    lgd.ItemTokenSize = [16 9];

    ax.Units = 'normalized';
    ax.PositionConstraint = 'outerposition';
    ax.OuterPosition = [0.04 0.06 0.92 0.82];
    drawnow;
    ax.LooseInset = max(ax.TightInset,0.025);

    if saveFigures
        stem = fullfile(figureDir,"pilot_convergence_"+mission_code(missions(m)));
        print(fig,char(stem+".eps"),'-depsc','-painters');
        exportgraphics(fig,char(stem+".png"),'Resolution',300);
        figureFiles(m,1) = stem+".eps";
        figureFiles(m,2) = stem+".png";
    end
end

fprintf('Full-FE convergence preview complete.\n');
if saveFigures
    fprintf('Figures saved under:\n%s\n',figureDir);
end
end

function analysisDir = newest_analysis_directory(root)
assert(isfolder(root),'Comparison pilot root does not exist: %s',root);
directories = dir(fullfile(root,'FE_DATA_*'));
directories = directories([directories.isdir]);
assert(~isempty(directories), ...
    'No FE_DATA analysis directory was found under %s.',root);
[~,idx] = max([directories.datenum]);
analysisDir = fullfile(directories(idx).folder,directories(idx).name);
end

function code = mission_code(mission)
switch string(mission)
    case "LUNAR_GATEWAY"
        code = "lg";
    case "LOW_THRUST_TRANSFER"
        code = "lt";
    case "GATEWAY_IMPULSE"
        code = "gi";
    otherwise
        code = lower(string(mission));
end
end
