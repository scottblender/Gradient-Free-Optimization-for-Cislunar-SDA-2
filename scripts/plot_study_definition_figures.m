function outputs = plot_study_definition_figures(inspectFigures)
% Generate catalog, slot-definition, geometry, and target-case figures.

if nargin<1 || isempty(inspectFigures), inspectFigures = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
setup_project();

fprintf('\n--- Study-definition figures ---\n');

fprintf('\n1/5 Orbit-family catalog figures\n');
outputs.catalog = create_orbit_catalog_figures(inspectFigures);

fprintf('\n2/5 Equal-time slot definition\n');
outputs.slots = create_slot_definition(inspectFigures);

fprintf('\n3/5 Unified visibility / keepout geometry\n');
outputs.visibilityGeometry = create_visibility_keepout_figure(inspectFigures);

fprintf('\n4/5 RA/Dec measurement geometry\n');
outputs.measurementGeometry = create_measurement_model_figure(inspectFigures);

fprintf('\n5/5 Tracking cases\n');
fprintf('The low-thrust panel solves the transfer and can take several minutes.\n');
outputs.cases = create_tracking_cases(inspectFigures);

fprintf('\nAll study-definition figures were generated.\n');
end



function outputs = create_orbit_catalog_figures(inspectFigure)
% Plot the selected orbit families in the original study style.
% The accompanying CSV files provide the quantitative family table data.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results, 'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

catalog = load(projectPaths.catalog, 'T');
T = catalog.T;

required = {'orbitFamily','state','Period (TU) ','Stability index  '};
assert(all(ismember(required,T.Properties.VariableNames)), ...
    'The orbit catalog is missing required plotting variables.');

LU = 384400;       % km
TU = 375695;       % s
Rmoon = 1737.1;    % km; retained for consistency with the CR3BP study
mu = 1.215058560962404E-2;
xMoon = 1-mu;

familyOrder = [ ...
    "NHL1","NHL2","SHL1","SHL2", ...
    "NNRHL1","NNRHL2","SNRHL1","SNRHL2","DRO"];

family = string(T.orbitFamily);
assert(all(ismember(family,familyOrder)), ...
    'The catalog contains an unexpected orbit family.');

orbitID = strings(height(T),1);
if ismember('orbitID',T.Properties.VariableNames)
    orbitID = string(T.orbitID);
end

nOrbit = height(T);
periluneAltitude_km = zeros(nOrbit,1);
apoluneAltitude_km = zeros(nOrbit,1);
inPlaneAmplitude_km = zeros(nOrbit,1);
outOfPlaneAmplitude_km = zeros(nOrbit,1);

for k = 1:nOrbit
    state = T.state{k};
    position = state(:,1:3);
    moonRelative = position-[xMoon,0,0];
    moonDistance_km = vecnorm(moonRelative,2,2)*LU;

    periluneAltitude_km(k) = min(moonDistance_km)-Rmoon;
    apoluneAltitude_km(k) = max(moonDistance_km)-Rmoon;

    % Reproducible geometric amplitudes based on the rotating-frame
    % bounding box. A_xy is half the diagonal of the x-y extent.
    dx = max(position(:,1))-min(position(:,1));
    dy = max(position(:,2))-min(position(:,2));
    dz = max(position(:,3))-min(position(:,3));
    inPlaneAmplitude_km(k) = 0.5*hypot(dx,dy)*LU;
    outOfPlaneAmplitude_km(k) = 0.5*dz*LU;
end

period_TU = T.('Period (TU) ');
period_days = period_TU*TU/86400;
stabilityIndex = T.('Stability index  ');

orbitMetrics = table( ...
    orbitID,family,periluneAltitude_km,apoluneAltitude_km,period_TU, ...
    period_days,stabilityIndex,inPlaneAmplitude_km, ...
    outOfPlaneAmplitude_km);

nFamily = numel(familyOrder);
count = zeros(nFamily,1);
periluneMin_km = zeros(nFamily,1);
periluneMax_km = zeros(nFamily,1);
apoluneMin_km = zeros(nFamily,1);
apoluneMax_km = zeros(nFamily,1);
periodMin_TU = zeros(nFamily,1);
periodMax_TU = zeros(nFamily,1);
periodMin_days = zeros(nFamily,1);
periodMax_days = zeros(nFamily,1);
stabilityMin = zeros(nFamily,1);
stabilityMax = zeros(nFamily,1);
inPlaneMin_km = zeros(nFamily,1);
inPlaneMax_km = zeros(nFamily,1);
outOfPlaneMin_km = zeros(nFamily,1);
outOfPlaneMax_km = zeros(nFamily,1);

lagrangePoint = ["L1";"L2";"L1";"L2"; ...
    "L1";"L2";"L1";"L2";"Moon-centered"];

for k = 1:nFamily
    use = family==familyOrder(k);
    count(k) = nnz(use);
    assert(count(k)>0,'No catalog entries found for %s.',familyOrder(k));

    periluneMin_km(k) = min(periluneAltitude_km(use));
    periluneMax_km(k) = max(periluneAltitude_km(use));
    apoluneMin_km(k) = min(apoluneAltitude_km(use));
    apoluneMax_km(k) = max(apoluneAltitude_km(use));
    periodMin_TU(k) = min(period_TU(use));
    periodMax_TU(k) = max(period_TU(use));
    periodMin_days(k) = min(period_days(use));
    periodMax_days(k) = max(period_days(use));
    stabilityMin(k) = min(stabilityIndex(use));
    stabilityMax(k) = max(stabilityIndex(use));
    inPlaneMin_km(k) = min(inPlaneAmplitude_km(use));
    inPlaneMax_km(k) = max(inPlaneAmplitude_km(use));
    outOfPlaneMin_km(k) = min(outOfPlaneAmplitude_km(use));
    outOfPlaneMax_km(k) = max(outOfPlaneAmplitude_km(use));
end

Family = familyOrder(:);
familySummary = table( ...
    Family,lagrangePoint,count, ...
    periluneMin_km,periluneMax_km,apoluneMin_km,apoluneMax_km, ...
    periodMin_TU,periodMax_TU,periodMin_days,periodMax_days, ...
    stabilityMin,stabilityMax,inPlaneMin_km,inPlaneMax_km, ...
    outOfPlaneMin_km,outOfPlaneMax_km);

summaryFile = fullfile(outputDir,'orbit_family_summary.csv');
writetable(familySummary,summaryFile);

metricFile = fullfile(outputDir,'orbit_catalog_metrics.csv');
writetable(orbitMetrics,metricFile);


% Preserve the original family-trajectory presentation while using the
% rebuilt catalog directly. Each legend uses the same position and layout.
familyGroups = { ...
    ["NHL1","NHL2"], ...
    ["SHL1","SHL2"], ...
    ["NNRHL1","NNRHL2"], ...
    ["SNRHL1","SNRHL2"], ...
    ["DRO"]};
figureNames = [ ...
    "northern_halo", ...
    "southern_halo", ...
    "northern_rectilinear", ...
    "southern_rectilinear", ...
    "dro_family"];

[xL1,xL2] = cr3bp_L1L2(mu);

cL1 = [0.05,0.32,0.82];
cL2 = [0.90,0.16,0.12];
cMoon = [0.70,0.70,0.70];
cPoint = [0.85,0.85,0.85];

numPairOrbitsPerFamily = 16;
numDroOrbits = 16;
maxPointsPerOrbit = 300;
figureFiles = strings(numel(familyGroups),1);

for groupIndex = 1:numel(familyGroups)
    fig = publication_figure(6.5,5.4);
    ax = axes(fig);
    hold(ax,'on');
    box(ax,'on');
    grid(ax,'off');
    axis(ax,'equal');
    set(ax,'TickLabelInterpreter','tex','Layer','top');

    group = familyGroups{groupIndex};

    if numel(group)==2
        ax.Projection = 'perspective';
        view(ax,-37.5,30);

        familyHandles = gobjects(2,1);
        familyLabels = strings(2,1);
        colors = [cL1;cL2];
        perFamily = numPairOrbitsPerFamily;

        for member = 1:2
            use = family==group(member);
            familyRows = find(use);
            assert(~isempty(familyRows), ...
                'No selected orbits found for %s.',group(member));

            assert(numel(familyRows)==50, ...
                'Expected 50 selected orbits for %s, found %d.', ...
                group(member),numel(familyRows));
            selectedIndex = unique(round(linspace( ...
                1,numel(familyRows),min(perFamily,numel(familyRows)))));
            selectedRows = familyRows(selectedIndex);

            for plotted = 1:numel(selectedRows)
                state = T.state{selectedRows(plotted)};
                step = max(1,round(size(state,1)/maxPointsPerOrbit));
                handle = plot3(ax,state(1:step:end,1), ...
                    state(1:step:end,2),state(1:step:end,3),'-', ...
                    'Color',colors(member,:),'LineWidth',0.85);
                if plotted==1
                    familyHandles(member) = handle;
                end
            end
            familyLabels(member) = "L"+string(member);
        end

        moonHandle = plot3(ax,1-mu,0,0,'o', ...
            'MarkerSize',6, ...
            'MarkerFaceColor',cMoon, ...
            'MarkerEdgeColor',[0.45,0.45,0.45], ...
            'LineWidth',0.9);
        l1PointHandle = plot3(ax,xL1,0,0,'^', ...
            'MarkerSize',7, ...
            'MarkerFaceColor',cPoint, ...
            'MarkerEdgeColor',[0.55,0.55,0.55], ...
            'LineWidth',0.9);
        l2PointHandle = plot3(ax,xL2,0,0,'v', ...
            'MarkerSize',7, ...
            'MarkerFaceColor',cPoint, ...
            'MarkerEdgeColor',[0.55,0.55,0.55], ...
            'LineWidth',0.9);

        legendHandles = [ ...
            familyHandles;moonHandle;l1PointHandle;l2PointHandle];
        legendLabels = [ ...
            familyLabels;"Moon";"L1 point";"L2 point"];
        zlabel(ax,'z (LU)');
        axis(ax,'tight');
        axis(ax,'vis3d');
    else
        ax.Projection = 'orthographic';
        view(ax,2);

        familyRows = find(family=="DRO");
        assert(~isempty(familyRows),'No selected DROs were found.');
        assert(numel(familyRows)==50, ...
            'Expected 50 selected DROs, found %d.',numel(familyRows));
        selectedRows = unique(round(linspace( ...
            1,numel(familyRows),min(numDroOrbits,numel(familyRows)))));

        droHandle = gobjects(1);
        allX = [];
        allY = [];
        for plotted = 1:numel(selectedRows)
            state = T.state{familyRows(selectedRows(plotted))};
            step = max(1,round(size(state,1)/maxPointsPerOrbit));
            statePlot = state(1:step:end,:);
            handle = plot(ax,statePlot(:,1),statePlot(:,2),'-', ...
                'Color',cL1,'LineWidth',1.45);
            allX = [allX;statePlot(:,1)]; %#ok<AGROW>
            allY = [allY;statePlot(:,2)]; %#ok<AGROW>
            if plotted==1
                droHandle = handle;
            end
        end

        moonRadius = 1737.1/LU;
        angle = linspace(0,2*pi,200);
        moonHandle = fill(ax,(1-mu)+moonRadius*cos(angle), ...
            moonRadius*sin(angle),cMoon,'EdgeColor','none');

        legendHandles = [droHandle;moonHandle];
        legendLabels = ["DRO";"Moon"];
        zlabel(ax,'');

        xData = [allX;(1-mu)+moonRadius*cos(angle(:))];
        yData = [allY;moonRadius*sin(angle(:))];
        xPad = 0.05*max(max(xData)-min(xData),eps);
        yPad = 0.05*max(max(yData)-min(yData),eps);
        xlim(ax,[min(xData)-xPad,max(xData)+xPad]);
        ylim(ax,[min(yData)-yPad,max(yData)+yPad]);
    end

    xlabel(ax,'x (LU)');
    ylabel(ax,'y (LU)');
    format_publication_axes(ax,13);

    legendHandle = legend(ax,legendHandles,cellstr(legendLabels), ...
        'Location','northeast','Orientation','vertical');
    legendHandle.Box = 'on';
    legendHandle.FontName = 'Times New Roman';
    legendHandle.FontSize = 12;
    legendHandle.FontWeight = 'bold';
    legendHandle.ItemTokenSize = [14,8];
    legendHandle.NumColumns = 1;

    ax.Units = 'normalized';
    ax.Position = [0.12,0.13,0.80,0.80];
    ax.LooseInset = max(ax.TightInset,0.015);

    figureFiles(groupIndex) = fullfile( ...
        outputDir,figureNames(groupIndex)+".eps");
    inspect_before_export(fig,inspectFigure, ...
        figureNames(groupIndex)+" orbit-family");
    export_publication_eps(fig,figureFiles(groupIndex));
    close(fig);
end

outputs = struct();
outputs.figures = figureFiles;
outputs.familySummary = string(summaryFile);
outputs.orbitMetrics = string(metricFile);
outputs.numOrbits = nOrbit;
outputs.numFamilies = nFamily;

fprintf('Saved orbit-family figures and catalog tables to:\n  %s\n',outputDir);
end

function outputs = create_slot_definition(inspectFigure)
% Export two complementary illustrations of the exact 50-slot convention.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

catalog = load(projectPaths.catalog,'T');
T = catalog.T;

family = string(T.orbitFamily);
orbitIndex = find(family=="NNRHL1",1,'first');
assert(~isempty(orbitIndex), ...
    'No representative northern NRHO L1 orbit was found.');

periodAll = T.('Period (TU) ');
period = periodAll(orbitIndex);
rawTime = T.time{orbitIndex};
rawState = T.state{orbitIndex};

[uniqueTime,uniqueIndex] = unique(rawTime);
uniqueState = rawState(uniqueIndex,:);
interpolant = griddedInterpolant(uniqueTime,uniqueState,'spline');

numSlots = 50;
deltaTime = period/numSlots;
slotNumber = (1:numSlots).';
slotTime = (slotNumber-1)*deltaTime;
slotState = interpolant(slotTime);

assert(slotTime(1)==0,'The first slot must occur at t=0.');
assert(abs(slotTime(end)-49*period/50)<=10*eps(period), ...
    'The last slot must occur at 49T/50.');
assert(all(diff(slotTime)>0),'Slot epochs are not strictly increasing.');
assert(all(slotTime<period),'The periodic endpoint must not be stored.');

nextPosition = [slotState(2:end,1:3);slotState(1,1:3)];
adjacentChord_km = vecnorm( ...
    nextPosition-slotState(:,1:3),2,2)*384400;

selectedSlot = 17;
nextSlot = selectedSlot+1;
selectedColor = [0.85,0.25,0.20];
nextColor = [0.20,0.50,0.80];
orbitColor = [0.27,0.31,0.86];
neutralColor = [0.25,0.25,0.25];
mu = 1.215058560962404E-2;
LU = 384400;

% Figure 1: the orbit and its 50 equal-time candidate states.
figGeometry = publication_figure(6.5,6.5);
ax = axes(figGeometry);
prepare_axes(ax);

plotStep = max(1,round(size(rawState,1)/500));
hOrbit = plot3(ax,rawState(1:plotStep:end,1), ...
    rawState(1:plotStep:end,2),rawState(1:plotStep:end,3), ...
    '-','Color',orbitColor,'LineWidth',2.5);
hSlots = plot3(ax,slotState(:,1),slotState(:,2),slotState(:,3), ...
    'o','MarkerSize',5,'MarkerFaceColor','w', ...
    'MarkerEdgeColor',neutralColor,'LineWidth',1.0);
hSelected = plot3(ax,slotState(selectedSlot,1), ...
    slotState(selectedSlot,2),slotState(selectedSlot,3), ...
    'o','MarkerSize',9,'MarkerFaceColor',selectedColor, ...
    'MarkerEdgeColor','k','LineWidth',1.2);
hNext = plot3(ax,slotState(nextSlot,1), ...
    slotState(nextSlot,2),slotState(nextSlot,3), ...
    's','MarkerSize',9,'MarkerFaceColor',nextColor, ...
    'MarkerEdgeColor','k','LineWidth',1.2);
hMoon = draw_moon(ax,mu,LU);

set(ax,'FontName','Times New Roman','FontSize',18, ...
    'FontWeight','bold','LineWidth',1.8);
legendHandle = legend(ax,[hOrbit,hSlots,hSelected,hNext,hMoon], ...
    {'Orbit','Candidate slots','Slot j','Slot j+1','Moon'}, ...
    'Location','northoutside','Orientation','horizontal', ...
    'FontName','Times New Roman','FontSize',15,'FontWeight','bold');
legendHandle.NumColumns = 3;
legendHandle.Box = 'on';
place_legend_above(legendHandle,3,13);
axis(ax,'tight');
axis(ax,'vis3d');
ax.Position = [0.13,0.14,0.74,0.66];

geometryFile = fullfile(outputDir,'slot_geometry_equal_time.eps');
inspect_before_export(figGeometry,inspectFigure, ...
    'equal-time slot geometry');
export_publication_eps(figGeometry,geometryFile);
close(figGeometry);

% Figure 2: the exact normalized phase grid and excluded endpoint.
figPhase = publication_figure(7.2,3.8);
ax = axes(figPhase);
hold(ax,'on');
box(ax,'on');

phase = slotTime/period;
plot(ax,[0,1],[0,0],'-','Color',0.65*[1,1,1], ...
    'LineWidth',1.5);
hCandidate = scatter(ax,phase,zeros(size(phase)),32,'w','filled', ...
    'MarkerEdgeColor',neutralColor,'LineWidth',0.9);
hSelectedPhase = scatter(ax,phase(selectedSlot),0,90, ...
    selectedColor,'filled','MarkerEdgeColor','k','LineWidth',1.1);
hNextPhase = scatter(ax,phase(nextSlot),0,90,nextColor,'s','filled', ...
    'MarkerEdgeColor','k','LineWidth',1.1);
hEndpoint = plot(ax,1,0,'o','MarkerSize',9,'MarkerFaceColor','w', ...
    'MarkerEdgeColor',[0.75,0.20,0.20],'LineWidth',1.8);

plot(ax,phase([selectedSlot,nextSlot]),[0.16,0.16], ...
    '-k','LineWidth',1.5);
plot(ax,phase([selectedSlot,selectedSlot]),[0,0.16],':k');
plot(ax,phase([nextSlot,nextSlot]),[0,0.16],':k');
text(ax,mean(phase([selectedSlot,nextSlot])),0.20, ...
    '\Delta t/T=1/50','HorizontalAlignment','center', ...
    'FontName','Times New Roman','FontSize',16,'FontWeight','bold');
text(ax,0.99,-0.025,{'t=T','not stored'}, ...
    'HorizontalAlignment','right','VerticalAlignment','top', ...
    'FontName','Times New Roman','FontSize',15,'FontWeight','bold');

xlabel(ax,'Normalized epoch, t/T');
yticks(ax,[]);
ylim(ax,[-0.18,0.30]);
xlim(ax,[-0.02,1.02]);
set(ax,'FontName','Times New Roman','FontSize',18, ...
    'FontWeight','bold','LineWidth',1.8,'TickLabelInterpreter','tex');

legendHandle = legend(ax, ...
    [hCandidate,hSelectedPhase,hNextPhase,hEndpoint], ...
    {'Candidate slots','Slot j','Slot j+1','Excluded endpoint'}, ...
    'Location','northoutside','Orientation','horizontal', ...
    'FontName','Times New Roman','FontSize',14,'FontWeight','bold');
legendHandle.Box = 'on';
place_legend_above(legendHandle,2,13);
ax.Position = [0.12,0.19,0.80,0.56];

phaseFile = fullfile(outputDir,'slot_phase_grid.eps');
inspect_before_export(figPhase,inspectFigure, ...
    'endpoint-excluded phase grid');
export_publication_eps(figPhase,phaseFile);
close(figPhase);

orbitID = "";
if ismember('orbitID',T.Properties.VariableNames)
    orbitID = string(T.orbitID(orbitIndex));
end

slotSummary = table( ...
    orbitIndex,orbitID,family(orbitIndex),numSlots,period,deltaTime, ...
    min(adjacentChord_km),median(adjacentChord_km),max(adjacentChord_km), ...
    'VariableNames',{'catalogRow','orbitID','family','numSlots', ...
    'period_TU','deltaTime_TU','minimumChord_km', ...
    'medianChord_km','maximumChord_km'});

summaryFile = fullfile(outputDir,'slot_definition_summary.csv');
writetable(slotSummary,summaryFile);

outputs = struct();
outputs.figures = [string(geometryFile);string(phaseFile)];
outputs.geometryFigure = string(geometryFile);
outputs.phaseFigure = string(phaseFile);
outputs.summary = string(summaryFile);
outputs.slotSummary = slotSummary;

fprintf('Saved the two slot-definition figures to:\n  %s\n',outputDir);
end



function outputs = create_visibility_keepout_figure(inspectFigure)
% Illustrate the unified minimum-angular-separation visibility geometry.
% The schematic is body-generic and intentionally contains only the
% geometry needed to explain physical occultation and sensor keepout.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

fig = publication_figure(7.2,5.1);

cObserver = [0.90,0.12,0.10];
cTarget = [0.00,0.39,0.72];
cBody = [0.70,0.70,0.70];
cOcc = [0.38,0.38,0.38];
cExclusion = [0.92,0.55,0.05];
cLos = [0.05,0.05,0.05];
cOccShade = [0.82,0.82,0.82];
cExclusionShade = [0.98,0.89,0.62];

observer = [-2.20,0.00];
body = [1.05,0.00];
bodyRadius = 0.66;
bodyRange = norm(body-observer);
thetaOcc = asin(bodyRadius/bodyRange);

% Choose a representative center-referenced exclusion threshold that
% exceeds the physical limb angle so the effective margin is visible.
thetaExclusion = thetaOcc + deg2rad(8);
thetaKeepout = max(thetaOcc,thetaExclusion);
thetaB = thetaKeepout + deg2rad(13);
targetRange = 4.75;
target = observer + targetRange*[cos(thetaB),sin(thetaB)];

ax = axes(fig,'Units','normalized','Position',[0.05,0.06,0.90,0.88]);
hold(ax,'on');
axis(ax,'equal');
axis(ax,'off');

bodyAngle = linspace(0,2*pi,240);
sectorRadius = 3.45;

% Physical occultation sector.
occAngles = linspace(-thetaOcc,thetaOcc,220);
patch(ax,[observer(1),observer(1)+sectorRadius*cos(occAngles),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(occAngles),observer(2)], ...
    cOccShade,'EdgeColor','none','HandleVisibility','off');

% Effective exclusion-margin bands between the physical limb and the
% configured keep-out boundary. Plot both sides of the exclusion cone.
upperMargin = linspace(thetaOcc,thetaKeepout,120);
patch(ax,[observer(1),observer(1)+sectorRadius*cos(upperMargin),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(upperMargin),observer(2)], ...
    cExclusionShade,'EdgeColor','none','HandleVisibility','off');
lowerMargin = linspace(-thetaKeepout,-thetaOcc,120);
patch(ax,[observer(1),observer(1)+sectorRadius*cos(lowerMargin),observer(1)], ...
    [observer(2),observer(2)+sectorRadius*sin(lowerMargin),observer(2)], ...
    cExclusionShade,'EdgeColor','none','HandleVisibility','off');

% Body and reference directions.
fill(ax,body(1)+bodyRadius*cos(bodyAngle), ...
    body(2)+bodyRadius*sin(bodyAngle),cBody, ...
    'EdgeColor','k','LineWidth',1.2);
plot(ax,[observer(1),body(1)],[observer(2),body(2)],'--', ...
    'Color',cOcc,'LineWidth',1.3);

occEnd = observer + 3.65*[cos(thetaOcc),sin(thetaOcc)];
keepEnd = observer + 3.65*[cos(thetaKeepout),sin(thetaKeepout)];
plot(ax,[observer(1),occEnd(1)],[observer(2),occEnd(2)],':', ...
    'Color',cOcc,'LineWidth',1.9);
plot(ax,[observer(1),keepEnd(1)],[observer(2),keepEnd(2)],'--', ...
    'Color',cExclusion,'LineWidth',2.1);
plot(ax,[observer(1),target(1)],[observer(2),target(2)],'-', ...
    'Color',cLos,'LineWidth',2.5);

plot(ax,observer(1),observer(2),'o','MarkerSize',11, ...
    'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax,target(1),target(2),'o','MarkerSize',11, ...
    'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

% Angle arcs.
draw_angle_arc_2d(ax,observer,0,thetaOcc,0.95,cOcc,1.9);
draw_angle_arc_2d(ax,observer,0,thetaKeepout,1.48,cExclusion,2.1);
draw_angle_arc_2d(ax,observer,0,thetaB,2.10,cTarget,2.1);

% Object labels.
text(ax,observer(1)-0.02,observer(2)-0.46,'Observer', ...
    'Color',cObserver,'FontWeight','bold','FontSize',17, ...
    'HorizontalAlignment','center');
text(ax,target(1)+0.18,target(2)-0.02,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',17, ...
    'HorizontalAlignment','left');
text(ax,body(1),body(2)-1.16,'Body b', ...
    'FontWeight','bold','FontSize',17, ...
    'HorizontalAlignment','center','BackgroundColor','w', ...
    'Margin',0.8);

% Keep the angular labels outside the shaded geometry. Each leader points
% to the middle of the arc represented by that label.
occArcPoint = observer + ...
    0.95*[cos(0.55*thetaOcc),sin(0.55*thetaOcc)];
keepoutArcPoint = observer + ...
    1.48*[cos(0.62*thetaKeepout),sin(0.62*thetaKeepout)];
targetArcPoint = observer + ...
    2.10*[cos(0.72*thetaB),sin(0.72*thetaB)];

occLabel = observer + [1.05,-0.62];
keepoutLabel = observer + [1.15,1.10];
targetAngleLabel = observer + [2.35,1.18];

draw_text_callout(ax,occLabel,occArcPoint, ...
    '\theta_{occ,b}',cOcc,16);
draw_text_callout(ax,keepoutLabel,keepoutArcPoint, ...
    '\theta_{keepout,b}',cExclusion,16);
draw_text_callout(ax,targetAngleLabel,targetArcPoint, ...
    '\theta_b',cTarget,16);

ptOccRegion = observer + ...
    2.05*[cos(-0.25*thetaOcc),sin(-0.25*thetaOcc)];
occCallout = observer + [1.65,-1.28];
text(ax,occCallout(1),occCallout(2), ...
    {'physical';'occultation'}, ...
    'Color',cOcc,'FontSize',15,'FontAngle','italic', ...
    'FontWeight','bold','HorizontalAlignment','center', ...
    'VerticalAlignment','middle');
occArrowStart = occCallout + [0,0.22];
quiver(ax,occArrowStart(1),occArrowStart(2), ...
    ptOccRegion(1)-occArrowStart(1), ...
    ptOccRegion(2)-occArrowStart(2),0, ...
    'Color',cOcc,'LineWidth',1.3,'MaxHeadSize',0.18);

ptMargin = observer + ...
    2.55*[cos(0.5*(thetaOcc+thetaKeepout)), ...
    sin(0.5*(thetaOcc+thetaKeepout))];
marginCallout = observer + [2.92,2.08];
text(ax,marginCallout(1),marginCallout(2), ...
    {'effective';'exclusion margin'}, ...
    'Color',cExclusion,'FontSize',15,'FontAngle','italic', ...
    'FontWeight','bold','HorizontalAlignment','center', ...
    'VerticalAlignment','middle','BackgroundColor','w', ...
    'Margin',0.8);
marginArrowStart = marginCallout + [0,-0.22];
quiver(ax,marginArrowStart(1),marginArrowStart(2), ...
    ptMargin(1)-marginArrowStart(1), ...
    ptMargin(2)-marginArrowStart(2),0, ...
    'Color',cExclusion,'LineWidth',1.3,'MaxHeadSize',0.18);

xlim(ax,[-3.20,3.25]);
ylim(ax,[-2.10,3.50]);
set(findall(fig,'Type','text'),'FontName','Times New Roman');

figureFile = fullfile(outputDir,'visibility_keepout_geometry.eps');
inspect_before_export(fig,inspectFigure,'unified visibility / keepout geometry');
export_publication_eps(fig,figureFile);
close(fig);

outputs = struct();
outputs.figure = string(figureFile);
outputs.thetaExclusion_deg = struct('earth',15,'moon',10,'sun',20);
fprintf('Saved visibility / keepout geometry to:\n  %s\n',figureFile);
end


function outputs = create_measurement_model_figure(inspectFigure)
% Illustrate the implemented angles-only RA/Dec relative LOS geometry.
% Visibility and keepout constraints intentionally do not appear here.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

cObserver = [0.90,0.12,0.10];
cTarget = [0.00,0.39,0.72];
cProjection = [0.42,0.42,0.42];
cAngle = [0.88,0.43,0.08];

rho = [3.15,2.10,1.75];
rhoXY = hypot(rho(1),rho(2));
alpha = atan2(rho(2),rho(1));
delta = asin(rho(3)/norm(rho));

% ---------------- (a) Right ascension ----------------
figRa = publication_figure(4.45,4.55);
ax1 = axes(figRa,'Units','normalized','Position',[0.10,0.12,0.82,0.80]);
hold(ax1,'on');
axis(ax1,'equal');
axis(ax1,'off');

projection = rho(1:2);
quiver(ax1,0,0,4.25,0,0,'Color','k','LineWidth',1.8,'MaxHeadSize',0.08);
quiver(ax1,0,0,0,3.45,0,'Color','k','LineWidth',1.8,'MaxHeadSize',0.08);
plot(ax1,[0,projection(1)],[0,projection(2)],'-k','LineWidth',2.2);
plot(ax1,0,0,'o','MarkerSize',10,'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax1,projection(1),projection(2),'o','MarkerSize',9,'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

alphaSamples = linspace(0,alpha,120);
alphaRadius = 1.20;
plot(ax1,alphaRadius*cos(alphaSamples),alphaRadius*sin(alphaSamples), ...
    '-','Color',cAngle,'LineWidth',2.0);

text(ax1,4.40,-0.08,'x','FontWeight','bold','FontSize',17);
text(ax1,-0.10,3.74,'y', ...
    'FontWeight','bold','FontSize',17, ...
    'HorizontalAlignment','center','VerticalAlignment','bottom');
text(ax1,0,-0.43,'Observer','Color',cObserver, ...
    'FontWeight','bold','FontSize',15,'HorizontalAlignment','center');
text(ax1,projection(1),projection(2)+0.28,'Target projection', ...
    'Color',cTarget,'FontWeight','bold','FontSize',15, ...
    'HorizontalAlignment','center','VerticalAlignment','bottom');
text(ax1,1.68,2.04,'\rho_{xy}', ...
    'Color',cProjection,'FontWeight','bold','FontSize',16);
text(ax1,1.52*cos(alpha/2),1.52*sin(alpha/2)+0.06,'\alpha', ...
    'Color',cAngle,'FontWeight','bold','FontSize',18);

title(ax1,'(a) Right ascension, \alpha', ...
    'FontSize',16,'FontWeight','bold');
xlim(ax1,[-0.72,4.65]);
ylim(ax1,[-0.72,3.85]);
set(findall(figRa,'Type','text'),'FontName','Times New Roman');

raFigureFile = fullfile(outputDir, ...
    'measurement_model_right_ascension.eps');
inspect_before_export(figRa,inspectFigure, ...
    'right-ascension measurement geometry');
export_publication_eps(figRa,raFigureFile);
close(figRa);

% ---------------- (b) Declination ----------------
figDec = publication_figure(4.45,4.55);
ax2 = axes(figDec,'Units','normalized','Position',[0.10,0.12,0.82,0.80]);
hold(ax2,'on');
axis(ax2,'equal');
axis(ax2,'off');

target = [rhoXY,rho(3)];
quiver(ax2,0,0,4.35,0,0,'Color','k','LineWidth',1.8,'MaxHeadSize',0.08);
quiver(ax2,0,0,0,3.25,0,'Color','k','LineWidth',1.8,'MaxHeadSize',0.08);
plot(ax2,[0,target(1)],[0,target(2)],'-k','LineWidth',2.2);
plot(ax2,[target(1),target(1)],[0,target(2)],'--', ...
    'Color',cProjection,'LineWidth',1.5);
plot(ax2,0,0,'o','MarkerSize',10,'MarkerFaceColor',cObserver,'MarkerEdgeColor','k');
plot(ax2,target(1),target(2),'o','MarkerSize',9,'MarkerFaceColor',cTarget,'MarkerEdgeColor','k');

deltaSamples = linspace(0,delta,120);
deltaRadius = 1.22;
plot(ax2,deltaRadius*cos(deltaSamples),deltaRadius*sin(deltaSamples), ...
    '-','Color',cAngle,'LineWidth',2.0);

% rho_xy is the horizontal coordinate in this panel and therefore
% labels the horizontal axis at its positive arrow tip.
text(ax2,4.48,-0.08,'\rho_{xy}', ...
    'Color','k','FontWeight','bold','FontSize',16, ...
    'HorizontalAlignment','left','VerticalAlignment','middle');
text(ax2,-0.12,3.42,'z','FontWeight','bold','FontSize',17);
text(ax2,0,-0.43,'Observer','Color',cObserver, ...
    'FontWeight','bold','FontSize',15,'HorizontalAlignment','center');
text(ax2,target(1),target(2)+0.28,'Target', ...
    'Color',cTarget,'FontWeight','bold','FontSize',15, ...
    'HorizontalAlignment','center','VerticalAlignment','bottom');
text(ax2,2.02,1.88,'\rho','FontWeight','bold','FontSize',16);
text(ax2,1.56*cos(delta/2),1.56*sin(delta/2)+0.06,'\delta', ...
    'Color',cAngle,'FontWeight','bold','FontSize',18);

title(ax2,'(b) Declination, \delta', ...
    'FontSize',16,'FontWeight','bold');
xlim(ax2,[-0.72,4.75]);
ylim(ax2,[-0.72,3.60]);
set(findall(figDec,'Type','text'),'FontName','Times New Roman');

decFigureFile = fullfile(outputDir, ...
    'measurement_model_declination.eps');
inspect_before_export(figDec,inspectFigure, ...
    'declination measurement geometry');
export_publication_eps(figDec,decFigureFile);
close(figDec);

outputs = struct();
outputs.figures = [string(raFigureFile);string(decFigureFile)];
outputs.rightAscensionFigure = string(raFigureFile);
outputs.declinationFigure = string(decFigureFile);
outputs.measurementType = "ANGLES_ONLY";

fprintf('Saved right-ascension measurement geometry to:\n  %s\n', ...
    raFigureFile);
fprintf('Saved declination measurement geometry to:\n  %s\n', ...
    decFigureFile);
end

function [fig,axesHandles,textAx] = schematic_figure_layout( ...
    numPanels,geometryHeightInches,textLineCount)
%SCHEMATIC_FIGURE_LAYOUT Create non-overlapping geometry and text regions.
% Width grows with panel count. Height grows with geometry plus a footer
% whose physical height is proportional to the requested text-row count.

validateattributes(numPanels,{'numeric'},{'scalar','integer','positive'});
validateattributes(geometryHeightInches,{'numeric'}, ...
    {'scalar','real','finite','positive'});
validateattributes(textLineCount,{'numeric'}, ...
    {'scalar','integer','nonnegative'});

panelWidthInches = 4.35;
sideMarginInches = 0.42;
panelGapInches = 0.42;
topMarginInches = 0.42;
bottomMarginInches = 0.18;
footerLineInches = 0.42;
footerPaddingInches = 0.28;

if textLineCount > 0
    footerHeightInches = footerPaddingInches + ...
        textLineCount*footerLineInches;
else
    footerHeightInches = 0;
end

widthInches = 2*sideMarginInches + ...
    numPanels*panelWidthInches + (numPanels-1)*panelGapInches;
heightInches = bottomMarginInches + footerHeightInches + ...
    geometryHeightInches + topMarginInches;

fig = publication_figure(widthInches,heightInches);

geometryBottomInches = bottomMarginInches + footerHeightInches;
axesHandles = gobjects(numPanels,1);
for k = 1:numPanels
    leftInches = sideMarginInches + ...
        (k-1)*(panelWidthInches+panelGapInches);
    axesHandles(k) = axes(fig,'Units','normalized','Position',[ ...
        leftInches/widthInches, ...
        geometryBottomInches/heightInches, ...
        panelWidthInches/widthInches, ...
        geometryHeightInches/heightInches]);
    axesHandles(k).PositionConstraint = 'innerposition';
end

if textLineCount > 0
    textAx = axes(fig,'Units','normalized','Position',[ ...
        sideMarginInches/widthInches, ...
        bottomMarginInches/heightInches, ...
        (widthInches-2*sideMarginInches)/widthInches, ...
        footerHeightInches/heightInches]);
    textAx.PositionConstraint = 'innerposition';
    xlim(textAx,[0,1]);
    ylim(textAx,[0,1]);
    axis(textAx,'off');
else
    textAx = gobjects(1);
end
end


function draw_angle_arc_2d(ax,origin,startAngle,endAngle,radius,color,lineWidth)
angle = linspace(startAngle,endAngle,100);
plot(ax,origin(1)+radius*cos(angle),origin(2)+radius*sin(angle), ...
    '-','Color',color,'LineWidth',lineWidth);
end


function outputs = create_tracking_cases(inspectFigure)
% Plot the three fixed target scenarios from TargetCaseDatabase.mat.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

mu = 1.215058560962404E-2;
LU = 384400;
TU = 375695;
VU = LU/TU;
odeOptions = odeset('RelTol',1e-13,'AbsTol',1e-13);

gatewayCfg = target_case_config("LUNAR_GATEWAY");
[tGateway,sGateway,gatewayInfo] = build_target_truth( ...
    gatewayCfg,table(),{}, {}, {},mu,odeOptions);

transferCfg = target_case_config("LOW_THRUST_TRANSFER");
[tTransfer,sTransfer,transferInfo] = build_target_truth( ...
    transferCfg,table(),{}, {}, {},mu,odeOptions);

% Visual context only: recover the complete periodic orbits containing the
% fixed low-thrust boundary states. These catalog trajectories are used only
% for this figure and do not define the transfer endpoints or solver inputs.
catalog = load(projectPaths.catalog,'T');
departureOrbit = find_reference_orbit_for_state( ...
    catalog.T,transferCfg.transfer.fixedDepartureState);
arrivalOrbit = find_reference_orbit_for_state( ...
    catalog.T,transferCfg.transfer.fixedTargetState);

impulseCfg = target_case_config("GATEWAY_IMPULSE");
[tImpulse,sImpulse,impulseInfo] = build_target_truth( ...
    impulseCfg,table(),{}, {}, {},mu,odeOptions);

[~,sNominalAfterPerilune] = ode45( ...
    @(t,s) cr3bp_dynamics(t,s,mu),tImpulse, ...
    impulseInfo.statePreImpulse,odeOptions);

[xL1,xL2] = cr3bp_L1L2(mu);

figureFiles = strings(3,1);
cGateway = [0.85,0.27,0.22];
cTransfer = [0.27,0.31,0.86];
cImpulse = [0.55,0.30,0.72];

% Simulate different line alpha values by blending against the white
% figure background. This remains reliable in vector EPS output.
nominalAlpha = 0.45;
postImpulseAlpha = 0.95;
cNominal = nominalAlpha*cGateway+ ...
    (1-nominalAlpha)*[1,1,1];
cPostImpulse = postImpulseAlpha*cImpulse+ ...
    (1-postImpulseAlpha)*[1,1,1];

cReference = [0.48,0.48,0.48];
cPoint = [0.80,0.80,0.80];

figGateway = publication_figure(7.2,6.5);
ax = axes(figGateway); prepare_axes(ax);
hGateway = plot3(ax,sGateway(:,1),sGateway(:,2),sGateway(:,3),'-','Color',cGateway,'LineWidth',2.8);
hMoon = draw_moon(ax,mu,LU);
hL1 = plot3(ax,xL1,0,0,'^','MarkerFaceColor',cPoint, ...
    'MarkerEdgeColor','k','MarkerSize',9,'LineWidth',1.2);
hL2 = plot3(ax,xL2,0,0,'v','MarkerFaceColor',cPoint, ...
    'MarkerEdgeColor','k','MarkerSize',9,'LineWidth',1.2);
format_case_axes(ax);
legendHandle = legend(ax,[hGateway,hMoon,hL1,hL2], ...
    {'Nominal Gateway','Moon','L1','L2'}, ...
    'Location','northoutside','Orientation','horizontal');
format_case_legend(legendHandle,4); axis(ax,'tight'); axis(ax,'vis3d'); format_case_axes(ax);
figureFiles(1) = fullfile(outputDir,'case_lunar_gateway.eps');
inspect_before_export(figGateway,inspectFigure,'Lunar Gateway case');
export_publication_eps(figGateway,figureFiles(1)); close(figGateway);

figTransfer = publication_figure(7.2,6.5);
ax = axes(figTransfer); prepare_axes(ax);
hDeparture = plot3(ax,departureOrbit(:,1),departureOrbit(:,2),departureOrbit(:,3),'-','Color',cReference,'LineWidth',1.3);
plot3(ax,arrivalOrbit(:,1),arrivalOrbit(:,2),arrivalOrbit(:,3),'-','Color',cReference,'LineWidth',1.3,'HandleVisibility','off');
hTransfer = plot3(ax,sTransfer(:,1),sTransfer(:,2),sTransfer(:,3),'-','Color',cTransfer,'LineWidth',3.0);
hStart = plot3(ax,sTransfer(1,1),sTransfer(1,2),sTransfer(1,3),'o','MarkerSize',9,'MarkerFaceColor',cGateway,'MarkerEdgeColor','k','LineWidth',1.2);
hEnd = plot3(ax,sTransfer(end,1),sTransfer(end,2),sTransfer(end,3),'s','MarkerSize',9,'MarkerFaceColor',cTransfer,'MarkerEdgeColor','k','LineWidth',1.2);
hMoon = draw_moon(ax,mu,LU);
hL1 = plot3(ax,xL1,0,0,'^','MarkerFaceColor',cPoint,'MarkerEdgeColor','k','MarkerSize',7,'LineWidth',1.0);
hL2 = plot3(ax,xL2,0,0,'v','MarkerFaceColor',cPoint,'MarkerEdgeColor','k','MarkerSize',7,'LineWidth',1.0);
format_case_axes(ax);
legendHandle = legend(ax,[hDeparture,hTransfer,hStart,hEnd,hMoon,hL1,hL2],{'Endpoint orbits','Transfer','Start','End','Moon','L1','L2'},'Location','northoutside','Orientation','horizontal');
format_case_legend(legendHandle,4); axis(ax,'tight'); axis(ax,'vis3d'); format_case_axes(ax);
figureFiles(2) = fullfile(outputDir,'case_low_thrust_transfer.eps');
inspect_before_export(figTransfer,inspectFigure,'low-thrust transfer case');
export_publication_eps(figTransfer,figureFiles(2)); close(figTransfer);

figImpulse = publication_figure(7.2,6.5);
ax = axes(figImpulse); prepare_axes(ax);
hNominal = plot3(ax,sNominalAfterPerilune(:,1),sNominalAfterPerilune(:,2),sNominalAfterPerilune(:,3),'--','Color',cNominal,'LineWidth',2.2);
hImpulse = plot3(ax,sImpulse(:,1),sImpulse(:,2),sImpulse(:,3), ...
    '-','Color',cPostImpulse,'LineWidth',3.0);
hBurn = plot3(ax,sImpulse(1,1),sImpulse(1,2),sImpulse(1,3),'p','MarkerSize',12,'MarkerFaceColor',[0.95,0.65,0.15],'MarkerEdgeColor','k','LineWidth',1.2);
hMoon = draw_moon(ax,mu,LU);
hL1 = plot3(ax,xL1,0,0,'^','MarkerFaceColor',cPoint, ...
    'MarkerEdgeColor','k','MarkerSize',9,'LineWidth',1.2);
hL2 = plot3(ax,xL2,0,0,'v','MarkerFaceColor',cPoint, ...
    'MarkerEdgeColor','k','MarkerSize',9,'LineWidth',1.2);
format_case_axes(ax);
legendHandle = legend(ax,[hNominal,hImpulse,hBurn,hMoon,hL1,hL2], ...
    {'Nominal Gateway','Post-impulse','10 m/s burn','Moon','L1','L2'}, ...
    'Location','northoutside','Orientation','horizontal');
format_case_legend(legendHandle,3); axis(ax,'tight'); axis(ax,'vis3d'); format_case_axes(ax);
figureFiles(3) = fullfile(outputDir,'case_gateway_perilune_impulse.eps');
inspect_before_export(figImpulse,inspectFigure,'Gateway impulse case');
export_publication_eps(figImpulse,figureFiles(3)); close(figImpulse);

caseName = ["Lunar Gateway";"Low-thrust transfer";"Perilune impulse"];
duration_TU = [tGateway(end);tTransfer(end);tImpulse(end)];
targetDefinition = ["Nominal Gateway orbit";"Fixed-boundary continuous low-thrust transfer";"10 m/s prograde burn at nominal Gateway perilune"];
deltaV_m_s = [NaN;NaN;impulseInfo.deltaV_m_s];
impulseDirection = ["";"";string(impulseInfo.direction)];
nominalPeriluneEpoch_TU = [NaN;NaN;impulseInfo.periluneEpochNominal_TU];
transferFinalResidualNorm = [NaN;transferInfo.finalResidualNorm;NaN];
caseMetadata = table(caseName,targetDefinition,duration_TU,deltaV_m_s,impulseDirection,nominalPeriluneEpoch_TU,transferFinalResidualNorm);
metadataFile = fullfile(outputDir,'tracking_case_metadata.csv'); writetable(caseMetadata,metadataFile);

conditionCase = ["Lunar Gateway";"Lunar Gateway";"Low-thrust transfer";"Low-thrust transfer";"Perilune impulse";"Perilune impulse";"Perilune impulse"];
condition = ["Initial";"Final";"Departure";"Arrival";"Pre-impulse";"Post-impulse";"Final"];
caseEpoch_TU = [tGateway(1);tGateway(end);tTransfer(1);tTransfer(end);0;0;tImpulse(end)];
referenceEpoch_TU = [0;gatewayCfg.gateway.period;NaN;NaN;impulseInfo.periluneEpochNominal_TU;impulseInfo.periluneEpochNominal_TU;NaN];
stateND = [sGateway(1,:);sGateway(end,:);transferCfg.transfer.fixedDepartureState(:).';transferCfg.transfer.fixedTargetState(:).';impulseInfo.statePreImpulse(:).';impulseInfo.statePostImpulse(:).';sImpulse(end,:)];
stateConditionsND = table(conditionCase,condition,caseEpoch_TU,referenceEpoch_TU,stateND(:,1),stateND(:,2),stateND(:,3),stateND(:,4),stateND(:,5),stateND(:,6),'VariableNames',{'caseName','condition','caseEpoch_TU','referenceEpoch_TU','x_LU','y_LU','z_LU','vx_LU_TU','vy_LU_TU','vz_LU_TU'});
stateDimensional = stateND; stateDimensional(:,1:3) = stateDimensional(:,1:3)*LU; stateDimensional(:,4:6) = stateDimensional(:,4:6)*VU;
stateConditionsDimensional = table(conditionCase,condition,caseEpoch_TU,referenceEpoch_TU,stateDimensional(:,1),stateDimensional(:,2),stateDimensional(:,3),stateDimensional(:,4),stateDimensional(:,5),stateDimensional(:,6),'VariableNames',{'caseName','condition','caseEpoch_TU','referenceEpoch_TU','x_km','y_km','z_km','vx_km_s','vy_km_s','vz_km_s'});
normalizedStateFile = fullfile(outputDir,'tracking_case_state_conditions_nd.csv'); dimensionalStateFile = fullfile(outputDir,'tracking_case_state_conditions_dimensional.csv');
writetable(stateConditionsND,normalizedStateFile); writetable(stateConditionsDimensional,dimensionalStateFile);
latexRowsFile = fullfile(outputDir,'tracking_case_state_rows.tex'); write_latex_state_rows(latexRowsFile,stateConditionsND);

reproduction = struct();
reproduction.frame = "Earth-Moon barycentric rotating CR3BP";
reproduction.units = struct('position',"LU",'velocity',"LU/TU",'time',"TU",'LU_km',LU,'TU_s',TU,'VU_km_s',VU,'mu',mu);
reproduction.gateway = struct('config',gatewayCfg.gateway,'initialState',sGateway(1,:).','finalState',sGateway(end,:).');
reproduction.transfer = struct('config',transferCfg.transfer,'initialState',sTransfer(1,:).','finalState',sTransfer(end,:).','lambda0',transferInfo.lambda0,'timeOfFlight_TU',transferInfo.tf,'finalResidualNorm',transferInfo.finalResidualNorm);
reproduction.impulse = struct('config',impulseCfg.impulse,'nominalPeriluneEpoch_TU',impulseInfo.periluneEpochNominal_TU,'preImpulseState',impulseInfo.statePreImpulse,'postImpulseState',impulseInfo.statePostImpulse,'deltaVVector_LU_TU',impulseInfo.deltaVVector_LU_TU,'finalState',sImpulse(end,:).');
reproductionFile = fullfile(outputDir,'tracking_case_reproduction.mat'); save(reproductionFile,'reproduction','-v7');

outputs = struct(); outputs.figures = figureFiles; outputs.gatewayFigure = figureFiles(1); outputs.lowThrustFigure = figureFiles(2); outputs.impulseFigure = figureFiles(3);
outputs.metadata = string(metadataFile); outputs.normalizedStateConditionsFile = string(normalizedStateFile); outputs.dimensionalStateConditionsFile = string(dimensionalStateFile); outputs.latexStateRows = string(latexRowsFile); outputs.reproductionFile = string(reproductionFile);
outputs.stateConditionsND = stateConditionsND; outputs.stateConditionsDimensional = stateConditionsDimensional; outputs.gatewayInfo = gatewayInfo; outputs.transferInfo = transferInfo; outputs.impulseInfo = impulseInfo;

fprintf('Saved the three separate tracking-case figures to:\n  %s\n',outputDir);
fprintf('\nNormalized initial, maneuver, and final conditions:\n'); disp(stateConditionsND);
end


function orbitState = find_reference_orbit_for_state(T,targetState)
%FIND_REFERENCE_ORBIT_FOR_STATE Find the catalog trajectory nearest a state.
% This is a plot-only phase-independent lookup. It intentionally returns
% only the trajectory, not a catalog row, orbit ID, or observer slot.

assert(istable(T) && ismember('state',T.Properties.VariableNames), ...
    'Observer catalog must contain the state trajectory column.');

targetState = targetState(:).';
assert(numel(targetState)==6 && all(isfinite(targetState)), ...
    'Reference state must contain six finite CR3BP components.');

bestError = inf;
bestOrbit = [];

for k = 1:height(T)
    state = T.state{k};
    if isempty(state) || size(state,2)<6
        continue;
    end

    state = state(:,1:6);
    finiteRows = all(isfinite(state),2);
    state = state(finiteRows,:);
    if isempty(state)
        continue;
    end

    stateError = vecnorm(state-targetState,2,2);
    thisError = min(stateError);
    if thisError < bestError
        bestError = thisError;
        bestOrbit = T.state{k};
    end
end

assert(~isempty(bestOrbit) && isfinite(bestError), ...
    'Could not identify a reference periodic orbit for the fixed LT state.');

% The selected catalog trajectories are densely sampled, so a fixed state
% belonging to one of them should have a close phase-space neighbor. Keep
% the tolerance loose enough to accommodate interpolation of the stored LT
% endpoint while still catching an unrelated catalog/database mismatch.
assert(bestError < 2.5e-2, ...
    ['Fixed LT endpoint does not match the observer catalog closely enough ' ...
     'for reference-orbit plotting (minimum state error %.6e).'],bestError);

orbitState = bestOrbit(:,1:6);
end


function prepare_axes(ax)

hold(ax,'on');
box(ax,'on');
axis(ax,'equal');
view(ax,-37.5,30);
ax.Projection = 'perspective';
xlabel(ax,'x (LU)');
ylabel(ax,'y (LU)');
zlabel(ax,'z (LU)');
end


function h = draw_moon(ax,mu,LU)

radius = 1737.1/LU;
[x,y,z] = sphere(30);
h = surf(ax,radius*x+1-mu,radius*y,radius*z, ...
    'FaceColor',[0.72,0.72,0.72], ...
    'EdgeColor','none','FaceLighting','gouraud');
camlight(ax,'headlight');
material(ax,'dull');
end


function [xL1,xL2] = cr3bp_L1L2(mu)

equilibrium = @(x) x ...
    -(1-mu)*(x+mu)./abs(x+mu).^3 ...
    -mu*(x-(1-mu))./abs(x-(1-mu)).^3;

delta = (mu/3)^(1/3);
xL1 = fzero(equilibrium,[1-mu-delta,1-mu-1e-6]);
xL2 = fzero(equilibrium,[1-mu+1e-6,1-mu+delta+0.5]);
end


function write_latex_state_rows(fileName,stateTable)

fid = fopen(fileName,'w');
assert(fid>=0,'Could not create LaTeX state-row file: %s',fileName);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

for k = 1:height(stateTable)
    state = [ ...
        stateTable.x_LU(k),stateTable.y_LU(k),stateTable.z_LU(k), ...
        stateTable.vx_LU_TU(k),stateTable.vy_LU_TU(k), ...
        stateTable.vz_LU_TU(k)];

    latexSeparator = [',' char(92) ','];
    stateText = strjoin(compose('%.12g',state),latexSeparator);

    fprintf(fid,'%s & %s & %.12g & $[%s]^{\\mathsf{T}}$ \\\\\n', ...
        char(stateTable.caseName(k)),char(stateTable.condition(k)), ...
        stateTable.caseEpoch_TU(k),char(stateText));
end
end


function format_case_axes(ax)

format_publication_axes(ax,12);
ax.Units = 'normalized';

% Let MATLAB size the inner 3-D plot box from the rendered tick and axis
% labels. OuterPosition reserves a consistent band above the axes for the
% legend without forcing every viewing angle into one fixed Position.
ax.PositionConstraint = 'outerposition';
ax.OuterPosition = [0.02,0.04,0.96,0.80];
drawnow;
ax.LooseInset = max(ax.TightInset,0.035);
end


function format_case_legend(legendHandle,numColumns)

legendHandle.Box = 'on';
legendHandle.FontName = 'Times New Roman';
legendHandle.FontSize = 12;
legendHandle.FontWeight = 'bold';
legendHandle.ItemTokenSize = [16,9];
legendHandle.NumColumns = numColumns;
place_legend_above(legendHandle,numColumns,12);
end


function fig = publication_figure(widthInches,heightInches)

fig = figure( ...
    'Color','w', ...
    'Units','inches', ...
    'Position',[1,1,widthInches,heightInches], ...
    'PaperUnits','inches', ...
    'PaperPosition',[0,0,widthInches,heightInches], ...
    'PaperSize',[widthInches,heightInches], ...
    'PaperPositionMode','manual', ...
    'Renderer','painters', ...
    'InvertHardcopy','off');
end


function format_publication_axes(ax,fontSize)

fontSize = max(fontSize,12);

set(ax, ...
    'FontName','Times New Roman', ...
    'FontSize',fontSize, ...
    'FontWeight','bold', ...
    'LineWidth',1.35, ...
    'TickLabelInterpreter','tex');

ax.XLabel.FontName = 'Times New Roman';
ax.YLabel.FontName = 'Times New Roman';
ax.ZLabel.FontName = 'Times New Roman';
ax.XLabel.FontSize = fontSize+2;
ax.YLabel.FontSize = fontSize+2;
ax.ZLabel.FontSize = fontSize+2;
ax.XLabel.FontWeight = 'bold';
ax.YLabel.FontWeight = 'bold';
ax.ZLabel.FontWeight = 'bold';
end


function place_legend_above(legendHandle,numColumns,fontSize)

fontSize = max(fontSize,12);

legendHandle.Location = 'northoutside';
legendHandle.Orientation = 'horizontal';
legendHandle.NumColumns = numColumns;
legendHandle.FontSize = fontSize;
drawnow;

% MATLAB's automatic northoutside placement can extend beyond the paper
% canvas. Preserve its natural size, center it, and keep it inside the
% exported EPS bounding box.
legendHandle.Units = 'normalized';
position = legendHandle.Position;
position(1) = max(0.02,0.5-position(3)/2);
position(2) = min(position(2),0.97-position(4));
legendHandle.Position = position;
end


function draw_text_callout(ax,labelPosition,targetPosition, ...
    labelText,labelColor,fontSize)

fontSize = max(fontSize,12);
direction = targetPosition-labelPosition;
distance = norm(direction);

if distance > 0
    arrowStart = labelPosition + 0.46*direction/distance;
else
    arrowStart = labelPosition;
end

quiver(ax,arrowStart(1),arrowStart(2), ...
    targetPosition(1)-arrowStart(1), ...
    targetPosition(2)-arrowStart(2),0, ...
    'Color',labelColor,'LineWidth',1.3,'MaxHeadSize',0.18, ...
    'HandleVisibility','off');

text(ax,labelPosition(1),labelPosition(2),labelText, ...
    'Color',labelColor,'FontWeight','bold','FontSize',fontSize, ...
    'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'BackgroundColor','w','Margin',0.8);
end


function enforce_minimum_font_size(fig,minFontSize)

fontObjects = findall(fig,'-property','FontSize');
for k = 1:numel(fontObjects)
    try
        if fontObjects(k).FontSize < minFontSize
            fontObjects(k).FontSize = minFontSize;
        end
    catch
        % Some graphics proxy objects expose FontSize as read-only.
    end
end
end


function export_publication_eps(fig,fileName)

enforce_minimum_font_size(fig,12);
drawnow;
set(fig,'Renderer','painters','PaperPositionMode','manual');
print(fig,char(fileName),'-depsc2','-painters','-r600');
fprintf('Saved vector EPS: %s\n',fileName);
end


function inspect_before_export(fig,inspectFigure,description)

enforce_minimum_font_size(fig,12);

if inspectFigure
    figure(fig);
    drawnow;
    fprintf('Previewing the %s figure for 5 seconds before export.\n', ...
        description);
    pause(5);
end
end
