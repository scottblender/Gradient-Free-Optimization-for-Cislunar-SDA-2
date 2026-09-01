function outputs = plot_study_definition_figures(inspectFigures)
% Generate catalog, slot-definition, and target-case figures.

if nargin<1 || isempty(inspectFigures), inspectFigures = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
setup_project();

fprintf('\n--- Study-definition figures ---\n');

fprintf('\n1/3 Orbit-family catalog figures\n');
outputs.catalog = create_orbit_catalog_figures(inspectFigures);

fprintf('\n2/3 Equal-time slot definition\n');
outputs.slots = create_slot_definition(inspectFigures);

fprintf('\n3/3 Tracking cases\n');
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

            selectedRows = select_evenly_spaced_rows( ...
                familyRows,T.zAmplitude(familyRows),perFamily);

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
        zlabel(ax,'Z (LU)');
        axis(ax,'tight');
        axis(ax,'vis3d');
    else
        ax.Projection = 'orthographic';
        view(ax,2);

        familyRows = find(family=="DRO");
        assert(~isempty(familyRows),'No selected DROs were found.');
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

    xlabel(ax,'X (LU)');
    ylabel(ax,'Y (LU)');
    format_publication_axes(ax,13);

    legendHandle = legend(ax,legendHandles,cellstr(legendLabels), ...
        'Location','northeast','Orientation','vertical');
    legendHandle.Box = 'on';
    legendHandle.FontName = 'Times New Roman';
    legendHandle.FontSize = 10;
    legendHandle.FontWeight = 'normal';
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
orbitIndex = find(family=="NHL1",1,'first');
if isempty(orbitIndex)
    orbitIndex = find(family=="NNRHL1",1,'first');
end
assert(~isempty(orbitIndex), ...
    'No representative northern L1 orbit was found.');

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
[xL1,~] = cr3bp_L1L2(mu);

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
hL1 = plot3(ax,xL1,0,0,'^','MarkerSize',9, ...
    'MarkerFaceColor',[0.80,0.80,0.80], ...
    'MarkerEdgeColor','k','LineWidth',1.2);

set(ax,'FontName','Times New Roman','FontSize',18, ...
    'FontWeight','bold','LineWidth',1.8);
legendHandle = legend(ax,[hOrbit,hSlots,hSelected,hNext,hMoon,hL1], ...
    {'Orbit','Candidate slots','Slot j','Slot j+1','Moon','L1'}, ...
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
cNominal = [0.35,0.35,0.35];
cReference = [0.48,0.48,0.48];
cPoint = [0.80,0.80,0.80];

figGateway = publication_figure(7.2,6.5);
ax = axes(figGateway); prepare_axes(ax);
hGateway = plot3(ax,sGateway(:,1),sGateway(:,2),sGateway(:,3),'-','Color',cGateway,'LineWidth',2.8);
hMoon = draw_moon(ax,mu,LU);
plot3(ax,xL1,0,0,'^','MarkerFaceColor',cPoint,'MarkerEdgeColor','k','MarkerSize',9,'LineWidth',1.2);
plot3(ax,xL2,0,0,'v','MarkerFaceColor',cPoint,'MarkerEdgeColor','k','MarkerSize',9,'LineWidth',1.2);
format_case_axes(ax);
legendHandle = legend(ax,[hGateway,hMoon],{'Nominal target','Moon'},'Location','northoutside','Orientation','horizontal');
format_case_legend(legendHandle,2); axis(ax,'tight'); axis(ax,'vis3d'); format_case_axes(ax);
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
hImpulse = plot3(ax,sImpulse(:,1),sImpulse(:,2),sImpulse(:,3),'-','Color',cImpulse,'LineWidth',3.0);
hBurn = plot3(ax,sImpulse(1,1),sImpulse(1,2),sImpulse(1,3),'p','MarkerSize',12,'MarkerFaceColor',[0.95,0.65,0.15],'MarkerEdgeColor','k','LineWidth',1.2);
hMoon = draw_moon(ax,mu,LU);
plot3(ax,xL1,0,0,'^','MarkerFaceColor',cPoint,'MarkerEdgeColor','k','MarkerSize',9,'LineWidth',1.2);
plot3(ax,xL2,0,0,'v','MarkerFaceColor',cPoint,'MarkerEdgeColor','k','MarkerSize',9,'LineWidth',1.2);
format_case_axes(ax);
legendHandle = legend(ax,[hNominal,hImpulse,hBurn,hMoon],{'Nominal','Post-impulse','10 m/s burn','Moon'},'Location','northoutside','Orientation','horizontal');
format_case_legend(legendHandle,2); axis(ax,'tight'); axis(ax,'vis3d'); format_case_axes(ax);
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

    stateText = strjoin(compose('%.12g',state),',\,');

    fprintf(fid,'%s & %s & %.12g & $[%s]^{\\mathsf{T}}$ \\\\\n', ...
        char(stateTable.caseName(k)),char(stateTable.condition(k)), ...
        stateTable.caseEpoch_TU(k),char(stateText));
end
end


function selectedRows = select_evenly_spaced_rows( ...
    candidateRows,spacingValue,numberToSelect)

candidateRows = candidateRows(:);
spacingValue = spacingValue(:);
numberToSelect = min(numberToSelect,numel(candidateRows));

[spacingValue,order] = sort(spacingValue);
candidateRows = candidateRows(order);
targets = linspace(spacingValue(1),spacingValue(end),numberToSelect);

available = true(numel(candidateRows),1);
selectedRows = zeros(numberToSelect,1);

for index = 1:numberToSelect
    distance = abs(spacingValue-targets(index));
    distance(~available) = inf;
    [~,selected] = min(distance);
    selectedRows(index) = candidateRows(selected);
    available(selected) = false;
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
legendHandle.FontSize = 11;
legendHandle.FontWeight = 'bold';
legendHandle.ItemTokenSize = [16,9];
legendHandle.NumColumns = numColumns;
place_legend_above(legendHandle,numColumns,11);
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


function export_publication_eps(fig,fileName)

drawnow;
set(fig,'Renderer','painters','PaperPositionMode','manual');
print(fig,char(fileName),'-depsc2','-painters','-r600');
fprintf('Saved vector EPS: %s\n',fileName);
end


function inspect_before_export(fig,inspectFigure,description)

if inspectFigure
    figure(fig);
    drawnow;
    fprintf('Previewing the %s figure for 5 seconds before export.\n', ...
        description);
    pause(5);
end
end
