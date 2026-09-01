function outputs = plot_study_definition_figures(inspectFigures)
% Generate catalog, slot-definition, and target-case figures.

if nargin<1 || isempty(inspectFigures), inspectFigures = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
setup_project();

fprintf('\n--- Study-definition figures ---\n');

fprintf('\n1/3 Orbit catalog characteristics\n');
outputs.catalog = create_orbit_catalog_characteristics(inspectFigures);

fprintf('\n2/3 Equal-time slot definition\n');
outputs.slots = create_slot_definition(inspectFigures);

fprintf('\n3/3 Tracking cases\n');
fprintf('The low-thrust panel solves the transfer and can take several minutes.\n');
outputs.cases = create_tracking_cases(inspectFigures);

fprintf('\nAll study-definition figures were generated.\n');
end


function outputs = create_orbit_catalog_characteristics(inspectFigure)
% Plot quantitative characteristics of all 450 selected catalog orbits.
% The accompanying CSV provides the values needed for the family table.

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

familyCategory = categorical(family,familyOrder,familyOrder);

fig = figure('Color','w','Units','inches', ...
    'Position',[1,1,12,7.4], ...
    'PaperUnits','inches','PaperPosition',[0,0,12,7.4]);

layout = tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');

plot_metric(nexttile(layout),familyCategory,periluneAltitude_km, ...
    'Perilune altitude (km)',[0.18,0.45,0.75]);
plot_metric(nexttile(layout),familyCategory,apoluneAltitude_km, ...
    'Apolune altitude (km)',[0.85,0.33,0.25]);
plot_metric(nexttile(layout),familyCategory,period_days, ...
    'Orbital period (days)',[0.30,0.65,0.40]);
plot_metric(nexttile(layout),familyCategory,stabilityIndex, ...
    'Stability index',[0.55,0.40,0.75]);
plot_metric(nexttile(layout),familyCategory,inPlaneAmplitude_km, ...
    'In-plane amplitude, A_{xy} (km)',[0.90,0.60,0.15]);
plot_metric(nexttile(layout),familyCategory,outOfPlaneAmplitude_km, ...
    'Out-of-plane amplitude, A_z (km)',[0.20,0.65,0.70]);

figureFile = fullfile(outputDir,'orbit_catalog_characteristics.eps');
inspect_before_export(fig,inspectFigure,'orbit catalog characteristics');
exportgraphics(fig,figureFile,'ContentType','image','Resolution',600);
close(fig);

outputs = struct();
outputs.figure = string(figureFile);
outputs.familySummary = string(summaryFile);
outputs.orbitMetrics = string(metricFile);
outputs.numOrbits = nOrbit;
outputs.numFamilies = nFamily;

fprintf('Saved orbit-catalog figure and tables to:\n  %s\n',outputDir);
end



function plot_metric(ax,group,value,yLabel,color)

boxchart(ax,group,value, ...
    'BoxFaceColor',color);

grid(ax,'on');
box(ax,'on');
ylabel(ax,yLabel);
xtickangle(ax,35);
set(ax,'FontName','Times New Roman','FontSize',11, ...
    'FontWeight','bold','LineWidth',1.2,'TickLabelInterpreter','tex');
end


function outputs = create_slot_definition(inspectFigure)
% Illustrate the exact equal-time, endpoint-excluded 50-slot definition.

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
if isempty(orbitIndex)
    orbitIndex = find(family=="NHL1",1,'first');
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

fig = figure('Color','w','Units','inches', ...
    'Position',[1,1,12,4.2], ...
    'PaperUnits','inches','PaperPosition',[0,0,12,4.2]);
layout = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');

% (a) Geometric placement of all 50 equal-time samples.
ax1 = nexttile(layout);
hold(ax1,'on'); box(ax1,'on'); axis(ax1,'equal');
plotStep = max(1,round(size(rawState,1)/500));
hOrbit = plot3(ax1,rawState(1:plotStep:end,1), ...
    rawState(1:plotStep:end,2),rawState(1:plotStep:end,3), ...
    '-','Color',orbitColor,'LineWidth',1.8);
hSlots = plot3(ax1,slotState(:,1),slotState(:,2),slotState(:,3), ...
    'o','MarkerSize',4.5,'MarkerFaceColor','w', ...
    'MarkerEdgeColor',neutralColor,'LineWidth',0.9);
hSelected = plot3(ax1,slotState(selectedSlot,1),slotState(selectedSlot,2), ...
    slotState(selectedSlot,3),'o','MarkerSize',8, ...
    'MarkerFaceColor',selectedColor,'MarkerEdgeColor','k');
hNext = plot3(ax1,slotState(nextSlot,1),slotState(nextSlot,2), ...
    slotState(nextSlot,3),'s','MarkerSize',8, ...
    'MarkerFaceColor',nextColor,'MarkerEdgeColor','k');
xlabel(ax1,'x (LU)'); ylabel(ax1,'y (LU)'); zlabel(ax1,'z (LU)');
title(ax1,'(a) Equal-time states');
view(ax1,32,24);
legend(ax1,[hOrbit,hSlots,hSelected,hNext], ...
    {'Orbit','Candidate slots','Slot j','Slot j+1'}, ...
    'Location','northoutside','Orientation','horizontal','FontSize',10);

% (b) Exact normalized phase convention, including endpoint exclusion.
ax2 = nexttile(layout);
hold(ax2,'on'); box(ax2,'on');
phase = slotTime/period;
plot(ax2,[0,1],[0,0],'-','Color',0.65*[1,1,1],'LineWidth',1.2);
scatter(ax2,phase,zeros(size(phase)),24,'w','filled', ...
    'MarkerEdgeColor',neutralColor,'LineWidth',0.8);
scatter(ax2,phase(selectedSlot),0,70,selectedColor,'filled', ...
    'MarkerEdgeColor','k');
scatter(ax2,phase(nextSlot),0,70,nextColor,'s','filled', ...
    'MarkerEdgeColor','k');
plot(ax2,1,0,'o','MarkerSize',7,'MarkerFaceColor','w', ...
    'MarkerEdgeColor',[0.75,0.20,0.20],'LineWidth',1.5);
plot(ax2,phase([selectedSlot,nextSlot]),[0.13,0.13],'-k','LineWidth',1.2);
plot(ax2,phase([selectedSlot,selectedSlot]),[0,0.13],':k');
plot(ax2,phase([nextSlot,nextSlot]),[0,0.13],':k');
text(ax2,mean(phase([selectedSlot,nextSlot])),0.16, ...
    '\Delta t/T=1/50','HorizontalAlignment','center', ...
    'FontName','Times New Roman','FontWeight','bold');
text(ax2,0.99,-0.15,{'t=T','not stored'}, ...
    'HorizontalAlignment','right','VerticalAlignment','top', ...
    'FontName','Times New Roman','FontWeight','bold');
xlabel(ax2,'Normalized epoch, t/T');
yticks(ax2,[]); ylim(ax2,[-0.28,0.28]); xlim(ax2,[-0.02,1.02]);
title(ax2,'(b) Endpoint-excluded phase grid');

% (c) Equal time does not imply equal arc length.
ax3 = nexttile(layout);
hold(ax3,'on'); box(ax3,'on'); grid(ax3,'on');
plot(ax3,slotNumber,adjacentChord_km,'-o', ...
    'Color',orbitColor,'MarkerFaceColor',orbitColor, ...
    'MarkerSize',3.5,'LineWidth',1.5);
plot(ax3,selectedSlot,adjacentChord_km(selectedSlot),'o', ...
    'MarkerSize',8,'MarkerFaceColor',selectedColor,'MarkerEdgeColor','k');
yline(ax3,median(adjacentChord_km),'--','Median', ...
    'Color',0.35*[1,1,1],'LabelHorizontalAlignment','left');
xlabel(ax3,'Slot j');
ylabel(ax3,'Chord distance j to j+1 (km)');
xlim(ax3,[1,numSlots]);
title(ax3,'(c) Unequal spatial separation');

allAxes = [ax1,ax2,ax3];
set(allAxes,'FontName','Times New Roman','FontSize',11, ...
    'FontWeight','bold','LineWidth',1.2,'TickLabelInterpreter','tex');

figureFile = fullfile(outputDir,'equal_time_slot_definition.eps');
inspect_before_export(fig,inspectFigure,'equal-time slot-definition');
exportgraphics(fig,figureFile,'ContentType','image','Resolution',600);
close(fig);

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
outputs.figure = string(figureFile);
outputs.summary = string(summaryFile);
outputs.slotSummary = slotSummary;

fprintf('Saved equal-time slot-definition figure to:\n  %s\n',figureFile);
end


function outputs = create_tracking_cases(inspectFigure)
% Plot the three target scenarios used for the revised optimization study.

if nargin<1 || isempty(inspectFigure), inspectFigure = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

outputDir = fullfile(projectPaths.results,'study_definition_figures');
if ~isfolder(outputDir), mkdir(outputDir); end

catalog = load(projectPaths.catalog,'T');
T = catalog.T;

mu = 1.215058560962404E-2;
LU = 384400;
TU = 375695;
VU = LU/TU;
odeOptions = odeset('RelTol',1e-13,'AbsTol',1e-13);

periodAll = T.('Period (TU) ');
rawTimes = T.time;
rawStates = T.state;
numSlots = 50;
orbitDatabase = build_slot_database( ...
    rawTimes,rawStates,periodAll,numSlots);

% Case 1: nominal Lunar Gateway tracking.
gatewayCfg = struct();
gatewayCfg.s0 = [1.02202108343387,0,-0.182096487798513, ...
                 0,-0.103255420206012,0]';
gatewayCfg.period = 1.51110546287394;
gatewayCfg.dt = 0.001;
gatewayCfg.Nperiods = 1;
[tGateway,sGateway,gatewayInfo] = build_truth_gateway( ...
    gatewayCfg,mu,odeOptions);

% Case 2: low-thrust transfer using stable orbit IDs, never row numbers.
referenceFile = fullfile(projectPaths.data,'transfer_reference.mat');
assert(isfile(referenceFile), ...
    'Transfer reference file was not found: %s',referenceFile);
reference = load(referenceFile,'transferRef');
transferRef = reference.transferRef;

assert(ismember('orbitID',T.Properties.VariableNames), ...
    'The catalog does not contain stable orbitID values.');
catalogIDs = string(T.orbitID);

departureID = string(transferRef.dep.orbitID);
arrivalID = string(transferRef.arr.orbitID);
departureIndex = find(catalogIDs==departureID,1);
arrivalIndex = find(catalogIDs==arrivalID,1);
assert(~isempty(departureIndex) && ~isempty(arrivalIndex), ...
    'A transfer-reference orbit is absent from the selected catalog.');

departureStateError = norm( ...
    rawStates{departureIndex}(1,:)-transferRef.dep.state0);
arrivalStateError = norm( ...
    rawStates{arrivalIndex}(1,:)-transferRef.arr.state0);
departurePeriodError_TU = abs( ...
    periodAll(departureIndex)-transferRef.dep.period);
arrivalPeriodError_TU = abs( ...
    periodAll(arrivalIndex)-transferRef.arr.period);

assert(departureStateError<=1e-12 && arrivalStateError<=1e-12, ...
    'A low-thrust endpoint orbit changed during catalog rebuilding.');
assert(departurePeriodError_TU<=1e-12 && arrivalPeriodError_TU<=1e-12, ...
    'A low-thrust endpoint period changed during catalog rebuilding.');

endpoint = ["Departure";"Arrival"];
resolvedCatalogRow = [departureIndex;arrivalIndex];
resolvedOrbitID = [departureID;arrivalID];
slot = [double(transferRef.dep.slot);double(transferRef.arr.slot)];
sourceInitialStateError = [departureStateError;arrivalStateError];
periodError_TU = [departurePeriodError_TU;arrivalPeriodError_TU];
endpointAudit = table(endpoint,resolvedCatalogRow,resolvedOrbitID,slot, ...
    sourceInitialStateError,periodError_TU);

fprintf('\n--- Low-thrust endpoint reference audit ---\n');
disp(endpointAudit);

missionCfg = struct();
missionCfg.type = "LOW_THRUST_TRANSFER";
missionCfg.transfer.depOrbitID = departureID;
missionCfg.transfer.depOrbitIndex = departureIndex;
missionCfg.transfer.depSlot = transferRef.dep.slot;
missionCfg.transfer.arrOrbitID = arrivalID;
missionCfg.transfer.arrOrbitIndex = arrivalIndex;
missionCfg.transfer.arrSlot = transferRef.arr.slot;
missionCfg.transfer.dt = 0.001;
missionCfg.transfer.solverMode = "LOW_THRUST_CLASS";
missionCfg.transfer.lowthrust.sigma = 1.0;
missionCfg.transfer.lowthrust.m0 = 1.0;
missionCfg.transfer.lowthrust.Tmax = 0.3672;
missionCfg.transfer.lowthrust.ve = 39.8;
missionCfg.transfer.lowthrust.tf_guess = 2.0;
missionCfg.transfer.lowthrust.tf_lb = 0.1;
missionCfg.transfer.lowthrust.tf_ub = 12.0;
missionCfg.transfer.lowthrust.lambda_guess = ...
    [-0.25;0.75;0.35;-0.20;0.40;0.10;0.05];
missionCfg.transfer.lowthrust.lambda_lb = -20*ones(7,1);
missionCfg.transfer.lowthrust.lambda_ub = 20*ones(7,1);
missionCfg.transfer.lowthrust.w_pos_indirect = 1;
missionCfg.transfer.lowthrust.w_vel_indirect = 1;
missionCfg.transfer.lowthrust.w_norm_indirect = 1;
missionCfg.transfer.lowthrust.w_mass_indirect = 1;

[tTransfer,sTransfer,transferInfo] = build_target_truth( ...
    missionCfg,T,orbitDatabase,rawTimes,rawStates,mu,odeOptions);

departureOrbit = rawStates{departureIndex};
arrivalOrbit = rawStates{arrivalIndex};

% Case 3: instantaneous prograde impulse at nominal Gateway perilune.
impulseCfg = struct();
impulseCfg.s0 = gatewayCfg.s0;
impulseCfg.period = gatewayCfg.period;
impulseCfg.dt = 0.001;
impulseCfg.duration_TU = 1.5;
impulseCfg.deltaV_m_s = 10;
impulseCfg.deltaV_LU_TU = ...
    (impulseCfg.deltaV_m_s/1000)/VU;
impulseCfg.direction = "PROGRADE";
impulseCfg.periluneSearchSamples = 4001;

[tImpulse,sImpulse,impulseInfo] = build_truth_gateway_impulse( ...
    impulseCfg,mu,odeOptions);
[~,sNominalAfterPerilune] = ode45( ...
    @(t,s) cr3bp_dynamics(t,s,mu),tImpulse, ...
    impulseInfo.statePreImpulse,odeOptions);

[xL1,xL2] = cr3bp_L1L2(mu);

fig = figure('Color','w','Units','inches', ...
    'Position',[1,1,12,4.6], ...
    'PaperUnits','inches','PaperPosition',[0,0,12,4.6]);
layout = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');

cGateway = [0.85,0.27,0.22];
cTransfer = [0.27,0.31,0.86];
cImpulse = [0.55,0.30,0.72];
cReference = [0.50,0.72,0.84];
cNominal = [0.35,0.35,0.35];

% Nominal Gateway case.
ax1 = nexttile(layout);
prepare_axes(ax1);
hGateway = plot3(ax1,sGateway(:,1),sGateway(:,2),sGateway(:,3), ...
    '-','Color',cGateway,'LineWidth',2.0);
hMoon1 = draw_moon(ax1,mu,LU);
plot3(ax1,xL1,0,0,'^','MarkerFaceColor',[0.75,0.75,0.75], ...
    'MarkerEdgeColor','k','MarkerSize',6);
plot3(ax1,xL2,0,0,'v','MarkerFaceColor',[0.75,0.75,0.75], ...
    'MarkerEdgeColor','k','MarkerSize',6);
title(ax1,'(a) Lunar Gateway');
legend(ax1,[hGateway,hMoon1],{'Nominal target','Moon'}, ...
    'Location','northoutside','Orientation','horizontal','FontSize',10);

% Low-thrust transfer case.
ax2 = nexttile(layout);
prepare_axes(ax2);
plot3(ax2,departureOrbit(:,1),departureOrbit(:,2),departureOrbit(:,3), ...
    '-','Color',cReference,'LineWidth',1.1);
plot3(ax2,arrivalOrbit(:,1),arrivalOrbit(:,2),arrivalOrbit(:,3), ...
    '-','Color',cReference,'LineWidth',1.1);
hTransfer = plot3(ax2,sTransfer(:,1),sTransfer(:,2),sTransfer(:,3), ...
    '-','Color',cTransfer,'LineWidth',2.0);
hStart = plot3(ax2,sTransfer(1,1),sTransfer(1,2),sTransfer(1,3), ...
    'o','MarkerSize',6,'MarkerFaceColor',cGateway,'MarkerEdgeColor','k');
hEnd = plot3(ax2,sTransfer(end,1),sTransfer(end,2),sTransfer(end,3), ...
    's','MarkerSize',6,'MarkerFaceColor',cTransfer,'MarkerEdgeColor','k');
hMoon2 = draw_moon(ax2,mu,LU);
title(ax2,'(b) Low-thrust transfer');
legend(ax2,[hTransfer,hStart,hEnd,hMoon2], ...
    {'Transfer','Start','End','Moon'}, ...
    'Location','northoutside','Orientation','horizontal','FontSize',10);

% Gateway perilune-impulse case.
ax3 = nexttile(layout);
prepare_axes(ax3);
hNominal = plot3(ax3,sNominalAfterPerilune(:,1), ...
    sNominalAfterPerilune(:,2),sNominalAfterPerilune(:,3), ...
    '--','Color',cNominal,'LineWidth',1.5);
hImpulse = plot3(ax3,sImpulse(:,1),sImpulse(:,2),sImpulse(:,3), ...
    '-','Color',cImpulse,'LineWidth',2.0);
hBurn = plot3(ax3,sImpulse(1,1),sImpulse(1,2),sImpulse(1,3), ...
    'p','MarkerSize',9,'MarkerFaceColor',[0.95,0.65,0.15], ...
    'MarkerEdgeColor','k');
hMoon3 = draw_moon(ax3,mu,LU);
title(ax3,'(c) Perilune impulse');
legend(ax3,[hNominal,hImpulse,hBurn,hMoon3], ...
    {'Nominal','Post-impulse','10 m/s burn','Moon'}, ...
    'Location','northoutside','Orientation','horizontal','FontSize',10);

allAxes = [ax1,ax2,ax3];
set(allAxes,'FontName','Times New Roman','FontSize',12, ...
    'FontWeight','bold','LineWidth',1.1,'TickLabelInterpreter','tex');

figureFile = fullfile(outputDir,'tracking_cases.eps');
inspect_before_export(fig,inspectFigure,'tracking cases');
exportgraphics(fig,figureFile,'ContentType','image','Resolution',600);
close(fig);

caseName = ["Lunar Gateway";"Low-thrust transfer";"Perilune impulse"];
duration_TU = [tGateway(end);tTransfer(end);tImpulse(end)];
targetDefinition = [ ...
    "Nominal Gateway orbit"; ...
    "Continuous low-thrust cislunar transfer"; ...
    "10 m/s prograde burn at nominal Gateway perilune"];
initialCondition = ["Initial";"Initial";"Post-impulse"];
finalCondition = ["Final";"Final";"Final"];
departureOrbitID = ["";string(transferRef.dep.orbitID);""];
departureSlot = [NaN;double(transferRef.dep.slot);NaN];
arrivalOrbitID = ["";string(transferRef.arr.orbitID);""];
arrivalSlot = [NaN;double(transferRef.arr.slot);NaN];
deltaV_m_s = [NaN;NaN;impulseInfo.deltaV_m_s];
impulseDirection = ["";"";string(impulseInfo.direction)];
nominalPeriluneEpoch_TU = [NaN;NaN; ...
    impulseInfo.periluneEpochNominal_TU];
transferFinalResidualNorm = [NaN;transferInfo.finalResidualNorm;NaN];

caseMetadata = table( ...
    caseName,targetDefinition,duration_TU,initialCondition,finalCondition, ...
    departureOrbitID,departureSlot,arrivalOrbitID,arrivalSlot, ...
    deltaV_m_s,impulseDirection,nominalPeriluneEpoch_TU, ...
    transferFinalResidualNorm);

metadataFile = fullfile(outputDir,'tracking_case_metadata.csv');
writetable(caseMetadata,metadataFile);

endpointAuditFile = fullfile( ...
    outputDir,'low_thrust_endpoint_reference_audit.csv');
writetable(endpointAudit,endpointAuditFile);

% State conditions are the reproducible scenario definition. Orbit IDs and
% slots above are retained only as catalog provenance for the transfer.
conditionCase = [ ...
    "Lunar Gateway";"Lunar Gateway"; ...
    "Low-thrust transfer";"Low-thrust transfer"; ...
    "Perilune impulse";"Perilune impulse";"Perilune impulse"];

condition = [ ...
    "Initial";"Final"; ...
    "Initial";"Final"; ...
    "Pre-impulse";"Post-impulse";"Final"];

caseEpoch_TU = [ ...
    tGateway(1);tGateway(end); ...
    tTransfer(1);tTransfer(end); ...
    0;0;tImpulse(end)];

referenceEpoch_TU = [ ...
    0;gatewayCfg.period; ...
    NaN;NaN; ...
    impulseInfo.periluneEpochNominal_TU; ...
    impulseInfo.periluneEpochNominal_TU;NaN];

stateND = [ ...
    sGateway(1,:); ...
    sGateway(end,:); ...
    sTransfer(1,:); ...
    sTransfer(end,:); ...
    impulseInfo.statePreImpulse(:).'; ...
    impulseInfo.statePostImpulse(:).'; ...
    sImpulse(end,:)];

stateConditionsND = table( ...
    conditionCase,condition,caseEpoch_TU,referenceEpoch_TU, ...
    stateND(:,1),stateND(:,2),stateND(:,3), ...
    stateND(:,4),stateND(:,5),stateND(:,6), ...
    'VariableNames',{'caseName','condition','caseEpoch_TU', ...
    'referenceEpoch_TU','x_LU','y_LU','z_LU', ...
    'vx_LU_TU','vy_LU_TU','vz_LU_TU'});

stateDimensional = stateND;
stateDimensional(:,1:3) = stateDimensional(:,1:3)*LU;
stateDimensional(:,4:6) = stateDimensional(:,4:6)*VU;

stateConditionsDimensional = table( ...
    conditionCase,condition,caseEpoch_TU,referenceEpoch_TU, ...
    stateDimensional(:,1),stateDimensional(:,2), ...
    stateDimensional(:,3),stateDimensional(:,4), ...
    stateDimensional(:,5),stateDimensional(:,6), ...
    'VariableNames',{'caseName','condition','caseEpoch_TU', ...
    'referenceEpoch_TU','x_km','y_km','z_km', ...
    'vx_km_s','vy_km_s','vz_km_s'});

normalizedStateFile = fullfile( ...
    outputDir,'tracking_case_state_conditions_nd.csv');
dimensionalStateFile = fullfile( ...
    outputDir,'tracking_case_state_conditions_dimensional.csv');
writetable(stateConditionsND,normalizedStateFile);
writetable(stateConditionsDimensional,dimensionalStateFile);

latexRowsFile = fullfile(outputDir,'tracking_case_state_rows.tex');
write_latex_state_rows(latexRowsFile,stateConditionsND);

reproduction = struct();
reproduction.frame = "Earth-Moon barycentric rotating CR3BP";
reproduction.units = struct('position',"LU",'velocity',"LU/TU", ...
    'time',"TU",'LU_km',LU,'TU_s',TU,'VU_km_s',VU,'mu',mu);

reproduction.gateway = struct( ...
    'config',gatewayCfg, ...
    'initialState',sGateway(1,:).', ...
    'finalState',sGateway(end,:).');

reproduction.transfer = struct( ...
    'config',missionCfg.transfer, ...
    'departureOrbitID',departureID, ...
    'departureSlot',transferRef.dep.slot, ...
    'arrivalOrbitID',arrivalID, ...
    'arrivalSlot',transferRef.arr.slot, ...
    'initialState',sTransfer(1,:).', ...
    'finalState',sTransfer(end,:).', ...
    'lambda0',transferInfo.lambda0, ...
    'timeOfFlight_TU',transferInfo.tf, ...
    'finalResidualNorm',transferInfo.finalResidualNorm);

reproduction.impulse = struct( ...
    'config',impulseCfg, ...
    'nominalPeriluneEpoch_TU', ...
        impulseInfo.periluneEpochNominal_TU, ...
    'preImpulseState',impulseInfo.statePreImpulse, ...
    'postImpulseState',impulseInfo.statePostImpulse, ...
    'deltaVVector_LU_TU',impulseInfo.deltaVVector_LU_TU, ...
    'finalState',sImpulse(end,:).');

reproductionFile = fullfile( ...
    outputDir,'tracking_case_reproduction.mat');
save(reproductionFile,'reproduction','-v7');

outputs = struct();
outputs.figure = string(figureFile);
outputs.metadata = string(metadataFile);
outputs.endpointAuditFile = string(endpointAuditFile);
outputs.endpointAudit = endpointAudit;
outputs.normalizedStateConditionsFile = string(normalizedStateFile);
outputs.dimensionalStateConditionsFile = string(dimensionalStateFile);
outputs.latexStateRows = string(latexRowsFile);
outputs.reproductionFile = string(reproductionFile);
outputs.stateConditionsND = stateConditionsND;
outputs.stateConditionsDimensional = stateConditionsDimensional;
outputs.gatewayInfo = gatewayInfo;
outputs.transferInfo = transferInfo;
outputs.impulseInfo = impulseInfo;
outputs.departureIndex = departureIndex;
outputs.arrivalIndex = arrivalIndex;

fprintf('Saved three-case tracking figure to:\n  %s\n',figureFile);
fprintf('Transfer reference resolved to catalog rows %d and %d.\n', ...
    departureIndex,arrivalIndex);
fprintf('\nNormalized initial, maneuver, and final conditions:\n');
disp(stateConditionsND);
end



function orbitDatabase = build_slot_database(times,states,periods,numSlots)

orbitDatabase = cell(numel(periods),1);
for k = 1:numel(periods)
    [uniqueTime,uniqueIndex] = unique(times{k});
    uniqueState = states{k}(uniqueIndex,:);
    interpolant = griddedInterpolant(uniqueTime,uniqueState,'spline');
    slotTime = (0:numSlots-1)'*periods(k)/numSlots;
    orbitDatabase{k} = interpolant(slotTime);
end
end


function prepare_axes(ax)

hold(ax,'on');
box(ax,'on');
axis(ax,'equal');
view(ax,32,24);
ax.Projection = 'orthographic';
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


function inspect_before_export(fig,inspectFigure,description)

if inspectFigure
    figure(fig);
    drawnow;
    input(sprintf( ...
        'Inspect the %s figure, then press Enter to export: ', ...
        description),'s');
end
end


