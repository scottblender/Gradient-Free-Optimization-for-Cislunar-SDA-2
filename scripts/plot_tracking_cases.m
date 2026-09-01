function outputs = plot_tracking_cases()
% Plot the three target scenarios used for the revised optimization study.

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
    'Location','best','FontSize',8);

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
    {'Transfer','Start','End','Moon'},'Location','best','FontSize',8);

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
    'Location','best','FontSize',8);

allAxes = [ax1,ax2,ax3];
set(allAxes,'FontName','Times New Roman','FontSize',10.5, ...
    'FontWeight','bold','LineWidth',1.1,'TickLabelInterpreter','tex');

figureFile = fullfile(outputDir,'tracking_cases.eps');
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
