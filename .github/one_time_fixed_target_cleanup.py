from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")

def write(rel, text):
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def sub_once(text, pattern, repl, label, flags=re.S):
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise RuntimeError(f"{label}: expected one replacement, found {n}")
    return new

# Shared fixed-case configuration helper.
write("src/targetGeneration/target_case_config.m", r'''function missionCfg = target_case_config(missionType)
%TARGET_CASE_CONFIG Convert a stored fixed study case into run configuration.
%
% Fixed target scenarios are defined only in TargetCaseDatabase.mat. The
% observer-orbit catalog is not used to identify target endpoints.

caseDatabase = load_target_case_database();
missionType = upper(string(missionType));

missionCfg = struct();
missionCfg.type = missionType;

switch missionType
    case "LUNAR_GATEWAY"
        c = caseDatabase.gateway;
        missionCfg.gateway = struct( ...
            's0',c.state0(:), ...
            'period',c.period_TU, ...
            'dt',c.dt_TU, ...
            'Nperiods',c.Nperiods);

    case "LOW_THRUST_TRANSFER"
        c = caseDatabase.lowThrust;
        missionCfg.transfer = struct();
        missionCfg.transfer.fixedDepartureState = c.departureState(:).';
        missionCfg.transfer.fixedTargetState = c.arrivalState(:).';
        missionCfg.transfer.dt = c.dt_TU;
        missionCfg.transfer.solverMode = c.solverMode;
        missionCfg.transfer.lowthrust = c.lowthrust;

    case "GATEWAY_IMPULSE"
        c = caseDatabase.gatewayImpulse;
        missionCfg.impulse = struct( ...
            's0',c.nominalGatewayState0(:), ...
            'period',c.nominalPeriod_TU, ...
            'dt',c.dt_TU, ...
            'duration_TU',c.duration_TU, ...
            'deltaV_m_s',c.deltaV_m_s, ...
            'deltaV_LU_TU',c.deltaV_LU_TU, ...
            'direction',c.direction, ...
            'periluneSearchSamples',c.periluneSearchSamples);

    otherwise
        error('Unsupported fixed target case: %s',missionType);
end
end
''')

# Loader: generate the compact fixed-case database automatically if absent.
text = read("src/targetGeneration/load_target_case_database.m")
old = """databasePath = char(databasePath);
assert(isfile(databasePath), ...
    ['Target-case database was not found: %s\\n' ...
     'Run scripts/build_target_case_database.m first.'],databasePath);

S = load(databasePath,'caseDatabase');"""
new = """databasePath = char(databasePath);
if ~isfile(databasePath)
    build_target_case_database(databasePath);
end
assert(isfile(databasePath), ...
    'Target-case database was not created: %s',databasePath);

S = load(databasePath,'caseDatabase');"""
if old not in text:
    raise RuntimeError("target database loader block not found")
text = text.replace(old,new,1)
write("src/targetGeneration/load_target_case_database.m", text)

# Low-thrust solver: fixed states only; no observer catalog properties.
text = read("src/targetGeneration/LowThrustTransferSolver.m")
text = sub_once(text, r'    properties\n.*?    end\n\n    methods', '''    properties
        cfg
        mu
        ode_opts
    end

    methods''', "solver properties")
text = sub_once(text,
    r'        function obj = LowThrustTransferSolver\(cfg, T1, orbit_database, times, states, mu, ode_opts\)\n.*?        end\n\n        function \[t_target, s_target, info\] = solve',
    '''        function obj = LowThrustTransferSolver(cfg, mu, ode_opts)
            obj.cfg = cfg;
            obj.mu = mu;
            obj.ode_opts = ode_opts;
        end

        function [t_target, s_target, info] = solve''', "solver constructor")
text, n = re.subn(
    r'\n            info\.depOrbitIndex\s*=.*?;\n'
    r'            info\.depSlot\s*=.*?;\n'
    r'            info\.arrOrbitIndex\s*=.*?;\n'
    r'            info\.arrSlot\s*=.*?;', '', text, count=1)
if n != 1:
    raise RuntimeError("solver catalog metadata block not found")
text = sub_once(text,
    r'            if isfield\(tr,\'fixedDepartureState\'\).*?            end\n            info\.lambda_f',
    '''            info.endpointDefinition = "FIXED_STATES";
            info.departureStateSource = "FIXED_STATE";
            info.arrivalStateSource = "FIXED_STATE";
            info.lambda_f''', "solver source metadata")
text = sub_once(text,
    r'        function x_dep = getDepartureState\(obj\)\n.*?\n        function jArr = getArrivalSlot\(obj\)\n.*?\n        function x_arr = getArrivalTargetState\(obj\)\n.*?\n        function sigma = getSigma',
    '''        function x_dep = getDepartureState(obj)
            tr = obj.getTransferCfg();
            assert(isfield(tr,'fixedDepartureState') && ...
                ~isempty(tr.fixedDepartureState), ...
                ['Low-thrust transfer requires fixedDepartureState. ' ...
                 'Observer-catalog rows and slots are not target inputs.']);

            x_dep = tr.fixedDepartureState(:);
            if numel(x_dep) ~= 6
                error('transfer.fixedDepartureState must have 6 elements.');
            end
        end

        function x_arr = getArrivalTargetState(obj)
            tr = obj.getTransferCfg();
            assert(isfield(tr,'fixedTargetState') && ...
                ~isempty(tr.fixedTargetState), ...
                ['Low-thrust transfer requires fixedTargetState. ' ...
                 'Observer-catalog rows and slots are not target inputs.']);

            x_arr = tr.fixedTargetState(:);
            if numel(x_arr) ~= 6
                error('transfer.fixedTargetState must have 6 elements.');
            end
        end

        function sigma = getSigma''', "solver fixed endpoint accessors")
write("src/targetGeneration/LowThrustTransferSolver.m", text)

# Target truth dispatcher: low-thrust solver no longer receives catalog.
text = read("src/targetGeneration/build_target_truth.m")
text = sub_once(text,
    r'        solver = LowThrustTransferSolver\( \.\.\.\n'
    r'            missionCfg\.transfer, \.\.\.\n'
    r'            T1, orbit_database, times, states, mu, ode_opts\);',
    '''        solver = LowThrustTransferSolver( ...
            missionCfg.transfer, mu, ode_opts);''', "target truth solver constructor")
write("src/targetGeneration/build_target_truth.m", text)

# run_opt: all fixed targets come from TargetCaseDatabase.
text = read("run_opt.m")
new_switch = r'''switch missionCfg.type

    case "LUNAR_GATEWAY"
        fixedCase = target_case_config("LUNAR_GATEWAY");
        missionCfg.gateway = fixedCase.gateway;

    case "PERIODIC_ORBIT"
        missionCfg.periodic.orbitIndex = 1;
        missionCfg.periodic.dt         = 0.001;
        missionCfg.periodic.Nperiods   = 1;

    case "GATEWAY_IMPULSE"
        fixedCase = target_case_config("GATEWAY_IMPULSE");
        missionCfg.impulse = fixedCase.impulse;

        v = getenv("IMPULSE_DURATION_TU");
        if ~isempty(v)
            missionCfg.impulse.duration_TU = str2double(v);
        end

        v = getenv("IMPULSE_DV_MPS");
        if ~isempty(v)
            missionCfg.impulse.deltaV_m_s = str2double(v);
        end

        v = getenv("IMPULSE_DIRECTION");
        if ~isempty(v)
            missionCfg.impulse.direction = upper(string(v));
        end

        validateattributes(missionCfg.impulse.duration_TU, {'numeric'}, ...
            {'scalar','real','finite','positive'});
        validateattributes(missionCfg.impulse.deltaV_m_s, {'numeric'}, ...
            {'scalar','real','finite','positive'});

        missionCfg.impulse.deltaV_LU_TU = ...
            (missionCfg.impulse.deltaV_m_s / 1000) / VU;

    case "LOW_THRUST_TRANSFER"
        fixedCase = target_case_config("LOW_THRUST_TRANSFER");
        missionCfg.transfer = fixedCase.transfer;

    otherwise
        error("Unknown MISSION_TYPE: %s", missionCfg.type);
end

% ---------------- Override number of periods'''
text = sub_once(text, r'switch missionCfg\.type\n.*?\nend\n\n% ---------------- Override number of periods', new_switch, "run_opt mission switch")

new_truth_block = r'''% ---------------- Build/load target truth ----------------
useTransferCache = true;

if missionCfg.type == "LOW_THRUST_TRANSFER" && useTransferCache
    cacheKey = make_transfer_cache_key(missionCfg);
    cacheKey = cacheKey + "_" + study_hash({missionCfg.transfer,mu});
    cacheFile = fullfile(TransferCacheDir, cacheKey + ".mat");

    loadedFromCache = false;

    if isfile(cacheFile)
        try
            safe_printf('Loading cached transfer truth from:\n  %s\n', cacheFile);
            C = load(cacheFile, 't_target', 's_target', 'truthInfo', 'cacheMeta');
            t_target  = C.t_target;
            s_target  = C.s_target;
            truthInfo = C.truthInfo;
            if isfield(C, 'cacheMeta')
                safe_printf('Cached transfer key: %s\n', string(C.cacheMeta.cacheKey));
            end
            loadedFromCache = true;
        catch ME
            safe_printf(2, 'WARNING: failed to load transfer cache, rebuilding: %s\n', ME.message);
            try
                delete(cacheFile);
            catch
            end
        end
    end

    if ~loadedFromCache
        safe_printf('No valid cached transfer found. Computing transfer truth.\n');
        [t_target, s_target, truthInfo] = build_target_truth( ...
            missionCfg, T1, orbit_database, times, states, mu, ode_opts);

        cacheMeta = struct();
        cacheMeta.cacheKey    = cacheKey;
        cacheMeta.missionType = string(missionCfg.type);
        cacheMeta.created     = string(datetime('now'));
        cacheMeta.mu          = mu;
        tmpFile = fullfile(TransferCacheDir, cacheKey + "_t.mat");

        try
            if isfile(tmpFile), delete(tmpFile); end
            if isfile(cacheFile), delete(cacheFile); end
            save(tmpFile, 't_target', 's_target', 'truthInfo', 'cacheMeta', '-v7.3');
            movefile(tmpFile, cacheFile, 'f');
            safe_printf('Saved transfer truth cache to:\n  %s\n', cacheFile);
        catch ME
            safe_printf(2, 'WARNING: failed to save transfer cache: %s\n', ME.message);
            try
                if isfile(tmpFile), delete(tmpFile); end
            catch
            end
        end
    end
else
    [t_target, s_target, truthInfo] = build_target_truth( ...
        missionCfg, T1, orbit_database, times, states, mu, ode_opts);
end

% ---------------- Moon impact check ----------------'''
text = sub_once(text, r'% ---------------- Build/load target truth ----------------.*?% ---------------- Moon impact check ----------------', new_truth_block, "run_opt target truth block")
text = sub_once(text, r'hDepOrb = gobjects\(0\);.*?\nR_moon = 1737\.1 / LU;', r'''hStart  = gobjects(0);
hEnd    = gobjects(0);

isTransferMission = missionCfg.type == "LOW_THRUST_TRANSFER";

if isTransferMission
    hStart = plot3(ax, s_truth(1,1), s_truth(1,2), s_truth(1,3), 'o', ...
        'MarkerSize', 9, 'MarkerFaceColor', [0.85 0.27 0.22], ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.0);
    hEnd = plot3(ax, s_truth(end,1), s_truth(end,2), s_truth(end,3), 's', ...
        'MarkerSize', 9, 'MarkerFaceColor', [0.27 0.31 0.86], ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.0);
end

R_moon = 1737.1 / LU;''', "run_opt legacy transfer plotting")
text = sub_once(text,
    r'if isTransferMission\n    legHandles = \[legHandles; hDepOrb; hArrOrb; hStart; hEnd\];\n    legLabels  = \[legLabels; \{\'Departure orbit\'; \'Arrival orbit\'; \'Transfer start\'; \'Transfer end\'\}\];\nend',
    '''if isTransferMission
    legHandles = [legHandles; hStart; hEnd];
    legLabels  = [legLabels; {'Transfer start'; 'Transfer end'}];
end''', "run_opt transfer legend")
new_cache_fun = r'''function cacheKey = make_transfer_cache_key(missionCfg)
    tr = missionCfg.transfer;
    lt = tr.lowthrust;
    endpointHash = string(study_hash({tr.fixedDepartureState,tr.fixedTargetState}));
    shortHash = extractBefore(endpointHash, min(9,strlength(endpointHash)+1));

    cacheKey = sprintf('lt_fixed_%s_dt%s_tf%s', ...
        char(shortHash), ...
        local_num_str(get_field_or_default(tr, 'dt', 0)), ...
        local_num_str(get_field_or_default(lt, 'tf_guess', 0)));

    cacheKey = regexprep(cacheKey, '[^A-Za-z0-9_]', '_');
end

function v = get_field_or_default'''
text = sub_once(text, r'function cacheKey = make_transfer_cache_key\(missionCfg, slots_per_orbit\).*?\nfunction v = get_field_or_default', new_cache_fun, "run_opt transfer cache helper")
text = sub_once(text, r'\nfunction \[index, stateError\] = find_transfer_state_match\(T, referenceState\).*?\nfunction \[xL1, xL2\] = cr3bp_L1L2', '\nfunction [xL1, xL2] = cr3bp_L1L2', "run_opt obsolete endpoint matcher")
write("run_opt.m", text)

# Study-definition plots: target plots are catalog-independent.
text = read("scripts/plot_study_definition_figures.m")
new_tracking = r'''function outputs = create_tracking_cases(inspectFigure)
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
hTransfer = plot3(ax,sTransfer(:,1),sTransfer(:,2),sTransfer(:,3),'-','Color',cTransfer,'LineWidth',3.0);
hStart = plot3(ax,sTransfer(1,1),sTransfer(1,2),sTransfer(1,3),'o','MarkerSize',9,'MarkerFaceColor',cGateway,'MarkerEdgeColor','k','LineWidth',1.2);
hEnd = plot3(ax,sTransfer(end,1),sTransfer(end,2),sTransfer(end,3),'s','MarkerSize',9,'MarkerFaceColor',cTransfer,'MarkerEdgeColor','k','LineWidth',1.2);
hMoon = draw_moon(ax,mu,LU);
hL1 = plot3(ax,xL1,0,0,'^','MarkerFaceColor',cPoint,'MarkerEdgeColor','k','MarkerSize',7,'LineWidth',1.0);
hL2 = plot3(ax,xL2,0,0,'v','MarkerFaceColor',cPoint,'MarkerEdgeColor','k','MarkerSize',7,'LineWidth',1.0);
format_case_axes(ax);
legendHandle = legend(ax,[hTransfer,hStart,hEnd,hMoon,hL1,hL2],{'Transfer','Start','End','Moon','L1','L2'},'Location','northoutside','Orientation','horizontal');
format_case_legend(legendHandle,3); axis(ax,'tight'); axis(ax,'vis3d'); format_case_axes(ax);
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


function prepare_axes(ax)'''
text = sub_once(text, r'function outputs = create_tracking_cases\(inspectFigure\).*?\nfunction prepare_axes\(ax\)', new_tracking, "study figure target cases")
write("scripts/plot_study_definition_figures.m", text)

# Baseline result processing: reconstruct fixed cases from shared helper.
text = read("scripts/process_baseline_results.m")
new_build_mission = r'''function missionCfg = buildMissionCfg(runInfo)

switch lower(string(runInfo.mission))
    case "lg"
        missionCfg = target_case_config("LUNAR_GATEWAY");
        if isfinite(runInfo.periods) && runInfo.periods > 0
            missionCfg.gateway.Nperiods = round(runInfo.periods);
        end
    case "lt"
        missionCfg = target_case_config("LOW_THRUST_TRANSFER");
    otherwise
        error("Cannot determine mission type for run %s.", runInfo.runName);
end

missionCfg.optimization.numObservers = round(runInfo.numObservers);
end

function [t_target, s_target, truthInfo] = buildOrLoadTargetTruth'''
text = sub_once(text, r'function missionCfg = buildMissionCfg\(runInfo\).*?\nfunction \[t_target, s_target, truthInfo\] = buildOrLoadTargetTruth', new_build_mission, "baseline fixed mission config")
text = text.replace("cacheKey  = make_transfer_cache_key(missionCfg, baseCtx.slots_per_orbit);", "cacheKey  = make_transfer_cache_key(missionCfg);")
new_baseline_cache = r'''function cacheKey = make_transfer_cache_key(missionCfg)

tr = missionCfg.transfer;
lt = tr.lowthrust;
endpointHash = string(study_hash({tr.fixedDepartureState,tr.fixedTargetState}));
shortHash = extractBefore(endpointHash, min(9,strlength(endpointHash)+1));

cacheKey = sprintf('lt_fixed_%s_dt%s_tf%s', ...
    char(shortHash), ...
    local_num_str(get_field_or_default(tr, 'dt', 0)), ...
    local_num_str(get_field_or_default(lt, 'tf_guess', 0)));

cacheKey = regexprep(cacheKey, '[^A-Za-z0-9_]', '_');
end

function v = get_field_or_default'''
text = sub_once(text, r'function cacheKey = make_transfer_cache_key\(missionCfg, slots_per_orbit\).*?\nfunction v = get_field_or_default', new_baseline_cache, "baseline transfer cache helper")
write("scripts/process_baseline_results.m", text)

# Visibility test: observers keep slots; targets do not.
text = read("tests/test_visibility_trajectories.m")
text, n = re.subn(r'\nassert\(num_orbits >= 400, \.\.\.\n    \'The catalog must contain departure orbit 52 and arrival orbit 400\.\'\);\n', '\n', text, count=1)
if n != 1:
    raise RuntimeError("visibility legacy catalog size assertion not found")
text = sub_once(text, r'missionCfg = struct\(\);\nmissionCfg\.type = "LUNAR_GATEWAY";\n.*?missionCfg\.gateway\.Nperiods = gateway_periods;', 'missionCfg = target_case_config("LUNAR_GATEWAY");\nmissionCfg.gateway.Nperiods = gateway_periods;', "visibility gateway config")
text = sub_once(text, r'% ---------------- Generate low-thrust truth ----------------\n.*?\n\[t_transfer, s_transfer, transferInfo\] = build_target_truth', '''% ---------------- Generate low-thrust truth ----------------
% Fixed boundary states come from TargetCaseDatabase.mat.

missionCfg = target_case_config("LOW_THRUST_TRANSFER");

fprintf('\\nGenerating fixed-boundary low-thrust transfer.\\n');

[t_transfer, s_transfer, transferInfo] = build_target_truth''', "visibility low thrust config")
text = text.replace("% These are observers, separate from the transfer endpoints below.", "% These are observer candidates; target cases are loaded independently.")
write("tests/test_visibility_trajectories.m", text)

# Low-thrust integration test: validate fixed database states, not catalog.
write("tests/test_low_thrust_transfer_case.m", r'''function summary = test_low_thrust_transfer_case()
% End-to-end 120-FE pilot for the fixed-boundary low-thrust transfer.

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
setup_project();

missionCfg = target_case_config("LOW_THRUST_TRANSFER");
transferCfg = missionCfg.transfer;

assert(numel(transferCfg.fixedDepartureState)==6 && numel(transferCfg.fixedTargetState)==6, ...
    'Fixed low-thrust boundary states must each have six elements.');
assert(all(isfinite(transferCfg.fixedDepartureState)) && all(isfinite(transferCfg.fixedTargetState)), ...
    'Fixed low-thrust boundary states contain nonfinite values.');

fprintf('\n--- Low-thrust transfer pilot ---\n');
fprintf('Target definition: fixed departure and arrival states.\n');
summary = test_small_fe_case("GA", 0, "LOW_THRUST_TRANSFER");
assert(height(summary) == 1, 'Expected exactly one pilot result.');

checkNames = ["searchBudgetOK";"solverCallPatternOK";"historyOK";"bestOK";"recheckOK"];
for k = 1:numel(checkNames)
    assert(all(logical(summary.(char(checkNames(k))))), 'Pilot check failed: %s.', checkNames(k));
end
assert(string(summary.termination(1)) == "budget_reached", 'The pilot did not terminate at the FE budget.');

resultPath = fullfile(char(summary.runDir(1)), 'data', 'optimization_run.mat');
assert(isfile(resultPath), 'Pilot result file was not found: %s', resultPath);
saved = load(resultPath, 'runState'); runState = saved.runState; info = runState.truthInfo;

assert(string(info.type) == "LOW_THRUST_TRANSFER", 'The saved truth is not a low-thrust transfer.');
assert(string(info.endpointDefinition) == "FIXED_STATES", 'The transfer truth is not identified as a fixed-state case.');
assert(string(info.departureStateSource) == "FIXED_STATE" && string(info.arrivalStateSource) == "FIXED_STATE", 'The transfer solver did not use fixed boundary states.');
assert(norm(info.x_dep(:)-transferCfg.fixedDepartureState(:)) <= 1e-12, 'The solver departure state differs from TargetCaseDatabase.');
assert(norm(info.x_arr(:)-transferCfg.fixedTargetState(:)) <= 1e-12, 'The solver arrival state differs from TargetCaseDatabase.');
assert(info.exitflag > 0, 'The low-thrust fsolve call did not converge.');
assert(isfinite(info.finalResidualNorm) && info.finalResidualNorm <= 1e-8, 'Low-thrust final residual norm is too large: %.6e', info.finalResidualNorm);
assert(isfinite(info.tf) && info.tf > 0, 'Low-thrust transfer time is invalid.');
assert(isfinite(info.mass_final) && info.mass_final > 0, 'Low-thrust final mass is invalid.');

fprintf('\n--- Low-thrust results ---\n');
fprintf('Solver exit flag:       %d\n', info.exitflag);
fprintf('Final residual norm:     %.6e\n', info.finalResidualNorm);
fprintf('Transfer time:           %.6f TU\n', info.tf);
fprintf('Final mass:              %.6f\n', info.mass_final);
fprintf('Search evaluations:      %d\n', summary.searchFE(1));
fprintf('Best objective:          %.12g\n', runState.bestJ);
fprintf('\nLow-thrust transfer pilot passed.\n');
end
''')

# Gateway impulse truth test: use shared fixed-case definition.
text = read("tests/test_gateway_impulse_truth.m")
text = sub_once(text, r'cfg = struct\(\);\ncfg\.s0 = .*?cfg\.periluneSearchSamples = 4001;', 'missionCfg = target_case_config("GATEWAY_IMPULSE");\ncfg = missionCfg.impulse;', "impulse truth fixed config")
write("tests/test_gateway_impulse_truth.m", text)

# Replace old mixed catalog/transfer audit with observer-catalog-only test.
old_catalog_test = ROOT / "tests/test_catalog_dro_and_transfer.m"
if old_catalog_test.exists(): old_catalog_test.unlink()
write("tests/test_observer_catalog.m", r'''function audit = test_observer_catalog()
%TEST_OBSERVER_CATALOG Validate the filtered 450-orbit observer database.

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir); paths = setup_project();
assert(isfile(paths.catalog), 'Observer catalog was not found: %s',paths.catalog);
S = load(paths.catalog,'T'); T = S.T;
required = ["orbitFamily";"orbitID";"state";"time";"periluneAltitude_km";"apoluneAltitude_km";"xAmplitude_LU";"zAmplitude_LU";"jacobiConstant";"jacobiVariation";"period_TU";"stability";"manuscriptFamily";"region"];
names = string(T.Properties.VariableNames);
for k = 1:numel(required), assert(any(names==required(k)), 'Observer catalog is missing %s.',required(k)); end
assert(height(T)==450, 'Expected 450 observer orbits, found %d.',height(T));
assert(numel(unique(string(T.orbitID)))==height(T), 'Observer orbit IDs are not unique.');
assert(all(T.periluneAltitude_km >= -1e-3), 'At least one selected observer orbit intersects the Moon.');
assert(all(T.apoluneAltitude_km >= T.periluneAltitude_km), 'At least one observer orbit has invalid peri/apo geometry.');
assert(all(isfinite(T.jacobiConstant)), 'A selected observer orbit has a nonfinite Jacobi constant.');
assert(all(T.jacobiVariation <= 1e-8), 'Jacobi variation exceeds the catalog audit tolerance.');
familyOrder = ["NHO";"SHO";"NNRHO";"SNRHO";"NHO";"SHO";"NNRHO";"SNRHO";"DRO"];
regionOrder = ["L1";"L1";"L1";"L1";"L2";"L2";"L2";"L2";"--"];
counts = zeros(numel(familyOrder),1);
for k = 1:numel(familyOrder)
    use = string(T.manuscriptFamily)==familyOrder(k) & string(T.region)==regionOrder(k);
    counts(k) = nnz(use);
    assert(counts(k)==50, 'Expected 50 %s %s observer orbits, found %d.',familyOrder(k),regionOrder(k),counts(k));
end
isDRO = string(T.manuscriptFamily)=="DRO";
assert(all(T.stability(isDRO) <= 1+1e-8), 'A selected DRO exceeds the stability threshold.');
audit = struct('numOrbits',height(T),'familyCounts',counts,'maxJacobiVariation',max(T.jacobiVariation),'jacobiRange',[min(T.jacobiConstant),max(T.jacobiConstant)]);
fprintf('\nObserver catalog audit passed.\n');
fprintf('Selected observer orbits: %d\n',audit.numOrbits);
fprintf('Maximum Jacobi variation: %.6e\n',audit.maxJacobiVariation);
end
''')

# Structure test knows the new shared configuration and observer audit.
text = read("tests/test_project_structure.m")
old = "        'load_target_case_database', 'src/targetGeneration/load_target_case_database.m'\n"
if old in text and "'target_case_config'" not in text:
    text = text.replace(old, old + "        'target_case_config', 'src/targetGeneration/target_case_config.m'\n", 1)
old2 = "        'test_visibility_trajectories', 'tests/test_visibility_trajectories.m'\n"
if old2 in text and "'test_observer_catalog'" not in text:
    text = text.replace(old2, "        'test_observer_catalog', 'tests/test_observer_catalog.m'\n" + old2, 1)
write("tests/test_project_structure.m", text)

# Documentation.
text = read("README.md")
text = text.replace("`load_and_filter_data` is only needed to rebuild the catalog from raw JPL CSV\nfiles.", "`build_observer_orbit_catalog` is only needed to rebuild the observer catalog from raw JPL CSV\nfiles.")
text = text.replace("| Orbit catalog | `data/JPL_CR3BP_OrbitCatalog.mat` |", "| Observer orbit catalog | `data/JPL_CR3BP_OrbitCatalog.mat` |\n| Fixed target cases | `data/TargetCaseDatabase.mat` |")
if "Fixed Gateway, low-thrust, and Gateway-impulse target definitions" not in text:
    text += "\nFixed Gateway, low-thrust, and Gateway-impulse target definitions are stored in `data/TargetCaseDatabase.mat`. The loader creates this compact database from `scripts/build_target_case_database.m` when it is missing. Fixed target cases never resolve observer catalog rows or observer slots.\n"
write("README.md", text)

text = read("data/README.md")
text = text.replace("Optional raw orbit CSV files belong in `JPL_Data/`. Only run\n`scripts/load_and_filter_data.m` when you intend to rebuild the catalog.", "Optional raw orbit CSV files belong in `JPL_Data/`. Only run\n`scripts/build_observer_orbit_catalog.m` when you intend to rebuild the observer catalog.\n\n`TargetCaseDatabase.mat` contains the fixed Gateway, low-thrust, and impulse\nstudy-case definitions. It is generated by `scripts/build_target_case_database.m`\nand does not depend on observer-orbit row or slot selections.")
write("data/README.md", text)

# Delete obsolete target-provenance helper.
obsolete = ROOT / "src/targetGeneration/low_thrust_case_config.m"
if obsolete.exists(): obsolete.unlink()

# Repository-wide audit. Observer design slots are intentionally allowed;
# only fixed-target endpoint provenance/coupling is forbidden.
forbidden = [
    "low_thrust_case_config", "depOrbitIndex", "arrOrbitIndex", "depOrbitID", "arrOrbitID",
    "depSlotStateError", "arrSlotStateError", "legacyCatalogRow", "resolvedCatalogRow",
    "legacyIndex", "newIndex", "slotStateReferenceError", "fixedVsObserverSlotError",
    "endpointAudit", "transferRef", "old row 52", "old row-52", "old row 400", "old row-400",
    "Departure: orbit 52", "Arrival:   orbit 400"
]
hits = []
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts: continue
    if path.suffix.lower() not in {".m",".md",".ps1",".py",".yml",".yaml",".txt"}: continue
    if path == Path(__file__).resolve(): continue
    content = path.read_text(encoding="utf-8", errors="ignore")
    for token in forbidden:
        if token.lower() in content.lower(): hits.append(f"{path.relative_to(ROOT)}: {token}")
if hits:
    print("Forbidden fixed-target catalog provenance remains:", file=sys.stderr)
    print("\n".join(hits), file=sys.stderr)
    raise SystemExit(2)
if (ROOT / "tests/test_catalog_dro_and_transfer.m").exists():
    raise SystemExit("obsolete mixed catalog/transfer test still exists")

# Remove this one-time automation from the final cleanup commit.
for rel in [".github/one_time_fixed_target_cleanup.py", ".github/workflows/one-time-fixed-target-cleanup.yml"]:
    p = ROOT / rel
    if p.exists(): p.unlink()

print("Fixed-target cleanup and repository-wide provenance audit passed.")
