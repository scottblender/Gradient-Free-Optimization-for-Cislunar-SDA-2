from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / 'scripts' / 'plot_study_definition_figures.m'
text = path.read_text(encoding='utf-8')

old_truth = '''transferCfg = target_case_config("LOW_THRUST_TRANSFER");
[tTransfer,sTransfer,transferInfo] = build_target_truth( ...
    transferCfg,table(),{}, {}, {},mu,odeOptions);

impulseCfg = target_case_config("GATEWAY_IMPULSE");'''
new_truth = '''transferCfg = target_case_config("LOW_THRUST_TRANSFER");
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

impulseCfg = target_case_config("GATEWAY_IMPULSE");'''
if old_truth not in text:
    raise SystemExit('Could not find LT truth block')
text = text.replace(old_truth,new_truth,1)

old_colors = '''cImpulse = [0.55,0.30,0.72];
cNominal = [0.35,0.35,0.35];
cPoint = [0.80,0.80,0.80];'''
new_colors = '''cImpulse = [0.55,0.30,0.72];
cNominal = [0.35,0.35,0.35];
cReference = [0.48,0.48,0.48];
cPoint = [0.80,0.80,0.80];'''
if old_colors not in text:
    raise SystemExit('Could not find case colors block')
text = text.replace(old_colors,new_colors,1)

old_plot = '''figTransfer = publication_figure(7.2,6.5);
ax = axes(figTransfer); prepare_axes(ax);
hTransfer = plot3(ax,sTransfer(:,1),sTransfer(:,2),sTransfer(:,3),'-','Color',cTransfer,'LineWidth',3.0);
hStart = plot3(ax,sTransfer(1,1),sTransfer(1,2),sTransfer(1,3),'o','MarkerSize',9,'MarkerFaceColor',cGateway,'MarkerEdgeColor','k','LineWidth',1.2);
hEnd = plot3(ax,sTransfer(end,1),sTransfer(end,2),sTransfer(end,3),'s','MarkerSize',9,'MarkerFaceColor',cTransfer,'MarkerEdgeColor','k','LineWidth',1.2);
hMoon = draw_moon(ax,mu,LU);
hL1 = plot3(ax,xL1,0,0,'^','MarkerFaceColor',cPoint,'MarkerEdgeColor','k','MarkerSize',7,'LineWidth',1.0);
hL2 = plot3(ax,xL2,0,0,'v','MarkerFaceColor',cPoint,'MarkerEdgeColor','k','MarkerSize',7,'LineWidth',1.0);
format_case_axes(ax);
legendHandle = legend(ax,[hTransfer,hStart,hEnd,hMoon,hL1,hL2],{'Transfer','Start','End','Moon','L1','L2'},'Location','northoutside','Orientation','horizontal');
format_case_legend(legendHandle,3); axis(ax,'tight'); axis(ax,'vis3d'); format_case_axes(ax);'''
new_plot = '''figTransfer = publication_figure(7.2,6.5);
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
format_case_legend(legendHandle,4); axis(ax,'tight'); axis(ax,'vis3d'); format_case_axes(ax);'''
if old_plot not in text:
    raise SystemExit('Could not find LT plot block')
text = text.replace(old_plot,new_plot,1)

anchor = '''function prepare_axes(ax)
'''
helper = '''function orbitState = find_reference_orbit_for_state(T,targetState)
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
'''
if anchor not in text:
    raise SystemExit('Could not find prepare_axes anchor')
text = text.replace(anchor,helper,1)

# Guardrails: this restoration must not reintroduce target slot/row provenance.
for forbidden in ['depSlot','arrSlot','depOrbitIndex','arrOrbitIndex','legacyCatalogRow','resolvedCatalogRow','endpointAudit']:
    if forbidden.lower() in text.lower():
        raise SystemExit(f'Forbidden target provenance token reintroduced: {forbidden}')

path.write_text(text,encoding='utf-8')

# Remove one-time patch files from the final tree.
for rel in ['.github/restore_lt_endpoint_orbits.py','.github/workflows/restore-lt-endpoint-orbits.yml']:
    p = root / rel
    if p.exists():
        p.unlink()

print('Restored LT endpoint orbits as plot-only visual context.')
