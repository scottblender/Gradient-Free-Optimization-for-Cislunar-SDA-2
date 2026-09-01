from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'scripts/plot_study_definition_figures.m'
text = path.read_text(encoding='utf-8')

old = '''mu = 1.215058560962404E-2;
LU = 384400;
[xL1,~] = cr3bp_L1L2(mu);

% Figure 1: the orbit and its 50 equal-time candidate states.'''
new = '''mu = 1.215058560962404E-2;
[xL1,xL2] = cr3bp_L1L2(mu);

representativeFamily = family(orbitIndex);
if endsWith(representativeFamily,"L1")
    lagrangeX = xL1;
    lagrangeMarker = '^';
    lagrangeLabel = "L1";
elseif endsWith(representativeFamily,"L2")
    lagrangeX = xL2;
    lagrangeMarker = 'v';
    lagrangeLabel = "L2";
else
    error('Representative slot orbit must belong to an L1 or L2 family: %s', ...
        representativeFamily);
end

% Figure 1: the orbit and its 50 equal-time candidate states.'''
if old not in text:
    raise RuntimeError('Could not find slot Lagrange-point setup block.')
text = text.replace(old, new, 1)

old = '''hNext = plot3(ax,slotState(nextSlot,1), ...
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
    'FontName','Times New Roman','FontSize',15,'FontWeight','bold');'''
new = '''hNext = plot3(ax,slotState(nextSlot,1), ...
    slotState(nextSlot,2),slotState(nextSlot,3), ...
    's','MarkerSize',9,'MarkerFaceColor',nextColor, ...
    'MarkerEdgeColor','k','LineWidth',1.2);
hLagrange = plot3(ax,lagrangeX,0,0,lagrangeMarker,'MarkerSize',9, ...
    'MarkerFaceColor',[0.80,0.80,0.80], ...
    'MarkerEdgeColor','k','LineWidth',1.2);

set(ax,'FontName','Times New Roman','FontSize',18, ...
    'FontWeight','bold','LineWidth',1.8);
legendHandle = legend(ax,[hOrbit,hSlots,hSelected,hNext,hLagrange], ...
    {'Orbit','Candidate slots','Slot j','Slot j+1',char(lagrangeLabel)}, ...
    'Location','northoutside','Orientation','horizontal', ...
    'FontName','Times New Roman','FontSize',15,'FontWeight','bold');'''
if old not in text:
    raise RuntimeError('Could not find slot geometry Moon/L1 plotting block.')
text = text.replace(old, new, 1)

if 'hMoon = draw_moon(ax,mu,LU);' in text[text.index('function outputs = create_slot_definition'):text.index('% Figure 2: the exact normalized phase grid')]:
    raise RuntimeError('Moon plotting remains in slot geometry figure.')

path.write_text(text, encoding='utf-8')

for rel in [
    '.github/one_time_slot_lagrange_cleanup.py',
    '.github/workflows/one-time-slot-lagrange-cleanup.yml',
]:
    p = ROOT / rel
    if p.exists():
        p.unlink()

print('Removed Moon from slot geometry and made Lagrange marker family-aware.')
