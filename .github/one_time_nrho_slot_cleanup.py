from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'scripts/plot_study_definition_figures.m'
text = path.read_text(encoding='utf-8')

old_select = '''family = string(T.orbitFamily);\norbitIndex = find(family=="NHL1",1,'first');\nif isempty(orbitIndex)\n    orbitIndex = find(family=="NNRHL1",1,'first');\nend\nassert(~isempty(orbitIndex), ...\n    'No representative northern L1 orbit was found.');'''
new_select = '''family = string(T.orbitFamily);\norbitIndex = find(family=="NNRHL1",1,'first');\nassert(~isempty(orbitIndex), ...\n    'No representative northern NRHO L1 orbit was found.');'''
if old_select not in text:
    raise RuntimeError('Expected slot representative selection block was not found.')
text = text.replace(old_select, new_select, 1)

old_reference = '''mu = 1.215058560962404E-2;\n[xL1,xL2] = cr3bp_L1L2(mu);\n\nrepresentativeFamily = family(orbitIndex);\nif endsWith(representativeFamily,"L1")\n    lagrangeX = xL1;\n    lagrangeMarker = '^';\n    lagrangeLabel = "L1";\nelseif endsWith(representativeFamily,"L2")\n    lagrangeX = xL2;\n    lagrangeMarker = 'v';\n    lagrangeLabel = "L2";\nelse\n    error('Representative slot orbit must belong to an L1 or L2 family: %s', ...\n        representativeFamily);\nend'''
new_reference = '''mu = 1.215058560962404E-2;\nLU = 384400;'''
if old_reference not in text:
    raise RuntimeError('Expected family-aware Lagrange reference block was not found.')
text = text.replace(old_reference, new_reference, 1)

old_plot = '''hLagrange = plot3(ax,lagrangeX,0,0,lagrangeMarker,'MarkerSize',9, ...\n    'MarkerFaceColor',[0.80,0.80,0.80], ...\n    'MarkerEdgeColor','k','LineWidth',1.2);'''
new_plot = '''hMoon = draw_moon(ax,mu,LU);'''
if old_plot not in text:
    raise RuntimeError('Expected Lagrange marker plot block was not found.')
text = text.replace(old_plot, new_plot, 1)

old_legend = '''legendHandle = legend(ax,[hOrbit,hSlots,hSelected,hNext,hLagrange], ...\n    {'Orbit','Candidate slots','Slot j','Slot j+1',char(lagrangeLabel)}, ...'''
new_legend = '''legendHandle = legend(ax,[hOrbit,hSlots,hSelected,hNext,hMoon], ...\n    {'Orbit','Candidate slots','Slot j','Slot j+1','Moon'}, ...'''
if old_legend not in text:
    raise RuntimeError('Expected slot geometry legend block was not found.')
text = text.replace(old_legend, new_legend, 1)

for forbidden in ['find(family=="NHL1"', 'hLagrange', 'lagrangeLabel', 'lagrangeMarker', 'lagrangeX']:
    # Scope is intentionally file-wide because these names were introduced solely for slot geometry.
    if forbidden in text:
        raise RuntimeError(f'Obsolete slot geometry reference remains: {forbidden}')

if 'find(family=="NNRHL1",1,\'first\')' not in text:
    raise RuntimeError('Strict NNRHL1 representative selection is missing.')
if 'hMoon = draw_moon(ax,mu,LU);' not in text:
    raise RuntimeError('Moon was not restored to the slot geometry figure.')

path.write_text(text, encoding='utf-8')

for rel in [
    '.github/one_time_nrho_slot_cleanup.py',
    '.github/workflows/one-time-nrho-slot-cleanup.yml',
]:
    p = ROOT / rel
    if p.exists():
        p.unlink()

print('Slot geometry now uses a northern NRHO L1 orbit with the Moon and no Lagrange-point marker.')
