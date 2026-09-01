from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'scripts/plot_study_definition_figures.m'
text = path.read_text(encoding='utf-8')

text = text.replace(
    'numPairOrbitsPerFamily = 16;\nnumDroOrbits = 16;',
    'numPairOrbitsPerFamily = 10;\nnumDroOrbits = 10;',
    1,
)

old_pair = '''            selectedRows = select_evenly_spaced_rows( ...\n                familyRows,T.zAmplitude(familyRows),perFamily);'''
new_pair = '''            assert(numel(familyRows)==50, ...\n                'Expected 50 selected orbits for %s, found %d.', ...\n                group(member),numel(familyRows));\n            plotStride = numel(familyRows)/perFamily;\n            assert(plotStride==round(plotStride), ...\n                'Orbit plotting stride must be an integer.');\n            selectedRows = familyRows(plotStride:plotStride:end);'''
if old_pair not in text:
    raise RuntimeError('Paired-family selection block not found.')
text = text.replace(old_pair, new_pair, 1)

old_dro = '''        selectedRows = unique(round(linspace( ...\n            1,numel(familyRows),min(numDroOrbits,numel(familyRows)))));'''
new_dro = '''        assert(numel(familyRows)==50, ...\n            'Expected 50 selected DROs, found %d.',numel(familyRows));\n        plotStride = numel(familyRows)/numDroOrbits;\n        assert(plotStride==round(plotStride), ...\n            'DRO plotting stride must be an integer.');\n        selectedRows = plotStride:plotStride:numel(familyRows);'''
if old_dro not in text:
    raise RuntimeError('DRO selection block not found.')
text = text.replace(old_dro, new_dro, 1)

# The old amplitude-target helper is no longer used; remove it to keep the
# plotting definition explicit and avoid accidental re-use.
start = text.find('function selectedRows = select_evenly_spaced_rows(')
if start == -1:
    raise RuntimeError('Obsolete helper not found.')
end_marker = '\n\n\nfunction format_case_axes(ax)'
end = text.find(end_marker, start)
if end == -1:
    raise RuntimeError('Could not locate end of obsolete helper.')
text = text[:start] + 'function format_case_axes(ax)' + text[end + len(end_marker):]

# Static checks.
if 'numPairOrbitsPerFamily = 16' in text or 'numDroOrbits = 16' in text:
    raise RuntimeError('A 16-orbit plotting constant remains.')
if 'select_evenly_spaced_rows' in text:
    raise RuntimeError('Obsolete evenly-spaced helper remains.')
if 'numPairOrbitsPerFamily = 10' not in text or 'numDroOrbits = 10' not in text:
    raise RuntimeError('Ten-orbit plotting constants were not installed.')

path.write_text(text, encoding='utf-8')

for rel in [
    '.github/one_time_ten_orbit_plot.py',
    '.github/workflows/one-time-ten-orbit-plot.yml',
]:
    p = ROOT / rel
    if p.exists():
        p.unlink()

print('Updated orbit-family figures to plot every fifth member (10 of 50).')
