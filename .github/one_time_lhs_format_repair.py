from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'scripts/private/observer_catalog_filter_core.m'
text = path.read_text(encoding='utf-8')

old1 = """    fprintf('%s candidates: %d eligible; selected %d by LHS over %s.\\n', ...
        familyName, numel(familyIdx), numel(take), sampleCoordinate);"""
bad1 = """    fprintf('%s candidates: %d eligible; selected %d by LHS over %s.
', ...
        familyName, numel(familyIdx), numel(take), sampleCoordinate);"""
if bad1 not in text:
    raise RuntimeError('Malformed family LHS fprintf string was not found.')
text = text.replace(bad1, old1, 1)

old2 = """fprintf(\"Keeping %d total orbits after all-family LHS selection.\\n\", ...
    nnz(keepMask));"""
bad2 = """fprintf(\"Keeping %d total orbits after all-family LHS selection.
\", ...
    nnz(keepMask));"""
if bad2 not in text:
    raise RuntimeError('Malformed total LHS fprintf string was not found.')
text = text.replace(bad2, old2, 1)

if bad1 in text or bad2 in text:
    raise RuntimeError('Malformed LHS fprintf strings remain after repair.')

path.write_text(text, encoding='utf-8')

for rel in [
    '.github/one_time_lhs_format_repair.py',
    '.github/workflows/one-time-lhs-format-repair.yml',
]:
    p = ROOT / rel
    if p.exists():
        p.unlink()

print('Repaired LHS MATLAB fprintf escape sequences.')
