from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

# Repair any literal newline that was accidentally inserted inside a
# single-quoted format string supplied directly to fprintf/safe_printf.
call_pattern = re.compile(
    r"(?P<prefix>(?:fprintf|safe_printf)\(\s*(?:2\s*,\s*)?)'(?P<body>[^']*\n[^']*)'",
    re.MULTILINE,
)

changed = []
for path in ROOT.rglob('*.m'):
    if '.git' in path.parts:
        continue
    text = path.read_text(encoding='utf-8')

    def fix(match):
        body = match.group('body').replace('\r\n', '\n').replace('\r', '\n')
        body = body.replace('\n', r'\n')
        return match.group('prefix') + "'" + body + "'"

    new_text, n = call_pattern.subn(fix, text)
    if n:
        path.write_text(new_text, encoding='utf-8')
        changed.append((str(path.relative_to(ROOT)), n))

# Re-scan: a direct fprintf/safe_printf single-quoted format string must not
# physically cross a source line.
remaining = []
for path in ROOT.rglob('*.m'):
    if '.git' in path.parts:
        continue
    text = path.read_text(encoding='utf-8')
    if call_pattern.search(text):
        remaining.append(str(path.relative_to(ROOT)))
if remaining:
    raise SystemExit('Multiline format strings remain: ' + ', '.join(remaining))

# Stronger fixed-target provenance audit. Generic observer slot variables
# such as slot_index/slot_indices remain valid; target-specific dep/arr slot
# fields do not.
forbidden = [
    'depSlot', 'arrSlot', 'depOrbitIndex', 'arrOrbitIndex',
    'depOrbitID', 'arrOrbitID', 'low_thrust_case_config',
    'legacyCatalogRow', 'resolvedCatalogRow', 'endpointAudit',
    'slotStateReferenceError', 'fixedVsObserverSlotError',
]
hits = []
for path in ROOT.rglob('*.m'):
    if '.git' in path.parts or path == Path(__file__).resolve():
        continue
    text = path.read_text(encoding='utf-8', errors='ignore')
    for token in forbidden:
        if token.lower() in text.lower():
            hits.append(f'{path.relative_to(ROOT)}: {token}')
if hits:
    print('\n'.join(hits), file=sys.stderr)
    raise SystemExit(2)

# Temporary workflow files must not remain in the final repository.
for rel in [
    '.github/one_time_matlab_syntax_audit.py',
    '.github/workflows/one-time-matlab-syntax-audit.yml',
]:
    p = ROOT / rel
    if p.exists():
        p.unlink()

print('MATLAB format-string repair passed.')
for name, count in changed:
    print(f'  repaired {count} format string(s): {name}')
print('Fixed-target dep/arr slot provenance audit passed.')
