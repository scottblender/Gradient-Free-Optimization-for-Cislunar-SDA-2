from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
core_path = ROOT / 'scripts/private/observer_catalog_filter_core.m'
test_path = ROOT / 'tests/test_observer_catalog.m'

core = core_path.read_text(encoding='utf-8')

new_selection = r'''% ---------------- Orbit-family selection ----------------
% Select 50 representative trajectories from the full eligible population
% of every family using one-dimensional Latin-hypercube targets along a
% geometry coordinate that spans the family continuation. Stability is
% deliberately NOT used as either a filter or a sampling coordinate so it
% remains an outcome variable for the optimization study.
K = 50;
LHS_SEED_BASE = 20260901;

T.stability = T.("Stability index  ");
T.period_TU = T.("Period (TU) ");

families = sort(unique(T.orbitFamily));
keepMask = false(height(T),1);

for f = 1:numel(families)

    familyName = families(f);
    familyIdx = find(T.orbitFamily == familyName);

    assert(numel(familyIdx) >= K, ...
        ['Fewer than %d eligible %s candidates remain after collision ' ...
         'and near-Gateway screening.'], K, familyName);

    if familyName == "DRO"
        % DROs are planar, so out-of-plane amplitude cannot parameterize
        % the family. Moon-relative apolune altitude provides a monotonic
        % geometric progression across the available DRO trajectories.
        sampleValue = T.apoluneAltitude_km(familyIdx);
        sampleCoordinate = "apolune altitude";
    else
        % Halo and NRHO families are naturally spanned by their
        % out-of-plane extent. Use the propagated maximum |z| amplitude,
        % not stability, to distribute the 50 representatives.
        sampleValue = T.zAmplitude(familyIdx);
        sampleCoordinate = "z amplitude";
    end

    localTake = select_family_lhs( ...
        sampleValue, K, LHS_SEED_BASE + f);
    take = familyIdx(localTake);

    fprintf('%s candidates: %d eligible; selected %d by LHS over %s.\n', ...
        familyName, numel(familyIdx), numel(take), sampleCoordinate);

    keepMask(take) = true;
end

fprintf("Keeping %d total orbits after all-family LHS selection.\n", ...
    nnz(keepMask));

T = T(keepMask,:);
'''

pattern = re.compile(
    r'% ---------------- Orbit-family selection ----------------\n.*?\nT = T\(keepMask,:\);\n',
    re.S,
)
core, n = pattern.subn(new_selection, core, count=1)
if n != 1:
    raise RuntimeError(f'Expected one orbit-family selection block, found {n}.')

core = core.replace(
    '% Preserve the original global z-amplitude ordering for Halo and\n'
    '% near-rectilinear Halo families. Organize the separately sampled DRO\n'
    '% population by ascending apolune altitude, then append it to the catalog.',
    '% Preserve the original global z-amplitude ordering for Halo and\n'
    '% near-rectilinear Halo families. Organize the LHS-sampled DRO population\n'
    '% by ascending apolune altitude, then append it to the catalog.',
    1,
)

helper_pattern = re.compile(
    r'function take = select_dro_lhs\(Tdro, K, seed\)\n.*?\nend\s*$',
    re.S,
)
new_helper = r'''function take = select_family_lhs(value, K, seed)
% Select actual catalog rows using Latin-hypercube targets distributed over
% a scalar geometric coordinate spanning one eligible orbit family.

value = value(:);
assert(numel(value) >= K, ...
    "The family candidate set must contain at least K rows.");
assert(all(isfinite(value)), ...
    "The family LHS coordinate contains nonfinite values.");

valueMin = min(value);
valueMax = max(value);
assert(valueMax > valueMin, ...
    "The family LHS coordinate range is zero.");

valueNormalized = (value - valueMin) ./ (valueMax - valueMin);

previousRng = rng;
cleanup = onCleanup(@() rng(previousRng)); %#ok<NASGU>
rng(seed, "twister");

% Draw one target in every equal-width stratum of [0,1]. The nearest
% currently available catalog trajectory is assigned to each target so the
% result remains a subset of the original JPL orbit population.
targets = lhsdesign(K, 1, "Criterion", "none");
targets = sort(targets);

available = true(numel(value),1);
take = zeros(K,1);

for k = 1:K
    distance = abs(valueNormalized - targets(k));
    distance(~available) = inf;
    [~, selected] = min(distance);
    take(k) = selected;
    available(selected) = false;
end
end
'''
core, n = helper_pattern.subn(new_helper, core, count=1)
if n != 1:
    raise RuntimeError(f'Expected one DRO LHS helper, found {n}.')

core_path.write_text(core, encoding='utf-8')

# Stability remains a required recorded property, but it is no longer a
# catalog-selection constraint for DRO or any other family.
test = test_path.read_text(encoding='utf-8')
test = test.replace(
    '%TEST_OBSERVER_CATALOG Validate the filtered 450-orbit observer database.',
    '%TEST_OBSERVER_CATALOG Validate the LHS-selected 450-orbit observer database.',
    1,
)
old = '''isDRO = string(T.manuscriptFamily)=="DRO";\nassert(all(T.stability(isDRO) <= 1+1e-8), 'A selected DRO exceeds the stability threshold.');\naudit = struct('numOrbits',height(T),'familyCounts',counts,'maxJacobiVariation',max(T.jacobiVariation),'jacobiRange',[min(T.jacobiConstant),max(T.jacobiConstant)]);'''
new = '''assert(all(isfinite(T.stability)), 'A selected observer orbit has a nonfinite stability index.');\naudit = struct('numOrbits',height(T),'familyCounts',counts,'maxJacobiVariation',max(T.jacobiVariation),'jacobiRange',[min(T.jacobiConstant),max(T.jacobiConstant)],'stabilityRange',[min(T.stability),max(T.stability)]);'''
if old not in test:
    raise RuntimeError('Expected DRO stability-threshold test block was not found.')
test = test.replace(old, new, 1)
test_path.write_text(test, encoding='utf-8')

# Static checks for the intended experimental boundary.
combined = core_path.read_text(encoding='utf-8') + '\n' + test_path.read_text(encoding='utf-8')
for forbidden in ['DRO_STABILITY_MAX', 'select_dro_lhs', 'A selected DRO exceeds the stability threshold']:
    if forbidden in combined:
        raise RuntimeError(f'Obsolete stability-selection logic remains: {forbidden}')

if 'sort(stability' in core_path.read_text(encoding='utf-8'):
    raise RuntimeError('Stability sorting remains in observer catalog selection.')

# Remove this one-time patcher after it has run.
for rel in [
    '.github/one_time_lhs_all_families.py',
    '.github/workflows/one-time-lhs-all-families.yml',
]:
    p = ROOT / rel
    if p.exists():
        p.unlink()

print('All-family LHS observer selection patch applied.')
print('Stability is retained as a catalog property but not used for selection.')
