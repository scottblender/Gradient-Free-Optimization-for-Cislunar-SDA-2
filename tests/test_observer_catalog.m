function audit = test_observer_catalog()
%TEST_OBSERVER_CATALOG Validate the LHS-selected 450-orbit observer database.

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
assert(all(isfinite(T.stability)), 'A selected observer orbit has a nonfinite stability index.');
audit = struct('numOrbits',height(T),'familyCounts',counts,'maxJacobiVariation',max(T.jacobiVariation),'jacobiRange',[min(T.jacobiConstant),max(T.jacobiConstant)],'stabilityRange',[min(T.stability),max(T.stability)]);
fprintf('\nObserver catalog audit passed.\n');
fprintf('Selected observer orbits: %d\n',audit.numOrbits);
fprintf('Maximum Jacobi variation: %.6e\n',audit.maxJacobiVariation);
end
