% ----- test_catalog_dro_and_transfer.m ----- %
% Validates:
%   1. Orbit catalog structure and descriptors
%   2. DRO count, stability, and geometric spread
%   3. Unique orbit identifiers
%   4. Remapped low-thrust transfer endpoints
%   5. Equal-time, endpoint-excluded orbit slots
%   6. Existing orbit-database cache, when available

clear;
clc;

fprintf("\n--- Catalog, DRO, and transfer audit ---\n\n");

% ---------------- Test configuration ----------------
EXPECTED_DRO_COUNT = 50;
DRO_STABILITY_MAX  = 1 + 1e-8;
SLOTS_PER_ORBIT    = 50;

EXPECTED_DEP_SLOT = 10;
EXPECTED_ARR_SLOT = 1;

STATE_TOLERANCE  = 1e-10;
PERIOD_TOLERANCE = 1e-12;
SLOT_TOLERANCE   = 1e-10;

% This is only a distribution-quality warning, not a hard failure.
MAX_NORMALIZED_DRO_GAP = 4 / EXPECTED_DRO_COUNT;

% ---------------- Project paths ----------------
testPath = mfilename("fullpath");
testDir = fileparts(testPath);
projectRoot = fileparts(testDir);

addpath(projectRoot);
projectPaths = setup_project();

catalogPath = projectPaths.catalog;
referencePath = fullfile( ...
    projectPaths.data, "transfer_reference.mat");

orbitCachePath = fullfile( ...
    projectPaths.orbitCache, ...
    sprintf("orbit_database_slots_%d.mat", ...
    SLOTS_PER_ORBIT));

assert(isfile(catalogPath), ...
    "Catalog file was not found:\n%s", catalogPath);

assert(isfile(referencePath), ...
    ['Transfer reference file was not found:\n%s\n' ...
     'Run capture_transfer_reference.m before rebuilding the catalog.'], ...
    referencePath);

% ---------------- Load catalog ----------------
catalogData = load(catalogPath);

assert(isfield(catalogData, "T"), ...
    "Catalog MAT-file does not contain table T.");

T = catalogData.T;

assert(istable(T), ...
    "Catalog variable T must be a table.");

fprintf("Loaded catalog:\n  %s\n", catalogPath);
fprintf("Selected catalog orbits: %d\n\n", height(T));

% ---------------- Required catalog columns ----------------
requiredColumns = [
    "orbitFamily"
    "Id"
    "orbitID"
    "state"
    "time"
    "periluneAltitude_km"
    "apoluneAltitude_km"
    "inPlaneAmplitude_km"
    "outPlaneAmplitude_km"
];

tableNames = string(T.Properties.VariableNames);

for i = 1:numel(requiredColumns)
    assert(any(tableNames == requiredColumns(i)), ...
        "Catalog is missing required column '%s'.", ...
        requiredColumns(i));
end

periods = get_table_column(T, [
    "period_TU"
    "Period (TU) "
]);

stability = get_table_column(T, [
    "stability"
    "Stability index  "
]);

families = string(T.orbitFamily);
orbitIDs = string(T.orbitID);

% ---------------- General catalog checks ----------------
assert(all(strlength(orbitIDs) > 0), ...
    "One or more catalog orbits have an empty orbitID.");

assert(numel(unique(orbitIDs)) == height(T), ...
    "The catalog contains duplicate orbit identifiers.");

assert(all(isfinite(periods)), ...
    "One or more catalog periods are nonfinite.");

assert(all(periods > 0), ...
    "One or more catalog periods are not positive.");

assert(all(isfinite(stability)), ...
    "One or more selected stability indices are nonfinite.");

descriptorNames = [
    "periluneAltitude_km"
    "apoluneAltitude_km"
    "inPlaneAmplitude_km"
    "outPlaneAmplitude_km"
];

for i = 1:numel(descriptorNames)

    values = T.(descriptorNames(i));

    assert(all(isfinite(values)), ...
        "Column '%s' contains nonfinite values.", ...
        descriptorNames(i));
end

assert(all(T.apoluneAltitude_km >= ...
    T.periluneAltitude_km), ...
    "At least one orbit has apolune below perilune.");

% Allow a very small numerical tolerance at the lunar surface.
assert(all(T.periluneAltitude_km >= -1e-3), ...
    ['At least one selected orbit penetrates the Moon. ' ...
     'Minimum perilune altitude = %.6e km.'], ...
    min(T.periluneAltitude_km));

if any(tableNames == "collides")
    assert(~any(T.collides), ...
        "The selected catalog contains a colliding orbit.");
end

if any(tableNames == "nearLG")
    assert(~any(T.nearLG), ...
        "The selected catalog contains an orbit marked near the Gateway.");
end

fprintf("Catalog structure checks passed.\n");
fprintf("Unique orbit IDs: %d/%d\n\n", ...
    numel(unique(orbitIDs)), height(T));

% ---------------- Family counts ----------------
uniqueFamilies = unique(families);
familyCounts = zeros(numel(uniqueFamilies),1);

for i = 1:numel(uniqueFamilies)
    familyCounts(i) = nnz(families == uniqueFamilies(i));
end

familySummary = table( ...
    uniqueFamilies, familyCounts, ...
    "VariableNames", ["Family", "Count"]);

fprintf("--- Selected orbit families ---\n");
disp(familySummary);

% ---------------- DRO checks ----------------
isDRO = families == "DRO";
droCount = nnz(isDRO);

assert(droCount == EXPECTED_DRO_COUNT, ...
    "Expected %d DROs, but found %d.", ...
    EXPECTED_DRO_COUNT, droCount);

droStability = stability(isDRO);

assert(all(droStability <= DRO_STABILITY_MAX), ...
    ['At least one selected DRO exceeds the stability threshold.\n' ...
     'Maximum selected DRO stability = %.12g'], ...
    max(droStability));

droApolune = T.apoluneAltitude_km(isDRO);
droPerilune = T.periluneAltitude_km(isDRO);
droPeriod = periods(isDRO);

assert(max(droApolune) > min(droApolune), ...
    "The selected DROs have no apolune-altitude variation.");

% Normalize the selected apolune-altitude range.
droApoluneNormalized = ...
    (droApolune - min(droApolune)) ./ ...
    (max(droApolune) - min(droApolune));

droApoluneNormalized = sort(droApoluneNormalized);
droGaps = diff(droApoluneNormalized);
maximumDroGap = max(droGaps);

% Use ten coarse bins as a simple coverage diagnostic.
numberOfBins = 10;
edges = linspace(0, 1, numberOfBins + 1);
binIndex = discretize(droApoluneNormalized, edges);

occupiedBins = numel(unique(binIndex(~isnan(binIndex))));

fprintf("--- DRO selection ---\n");
fprintf("Selected DROs:                 %d\n", droCount);
fprintf("Maximum stability index:       %.12g\n", ...
    max(droStability));
fprintf("Perilune altitude range:       %.3f to %.3f km\n", ...
    min(droPerilune), max(droPerilune));
fprintf("Apolune altitude range:        %.3f to %.3f km\n", ...
    min(droApolune), max(droApolune));
fprintf("Period range:                  %.6f to %.6f TU\n", ...
    min(droPeriod), max(droPeriod));
fprintf("Occupied apolune bins:         %d/%d\n", ...
    occupiedBins, numberOfBins);
fprintf("Maximum normalized gap:        %.6f\n\n", ...
    maximumDroGap);

if maximumDroGap > MAX_NORMALIZED_DRO_GAP
    warning( ...
        ['The DRO apolune distribution contains a relatively large gap.\n' ...
         'Maximum normalized gap = %.6f; warning threshold = %.6f.\n' ...
         'This may reflect sparse source data rather than an error.'], ...
        maximumDroGap, MAX_NORMALIZED_DRO_GAP);
end

if occupiedBins < numberOfBins
    warning( ...
        ['The selected DROs do not occupy every coarse apolune bin.\n' ...
         'Inspect the DRO distribution before using the final catalog.']);
end

fprintf("DRO hard checks passed.\n\n");

% ---------------- Transfer-reference checks ----------------
referenceData = load(referencePath, "transferRef");

assert(isfield(referenceData, "transferRef"), ...
    "transfer_reference.mat does not contain transferRef.");

transferRef = referenceData.transferRef;

depIndex = validate_transfer_reference( ...
    "Departure", transferRef.dep, T, orbitIDs, periods, ...
    EXPECTED_DEP_SLOT, SLOTS_PER_ORBIT, ...
    STATE_TOLERANCE, PERIOD_TOLERANCE);

arrIndex = validate_transfer_reference( ...
    "Arrival", transferRef.arr, T, orbitIDs, periods, ...
    EXPECTED_ARR_SLOT, SLOTS_PER_ORBIT, ...
    STATE_TOLERANCE, PERIOD_TOLERANCE);

assert(depIndex ~= arrIndex, ...
    "Departure and arrival resolve to the same catalog orbit.");

fprintf("\nTransfer-reference checks passed.\n\n");

% ---------------- Equal-time slot definition ----------------
% Independently reconstruct the slot epochs for both transfer orbits.
transferIndices = [depIndex, arrIndex];
transferLabels = ["Departure", "Arrival"];

for i = 1:numel(transferIndices)

    orbitIndex = transferIndices(i);
    period = periods(orbitIndex);

    slotTimes = ...
        (0:SLOTS_PER_ORBIT-1)' * period / SLOTS_PER_ORBIT;

    expectedSpacing = period / SLOTS_PER_ORBIT;

    assert(slotTimes(1) == 0, ...
        "%s slot 1 is not located at t = 0.", ...
        transferLabels(i));

    assert(all(diff(slotTimes) > 0), ...
        "%s slot epochs are not strictly increasing.", ...
        transferLabels(i));

    assert(max(abs(diff(slotTimes) - expectedSpacing)) ...
        < 100*eps(period), ...
        "%s slots are not equally spaced in time.", ...
        transferLabels(i));

    assert(slotTimes(end) < period, ...
        "%s slot definition includes the repeated period endpoint.", ...
        transferLabels(i));

    expectedFinalTime = ...
        (SLOTS_PER_ORBIT-1) * period / SLOTS_PER_ORBIT;

    assert(abs(slotTimes(end) - expectedFinalTime) ...
        < 100*eps(period), ...
        "%s final slot epoch is incorrect.", ...
        transferLabels(i));
end

fprintf("Equal-time slot-definition checks passed.\n\n");

% ---------------- Orbit cache check ----------------
cacheChecked = false;
maximumSlotStateError = NaN;

if isfile(orbitCachePath)

    cacheData = load( ...
        orbitCachePath, "orbit_database", "cacheMeta");

    assert(isfield(cacheData, "orbit_database"), ...
        "Orbit cache does not contain orbit_database.");

    assert(isfield(cacheData, "cacheMeta"), ...
        "Orbit cache does not contain cacheMeta.");

    cacheMeta = cacheData.cacheMeta;
    orbitDatabase = cacheData.orbit_database;

    assert(numel(orbitDatabase) == height(T), ...
        ['Orbit cache contains %d orbits, while the current ' ...
         'catalog contains %d.'], ...
        numel(orbitDatabase), height(T));

    assert(isfield(cacheMeta, "slotDefinition"), ...
        "Orbit cache metadata does not contain slotDefinition.");

    assert(string(cacheMeta.slotDefinition) == ...
        "equal_time_no_endpoint_v1", ...
        "Orbit cache uses an unexpected slot definition.");

    assert(isfield(cacheMeta, "slots_per_orbit") && ...
        cacheMeta.slots_per_orbit == SLOTS_PER_ORBIT, ...
        "Orbit cache contains the wrong number of slots.");

    if isfield(cacheMeta, "catalogHash")

        currentCatalogHash = study_hash(catalogPath, "file");

        assert(string(cacheMeta.catalogHash) == ...
            currentCatalogHash, ...
            ['Orbit cache was built from a different catalog.\n' ...
             'Run a small optimization pilot to rebuild it.']);
    end

    maximumSlotStateError = 0;

    for i = 1:numel(transferIndices)

        orbitIndex = transferIndices(i);

        tRaw = T.time{orbitIndex};
        sRaw = T.state{orbitIndex};
        period = periods(orbitIndex);

        slotTimes = ...
            (0:SLOTS_PER_ORBIT-1)' * ...
            period / SLOTS_PER_ORBIT;

        [tUnique, uniqueIndex] = unique(tRaw);
        sUnique = sRaw(uniqueIndex,:);

        interpolant = griddedInterpolant( ...
            tUnique, sUnique, "spline");

        expectedStates = interpolant(slotTimes);
        cachedStates = orbitDatabase{orbitIndex};

        assert(size(cachedStates,1) == SLOTS_PER_ORBIT, ...
            "Cached %s orbit has the wrong number of slots.", ...
            transferLabels(i));

        stateError = max( ...
            abs(cachedStates - expectedStates), [], "all");

        maximumSlotStateError = max( ...
            maximumSlotStateError, stateError);

        assert(stateError <= SLOT_TOLERANCE, ...
            ['Cached %s states do not use the expected ' ...
             'equal-time slot definition.\n' ...
             'Maximum state error = %.6e'], ...
            transferLabels(i), stateError);
    end

    cacheChecked = true;

    fprintf("Orbit cache checks passed.\n");
    fprintf("Maximum cached-slot state error: %.6e\n\n", ...
        maximumSlotStateError);

else

    fprintf(['Orbit cache was not found. The analytical slot checks ' ...
             'passed, but cached states were not checked.\n']);
    fprintf("Expected cache:\n  %s\n\n", orbitCachePath);
end

% ---------------- Final audit summary ----------------
catalogAudit = struct();

catalogAudit.numOrbits = height(T);
catalogAudit.numFamilies = numel(uniqueFamilies);
catalogAudit.numDRO = droCount;
catalogAudit.uniqueOrbitIDs = numel(unique(orbitIDs));
catalogAudit.droMaximumStability = max(droStability);
catalogAudit.droApoluneRange_km = [
    min(droApolune), max(droApolune)];
catalogAudit.droPeriodRange_TU = [
    min(droPeriod), max(droPeriod)];
catalogAudit.droOccupiedBins = occupiedBins;
catalogAudit.droMaximumNormalizedGap = maximumDroGap;
catalogAudit.departureIndex = depIndex;
catalogAudit.arrivalIndex = arrIndex;
catalogAudit.departureSlot = transferRef.dep.slot;
catalogAudit.arrivalSlot = transferRef.arr.slot;
catalogAudit.cacheChecked = cacheChecked;
catalogAudit.maximumSlotStateError = maximumSlotStateError;

fprintf("--- Results ---\n");
fprintf("Catalog orbits:                 %d\n", ...
    catalogAudit.numOrbits);
fprintf("Orbit families:                 %d\n", ...
    catalogAudit.numFamilies);
fprintf("Selected DROs:                  %d\n", ...
    catalogAudit.numDRO);
fprintf("Unique orbit IDs:               %d\n", ...
    catalogAudit.uniqueOrbitIDs);
fprintf("Departure row/slot:             %d/%d\n", ...
    catalogAudit.departureIndex, ...
    catalogAudit.departureSlot);
fprintf("Arrival row/slot:               %d/%d\n", ...
    catalogAudit.arrivalIndex, ...
    catalogAudit.arrivalSlot);
fprintf("Orbit cache checked:            %d\n", ...
    catalogAudit.cacheChecked);

fprintf("\nAll catalog and transfer tests passed.\n");

% ========================================================================
% Local functions
% ========================================================================

function values = get_table_column(T, candidateNames)

    tableNames = string(T.Properties.VariableNames);

    for i = 1:numel(candidateNames)

        match = find(tableNames == candidateNames(i), 1);

        if ~isempty(match)
            values = T.(tableNames(match));
            return
        end
    end

    error( ...
        "None of the expected table columns were found: %s", ...
        strjoin(candidateNames, ", "));
end

function orbitIndex = validate_transfer_reference( ...
    label, reference, T, orbitIDs, periods, ...
    expectedSlot, slotsPerOrbit, ...
    stateTolerance, periodTolerance)

    requiredFields = [
        "orbitID"
        "state0"
        "period"
        "slot"
    ];

    for i = 1:numel(requiredFields)
        assert(isfield(reference, requiredFields(i)), ...
            "%s reference is missing field '%s'.", ...
            label, requiredFields(i));
    end

    referenceID = string(reference.orbitID);
    matches = find(orbitIDs == referenceID);

    assert(numel(matches) == 1, ...
        ['%s orbit ID must match exactly one catalog row. ' ...
         'Number of matches = %d.'], ...
        label, numel(matches));

    orbitIndex = matches(1);

    if isfield(reference, "newIndex")

        assert(reference.newIndex == orbitIndex, ...
            ['%s stored newIndex is %d, but orbitID resolves ' ...
             'to row %d.'], ...
            label, reference.newIndex, orbitIndex);
    end

    catalogState0 = T.state{orbitIndex}(1,:);
    referenceState0 = reshape(reference.state0, 1, []);

    stateError = norm(catalogState0 - referenceState0);

    assert(stateError <= stateTolerance, ...
        ['%s state does not match the captured reference.\n' ...
         'State error = %.6e'], ...
        label, stateError);

    periodError = abs(periods(orbitIndex) - reference.period);

    assert(periodError <= periodTolerance, ...
        ['%s period does not match the captured reference.\n' ...
         'Period error = %.6e TU'], ...
        label, periodError);

    assert(reference.slot == expectedSlot, ...
        "%s slot changed from %d to %d.", ...
        label, expectedSlot, reference.slot);

    assert(reference.slot >= 1 && ...
        reference.slot <= slotsPerOrbit, ...
        "%s slot is outside the valid range.", label);

    fprintf("%s transfer orbit:\n", label);
    fprintf("  Catalog row:    %d\n", orbitIndex);
    fprintf("  Slot:           %d\n", reference.slot);
    fprintf("  Orbit ID:       %s\n", referenceID);
    fprintf("  State error:    %.6e\n", stateError);
    fprintf("  Period error:   %.6e TU\n", periodError);
end