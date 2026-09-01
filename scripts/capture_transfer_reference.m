clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
catalogPath = fullfile(projectRoot,"data", ...
    "JPL_CR3BP_OrbitCatalog.mat");
referencePath = fullfile(projectRoot,"data", ...
    "transfer_reference.mat");

% Original paper definition before catalog reordering.
legacyDepIndex = 51;
legacyArrIndex = 400;
depSlot = 10;
arrSlot = 1;

% Stable identities corresponding to legacy rows 51 and 400. Legacy row
% 52 is the southern mirror of row 51 and must not be used here.
expectedDepID = "northern_halo_l1:1015";
expectedArrID = "southern_halo_l2:97";

expectedDepState0 = [ ...
     0.8402957900765589,  6.021304054218165e-28, ...
     0.1583034195863712, -7.135606147838344e-16, ...
     0.2616335345519606, -9.179544887747458e-16];

S = load(catalogPath,"T");
T = S.T;
periods = T.("Period (TU) ");
tableNames = string(T.Properties.VariableNames);

if ismember("orbitID",tableNames)
    % Rebuilt catalog: resolve the original physical orbits by stable ID.
    catalogIds = strtrim(lower(string(T.orbitID)));
    depIndex = find(catalogIds==expectedDepID);
    arrIndex = find(catalogIds==expectedArrID);

    assert(numel(depIndex)==1, ...
        "Legacy orbit 51 (%s) did not resolve uniquely.",expectedDepID);
    assert(numel(arrIndex)==1, ...
        "Legacy orbit 400 (%s) did not resolve uniquely.",expectedArrID);
else
    % Legacy catalog: construct stable IDs from sourceFile and the exact
    % preserved source column named Id. Do not use a case-insensitive search
    % because the table can contain more than one ID-like column.
    idMatch = find(strtrim(tableNames)=="Id");
    assert(numel(idMatch)==1, ...
        'Expected one exact source column named Id; found %d.',numel(idMatch));
    assert(ismember("sourceFile",tableNames), ...
        "The legacy catalog does not contain sourceFile.");

    sourceIds = strtrim(string(T.(char(tableNames(idMatch)))));
    sourceStem = erase(lower(strtrim(string(T.sourceFile))),".csv");
    catalogIds = sourceStem+":"+sourceIds;

    depIndex = legacyDepIndex;
    arrIndex = legacyArrIndex;

    assert(catalogIds(depIndex)==expectedDepID, ...
        "Legacy row 51 resolved to %s instead of %s.", ...
        catalogIds(depIndex),expectedDepID);
    assert(catalogIds(arrIndex)==expectedArrID, ...
        "Legacy row 400 resolved to %s instead of %s.", ...
        catalogIds(arrIndex),expectedArrID);
end

departureStateError = norm( ...
    T.state{depIndex}(1,:) - expectedDepState0);
assert(departureStateError <= 1e-12, ...
    ['Resolved departure does not match old catalog row 51. ' ...
     'Initial-state error = %.6e.'],departureStateError);
assert(mean(T.state{depIndex}(:,3)) > 0, ...
    'Resolved departure is not on the northern L1 branch.');

transferRef = struct();

transferRef.dep.legacyIndex = legacyDepIndex;
transferRef.dep.newIndex = depIndex;
transferRef.dep.slot = depSlot;
transferRef.dep.state0 = T.state{depIndex}(1,:);
transferRef.dep.period = periods(depIndex);
transferRef.dep.family = T.orbitFamily(depIndex);
transferRef.dep.orbitID = expectedDepID;

transferRef.arr.legacyIndex = legacyArrIndex;
transferRef.arr.newIndex = arrIndex;
transferRef.arr.slot = arrSlot;
transferRef.arr.state0 = T.state{arrIndex}(1,:);
transferRef.arr.period = periods(arrIndex);
transferRef.arr.family = T.orbitFamily(arrIndex);
transferRef.arr.orbitID = expectedArrID;

save(referencePath,"transferRef");

fprintf("Saved original low-thrust transfer reference to:\n  %s\n", ...
    referencePath);
fprintf("Departure: legacy row %d -> current row %d, slot %d, %s\n", ...
    legacyDepIndex,depIndex,depSlot,expectedDepID);
fprintf("Arrival:   legacy row %d -> current row %d, slot %d, %s\n", ...
    legacyArrIndex,arrIndex,arrSlot,expectedArrID);
