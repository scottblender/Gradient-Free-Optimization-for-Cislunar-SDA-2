clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
catalogPath = fullfile(projectRoot,"data", ...
    "JPL_CR3BP_OrbitCatalog.mat");
referencePath = fullfile(projectRoot,"data", ...
    "transfer_reference.mat");

% Original paper definition before catalog reordering.
legacyDepIndex = 52;
legacyArrIndex = 400;
depSlot = 10;
arrSlot = 1;

% Stable identities corresponding to legacy rows 52 and 400. Legacy
% row 52 is the northern L1 halo used by the original low-thrust case.
expectedDepID = "northern_halo_l1:1015";
expectedArrID = "southern_halo_l2:97";

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
        "Legacy orbit 52 (%s) did not resolve uniquely.",expectedDepID);
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
        "Legacy row 52 resolved to %s instead of %s.", ...
        catalogIds(depIndex),expectedDepID);
    assert(catalogIds(arrIndex)==expectedArrID, ...
        "Legacy row 400 resolved to %s instead of %s.", ...
        catalogIds(arrIndex),expectedArrID);
end

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
