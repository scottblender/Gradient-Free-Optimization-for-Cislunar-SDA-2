clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
catalogPath = fullfile(projectRoot,"data", ...
    "JPL_CR3BP_OrbitCatalog.mat");
legacyCatalogPath = fullfile(projectRoot,"data", ...
    "JPL_CR3BP_OrbitCatalog_Old.mat");
referencePath = fullfile(projectRoot,"data", ...
    "transfer_reference.mat");

% Original intended transfer definition. Old row 51 is the northern L1
% orbit; old row 52 is its southern mirror.
legacyDepIndex = 51;
legacyArrIndex = 400;
depSlot = 10;
arrSlot = 1;

expectedDepID = "northern_halo_l1:1015";
expectedArrID = "southern_halo_l2:97";

% Fall back to the exact states extracted from the old catalog when the
% legacy MAT file is not retained locally.
legacyDepState0 = [ ...
     0.8402957900765589,  6.021304054218165e-28, ...
     0.1583034195863712, -7.135606147838344e-16, ...
     0.2616335345519606, -9.179544887747458e-16];
legacyArrState0 = [ ...
     1.074068135022175,   3.285715872558785e-27, ...
    -0.2020446972919714,  8.927284288140170e-15, ...
    -0.1910274217128591, -9.194522021181726e-15];

if isfile(legacyCatalogPath)
    legacyData = load(legacyCatalogPath,"T");
    legacyTable = legacyData.T;

    legacyDepState0 = legacyTable.state{legacyDepIndex}(1,:);
    legacyArrState0 = legacyTable.state{legacyArrIndex}(1,:);

    fprintf("Loaded endpoint states directly from old catalog:\n  %s\n", ...
        legacyCatalogPath);
else
    fprintf([ ...
        "Old catalog was not found; using the stored exact old-catalog " ...
        "endpoint states.\n"]);
end

catalogData = load(catalogPath,"T");
T = catalogData.T;
periods = T.("Period (TU) ");

[depIndex,departureStateError] = ...
    find_state_match(T,legacyDepState0);
[arrIndex,arrivalStateError] = ...
    find_state_match(T,legacyArrState0);

assert(departureStateError <= 1e-12, ...
    "Old departure did not match the rebuilt catalog.");
assert(arrivalStateError <= 1e-12, ...
    "Old arrival did not match the rebuilt catalog.");
assert(mean(T.state{depIndex}(:,3)) > 0, ...
    "Matched departure is not on the northern L1 branch.");

tableNames = string(T.Properties.VariableNames);
assert(ismember("orbitID",tableNames), ...
    "The rebuilt catalog does not contain orbitID.");

depOrbitID = lower(strtrim(string(T.orbitID(depIndex))));
arrOrbitID = lower(strtrim(string(T.orbitID(arrIndex))));

assert(depOrbitID==expectedDepID, ...
    "Old departure state matched %s instead of %s.", ...
    depOrbitID,expectedDepID);
assert(arrOrbitID==expectedArrID, ...
    "Old arrival state matched %s instead of %s.", ...
    arrOrbitID,expectedArrID);

transferRef = struct();

transferRef.dep.legacyIndex = legacyDepIndex;
transferRef.dep.newIndex = depIndex;
transferRef.dep.slot = depSlot;
transferRef.dep.state0 = legacyDepState0;
transferRef.dep.period = periods(depIndex);
transferRef.dep.family = T.orbitFamily(depIndex);
transferRef.dep.orbitID = depOrbitID;
transferRef.dep.stateMatchError = departureStateError;

transferRef.arr.legacyIndex = legacyArrIndex;
transferRef.arr.newIndex = arrIndex;
transferRef.arr.slot = arrSlot;
transferRef.arr.state0 = legacyArrState0;
transferRef.arr.period = periods(arrIndex);
transferRef.arr.family = T.orbitFamily(arrIndex);
transferRef.arr.orbitID = arrOrbitID;
transferRef.arr.stateMatchError = arrivalStateError;

save(referencePath,"transferRef");

fprintf("Saved state-matched low-thrust reference to:\n  %s\n", ...
    referencePath);
fprintf([ ...
    "Departure: old row %d -> current row %d, slot %d, %s, " ...
    "state error %.3e\n"], ...
    legacyDepIndex,depIndex,depSlot,depOrbitID,departureStateError);
fprintf([ ...
    "Arrival:   old row %d -> current row %d, slot %d, %s, " ...
    "state error %.3e\n"], ...
    legacyArrIndex,arrIndex,arrSlot,arrOrbitID,arrivalStateError);


function [index,minimumDistance] = find_state_match(T,referenceState)

initialStates = zeros(height(T),6);

for k = 1:height(T)
    initialStates(k,:) = T.state{k}(1,:);
end

distance = vecnorm(initialStates-referenceState,2,2);
[minimumDistance,index] = min(distance);

assert(minimumDistance < 1e-10, ...
    ['The old-catalog orbit was not retained in the rebuilt catalog. ' ...
     'Minimum initial-state difference = %.6e.'],minimumDistance);
end
