clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
catalogPath = fullfile(projectRoot,"data", ...
    "JPL_CR3BP_OrbitCatalog.mat");
legacyCatalogPath = fullfile(projectRoot,"data", ...
    "JPL_CR3BP_OrbitCatalog_Old.mat");
referencePath = fullfile(projectRoot,"data", ...
    "transfer_reference.mat");

% Exact transfer definition used by the original study.
legacyDepIndex = 52;
legacyArrIndex = 400;
depSlot = 10;
arrSlot = 1;
slotsPerOrbit = 50;

expectedDepID = "southern_halo_l1:1015";
expectedArrID = "southern_halo_l2:97";
slotDefinition = "equal_time_no_endpoint_v1";

assert(isfile(legacyCatalogPath), ...
    ['The old catalog is required to capture the original transfer.\n' ...
     'Expected file:\n%s'], legacyCatalogPath);
assert(isfile(catalogPath), ...
    'The current catalog was not found:\n%s', catalogPath);

legacyData = load(legacyCatalogPath,"T");
legacyTable = legacyData.T;
assert(height(legacyTable) >= max(legacyDepIndex,legacyArrIndex), ...
    'The old catalog does not contain rows 52 and 400.');

legacyPeriods = legacyTable.("Period (TU) ");
legacyDepPeriod = legacyPeriods(legacyDepIndex);
legacyArrPeriod = legacyPeriods(legacyArrIndex);

legacyDepState0 = legacyTable.state{legacyDepIndex}(1,:);
legacyArrState0 = legacyTable.state{legacyArrIndex}(1,:);

% The revised database uses 50 equal-time slots without storing t = T.
% Slot 10 is therefore located at 9T/50. The original inclusive linspace
% placed slot 10 at 9T/49; retain that state only as legacy provenance.
depSlotEpoch = (depSlot-1)*legacyDepPeriod/slotsPerOrbit;
arrSlotEpoch = (arrSlot-1)*legacyArrPeriod/slotsPerOrbit;
legacyInclusiveDepSlotEpoch = ...
    (depSlot-1)*legacyDepPeriod/(slotsPerOrbit-1);

depSlotState = evaluate_state_at_epoch( ...
    legacyTable.time{legacyDepIndex}, ...
    legacyTable.state{legacyDepIndex},depSlotEpoch);
arrSlotState = evaluate_state_at_epoch( ...
    legacyTable.time{legacyArrIndex}, ...
    legacyTable.state{legacyArrIndex},arrSlotEpoch);
legacyInclusiveDepSlotState = evaluate_state_at_epoch( ...
    legacyTable.time{legacyDepIndex}, ...
    legacyTable.state{legacyDepIndex}, ...
    legacyInclusiveDepSlotEpoch);

catalogData = load(catalogPath,"T");
T = catalogData.T;
periods = T.("Period (TU) ");

[depIndex,departureStateError] = ...
    find_state_match(T,legacyDepState0);
[arrIndex,arrivalStateError] = ...
    find_state_match(T,legacyArrState0);

assert(departureStateError <= 1e-12, ...
    'Old row 52 did not match the rebuilt catalog.');
assert(arrivalStateError <= 1e-12, ...
    'Old row 400 did not match the rebuilt catalog.');

tableNames = string(T.Properties.VariableNames);
assert(ismember("orbitID",tableNames), ...
    'The rebuilt catalog does not contain orbitID.');

depOrbitID = lower(strtrim(string(T.orbitID(depIndex))));
arrOrbitID = lower(strtrim(string(T.orbitID(arrIndex))));

assert(depOrbitID==expectedDepID, ...
    'Old row 52 matched %s instead of %s.', ...
    char(depOrbitID),char(expectedDepID));
assert(arrOrbitID==expectedArrID, ...
    'Old row 400 matched %s instead of %s.', ...
    char(arrOrbitID),char(expectedArrID));

currentDepSlotState = evaluate_state_at_epoch( ...
    T.time{depIndex},T.state{depIndex}, ...
    (depSlot-1)*periods(depIndex)/slotsPerOrbit);
currentArrSlotState = evaluate_state_at_epoch( ...
    T.time{arrIndex},T.state{arrIndex}, ...
    (arrSlot-1)*periods(arrIndex)/slotsPerOrbit);

departureSlotStateError = norm(currentDepSlotState-depSlotState);
arrivalSlotStateError = norm(currentArrSlotState-arrSlotState);

assert(departureSlotStateError <= 1e-10, ...
    ['Current departure slot does not match old row 52, slot 10. ' ...
     'State error = %.6e.'],departureSlotStateError);
assert(arrivalSlotStateError <= 1e-10, ...
    ['Current arrival slot does not match old row 400, slot 1. ' ...
     'State error = %.6e.'],arrivalSlotStateError);

transferRef = struct();
transferRef.slotDefinition = slotDefinition;
transferRef.slotsPerOrbit = slotsPerOrbit;

transferRef.dep.legacyIndex = legacyDepIndex;
transferRef.dep.newIndex = depIndex;
transferRef.dep.slot = depSlot;
transferRef.dep.state0 = legacyDepState0;
transferRef.dep.slotEpoch = depSlotEpoch;
transferRef.dep.slotState = depSlotState;
transferRef.dep.legacyInclusiveSlotEpoch = ...
    legacyInclusiveDepSlotEpoch;
transferRef.dep.legacyInclusiveSlotState = ...
    legacyInclusiveDepSlotState;
transferRef.dep.period = periods(depIndex);
transferRef.dep.family = T.orbitFamily(depIndex);
transferRef.dep.orbitID = depOrbitID;
transferRef.dep.stateMatchError = departureStateError;
transferRef.dep.slotStateMatchError = departureSlotStateError;

transferRef.arr.legacyIndex = legacyArrIndex;
transferRef.arr.newIndex = arrIndex;
transferRef.arr.slot = arrSlot;
transferRef.arr.state0 = legacyArrState0;
transferRef.arr.slotEpoch = arrSlotEpoch;
transferRef.arr.slotState = arrSlotState;
transferRef.arr.period = periods(arrIndex);
transferRef.arr.family = T.orbitFamily(arrIndex);
transferRef.arr.orbitID = arrOrbitID;
transferRef.arr.stateMatchError = arrivalStateError;
transferRef.arr.slotStateMatchError = arrivalSlotStateError;

save(referencePath,"transferRef");

fprintf('Saved state-matched low-thrust reference to:\n  %s\n', ...
    referencePath);
fprintf([ ...
    'Departure: old row %d -> current row %d, slot %d, %s, ' ...
    'orbit-IC error %.3e, slot-IC error %.3e\n'], ...
    legacyDepIndex,depIndex,depSlot,char(depOrbitID), ...
    departureStateError,departureSlotStateError);
fprintf([ ...
    'Arrival:   old row %d -> current row %d, slot %d, %s, ' ...
    'orbit-IC error %.3e, slot-IC error %.3e\n'], ...
    legacyArrIndex,arrIndex,arrSlot,char(arrOrbitID), ...
    arrivalStateError,arrivalSlotStateError);
fprintf('Corrected departure slot epoch: %.15g TU (9T/50)\n', ...
    depSlotEpoch);
fprintf('Legacy departure slot epoch:    %.15g TU (9T/49)\n', ...
    legacyInclusiveDepSlotEpoch);
fprintf('Corrected departure slot state [x y z vx vy vz]:\n');
fprintf('  %.15g  %.15g  %.15g  %.15g  %.15g  %.15g\n', ...
    depSlotState);


function state = evaluate_state_at_epoch(time,stateHistory,epoch)

[uniqueTime,uniqueIndex] = unique(time);
uniqueState = stateHistory(uniqueIndex,:);
interpolant = griddedInterpolant(uniqueTime,uniqueState,'spline');
state = reshape(interpolant(epoch),1,[]);
end


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
