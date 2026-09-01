function transferRef = low_thrust_case_config(T)
%LOW_THRUST_CASE_CONFIG Reproducible fixed-endpoint transfer definition.
%
% The low-thrust target uses the exact states from the original study.
% Catalog rows and slots are retained only as provenance. Observer
% candidates continue to use 50 equal-time slots over [0,T).

slotsPerOrbit = 50;
legacyDepIndex = 52;
legacyArrIndex = 400;
depSlot = 10;
arrSlot = 1;

depOrbitID = "southern_halo_l1:1015";
arrOrbitID = "southern_halo_l2:97";

% Orbit initial states identify the corresponding current catalog rows.
depOrbitState0 = [ ...
     0.8402957900765589,  6.021304054218165e-28, ...
    -0.1583034195863712, -7.135606147838344e-16, ...
     0.2616335345519606,  9.179544887747458e-16];
arrOrbitState0 = [ ...
     1.0740681350221752,   3.2857158725587851e-27, ...
    -0.20204469729197141,  8.9272842881401699e-15, ...
    -0.19102742171285914, -9.1945220211817261e-15];

% Authoritative low-thrust boundary states. The departure is old row 52,
% slot 10 from linspace(0,T,50), so its phase is 9/49. Arrival slot 1 is
% t=0 under both the legacy and corrected slot definitions.
fixedDepartureState = [ ...
     0.8688395541375723,   0.1110680873881317, ...
    -0.10760863551490674,  0.10657734318058584, ...
     0.14953221747069609,  0.19541894435638577];
fixedArrivalState = [ ...
     1.0740681350221752,   3.2857158725587851e-27, ...
    -0.20204469729197141,  8.9272842881401699e-15, ...
    -0.19102742171285914, -9.1945220211817261e-15];

[depIndex,depStateError] = find_state_match(T,depOrbitState0);
[arrIndex,arrStateError] = find_state_match(T,arrOrbitState0);

assert(depStateError <= 1e-12, ...
    'The old row-52 departure orbit is missing from the catalog.');
assert(arrStateError <= 1e-12, ...
    'The old row-400 arrival orbit is missing from the catalog.');
assert(ismember('orbitID',T.Properties.VariableNames), ...
    'The catalog does not contain orbitID.');

resolvedDepID = lower(strtrim(string(T.orbitID(depIndex))));
resolvedArrID = lower(strtrim(string(T.orbitID(arrIndex))));
assert(resolvedDepID==depOrbitID, ...
    'Departure state resolved to %s instead of %s.', ...
    char(resolvedDepID),char(depOrbitID));
assert(resolvedArrID==arrOrbitID, ...
    'Arrival state resolved to %s instead of %s.', ...
    char(resolvedArrID),char(arrOrbitID));

periods = T.("Period (TU) ");
depPeriod = periods(depIndex);
arrPeriod = periods(arrIndex);

% Corrected observer-slot states are included only to audit the catalog.
depObserverSlotEpoch = (depSlot-1)*depPeriod/slotsPerOrbit;
arrObserverSlotEpoch = (arrSlot-1)*arrPeriod/slotsPerOrbit;
depObserverSlotState = reconstruct_state( ...
    T.time{depIndex},T.state{depIndex},depObserverSlotEpoch);
arrObserverSlotState = reconstruct_state( ...
    T.time{arrIndex},T.state{arrIndex},arrObserverSlotEpoch);

transferRef = struct();
transferRef.slotDefinition = "equal_time_no_endpoint_v1";
transferRef.targetStateDefinition = "explicit_fixed_states_v1";
transferRef.slotsPerOrbit = slotsPerOrbit;

transferRef.dep.legacyIndex = legacyDepIndex;
transferRef.dep.newIndex = depIndex;
transferRef.dep.slot = depSlot;
transferRef.dep.state0 = depOrbitState0;
transferRef.dep.slotEpoch = depObserverSlotEpoch;
transferRef.dep.slotState = depObserverSlotState;
transferRef.dep.transferEpoch = ...
    (depSlot-1)*depPeriod/(slotsPerOrbit-1);
transferRef.dep.transferState = fixedDepartureState;
transferRef.dep.period = depPeriod;
transferRef.dep.family = T.orbitFamily(depIndex);
transferRef.dep.orbitID = resolvedDepID;
transferRef.dep.stateMatchError = depStateError;

transferRef.arr.legacyIndex = legacyArrIndex;
transferRef.arr.newIndex = arrIndex;
transferRef.arr.slot = arrSlot;
transferRef.arr.state0 = arrOrbitState0;
transferRef.arr.slotEpoch = arrObserverSlotEpoch;
transferRef.arr.slotState = arrObserverSlotState;
transferRef.arr.transferEpoch = 0;
transferRef.arr.transferState = fixedArrivalState;
transferRef.arr.period = arrPeriod;
transferRef.arr.family = T.orbitFamily(arrIndex);
transferRef.arr.orbitID = resolvedArrID;
transferRef.arr.stateMatchError = arrStateError;
end


function [index,minimumDistance] = find_state_match(T,referenceState)

initialStates = zeros(height(T),6);
for k = 1:height(T)
    initialStates(k,:) = T.state{k}(1,:);
end

distance = vecnorm(initialStates-referenceState,2,2);
[minimumDistance,index] = min(distance);
end


function state = reconstruct_state(time,stateHistory,epoch)

[uniqueTime,uniqueIndex] = unique(time);
interpolant = griddedInterpolant( ...
    uniqueTime,stateHistory(uniqueIndex,:),'spline');
state = reshape(interpolant(epoch),1,[]);
end
