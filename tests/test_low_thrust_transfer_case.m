function summary = test_low_thrust_transfer_case()
% End-to-end 120-FE pilot for the ID-resolved low-thrust transfer.
% The test checks the catalog references before optimization, then verifies
% transfer-solver convergence and the normal FE/history audit.

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
paths = setup_project();

catalogData = load(paths.catalog, 'T');
T = catalogData.T;

referencePath = fullfile(paths.data, 'transfer_reference.mat');
assert(isfile(referencePath), ...
    'Transfer reference file was not found: %s', referencePath);

referenceData = load(referencePath, 'transferRef');
transferRef = referenceData.transferRef;

assert(ismember('orbitID', T.Properties.VariableNames), ...
    'The catalog does not contain orbitID.');

orbitIDs = string(T.orbitID);
assert(numel(unique(orbitIDs)) == height(T), ...
    'The catalog orbitID values are not unique.');

if ismember('period_TU', T.Properties.VariableNames)
    periods = T.period_TU;
else
    periods = T.("Period (TU) ");
end

[depIndex, depStateError, depPeriodError] = check_reference( ...
    'Departure', transferRef.dep, T, orbitIDs, periods);

[arrIndex, arrStateError, arrPeriodError] = check_reference( ...
    'Arrival', transferRef.arr, T, orbitIDs, periods);

assert(transferRef.dep.slot == 10, ...
    'The departure slot changed from 10.');
assert(transferRef.arr.slot == 1, ...
    'The arrival slot changed from 1.');
assert(depIndex ~= arrIndex, ...
    'Departure and arrival resolved to the same orbit.');

fprintf('\n--- Low-thrust transfer pilot ---\n');
fprintf('Departure row/slot: %d/%d\n', ...
    depIndex, transferRef.dep.slot);
fprintf('Arrival row/slot:   %d/%d\n', ...
    arrIndex, transferRef.arr.slot);
fprintf('Departure orbitID:  %s\n', transferRef.dep.orbitID);
fprintf('Arrival orbitID:    %s\n\n', transferRef.arr.orbitID);

summary = test_small_fe_case( ...
    "GA", 0, "LOW_THRUST_TRANSFER");

assert(height(summary) == 1, ...
    'Expected exactly one pilot result.');

checkNames = [
    "searchBudgetOK"
    "solverCallPatternOK"
    "historyOK"
    "bestOK"
    "recheckOK"
];

for k = 1:numel(checkNames)
    values = summary.(char(checkNames(k)));
    assert(all(logical(values)), ...
        'Pilot check failed: %s.', checkNames(k));
end

assert(string(summary.termination(1)) == "budget_reached", ...
    'The pilot did not terminate at the FE budget.');

resultPath = fullfile( ...
    char(summary.runDir(1)), ...
    'data', 'optimization_run.mat');

assert(isfile(resultPath), ...
    'Pilot result file was not found: %s', resultPath);

saved = load(resultPath, 'runState');
runState = saved.runState;
info = runState.truthInfo;

assert(string(info.type) == "LOW_THRUST_TRANSFER", ...
    'The saved truth is not a low-thrust transfer.');
assert(info.depOrbitIndex == depIndex, ...
    'The transfer used the wrong departure orbit row.');
assert(info.arrOrbitIndex == arrIndex, ...
    'The transfer used the wrong arrival orbit row.');
assert(info.depSlot == transferRef.dep.slot, ...
    'The transfer used the wrong departure slot.');
assert(info.arrSlot == transferRef.arr.slot, ...
    'The transfer used the wrong arrival slot.');
assert(info.exitflag > 0, ...
    'The low-thrust fsolve call did not converge.');
assert(isfinite(info.finalResidualNorm) && ...
    info.finalResidualNorm <= 1e-8, ...
    'Low-thrust final residual norm is too large: %.6e', ...
    info.finalResidualNorm);
assert(isfinite(info.tf) && info.tf > 0, ...
    'Low-thrust transfer time is invalid.');
assert(isfinite(info.mass_final) && info.mass_final > 0, ...
    'Low-thrust final mass is invalid.');

fprintf('\n--- Low-thrust results ---\n');
fprintf('Solver exit flag:       %d\n', info.exitflag);
fprintf('Final residual norm:     %.6e\n', ...
    info.finalResidualNorm);
fprintf('Transfer time:           %.6f TU\n', info.tf);
fprintf('Final mass:              %.6f\n', info.mass_final);
fprintf('Search evaluations:      %d\n', ...
    summary.searchFE(1));
fprintf('Best objective:          %.12g\n', ...
    runState.bestJ);
fprintf('Departure state error:   %.6e\n', depStateError);
fprintf('Arrival state error:     %.6e\n', arrStateError);
fprintf('Departure period error:  %.6e TU\n', depPeriodError);
fprintf('Arrival period error:    %.6e TU\n', arrPeriodError);

fprintf('\nLow-thrust transfer pilot passed.\n');
end


function [index, stateError, periodError] = check_reference( ...
    label, reference, T, orbitIDs, periods)

requiredFields = {'orbitID','state0','period','slot'};

for k = 1:numel(requiredFields)
    assert(isfield(reference, requiredFields{k}), ...
        '%s reference is missing %s.', ...
        label, requiredFields{k});
end

matches = find(orbitIDs == string(reference.orbitID));

assert(numel(matches) == 1, ...
    '%s orbitID matched %d catalog rows.', ...
    label, numel(matches));

index = matches(1);

state0 = reshape(reference.state0, 1, []);
stateError = norm(T.state{index}(1,:) - state0);
periodError = abs(periods(index) - reference.period);

assert(stateError <= 1e-10, ...
    '%s state error is too large: %.6e.', ...
    label, stateError);
assert(periodError <= 1e-12, ...
    '%s period error is too large: %.6e TU.', ...
    label, periodError);

if isfield(reference, 'newIndex')
    assert(reference.newIndex == index, ...
        '%s stored newIndex does not match orbitID.', label);
end
end
