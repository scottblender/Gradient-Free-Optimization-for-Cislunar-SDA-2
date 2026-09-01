function result = test_gateway_impulse_case()
% End-to-end truth and 120-FE optimization pilot for Gateway impulse case.

truthAudit = test_gateway_impulse_truth();

summary = test_small_fe_case( ...
    "GA", 0, "GATEWAY_IMPULSE");

assert(height(summary) == 1, ...
    'Expected exactly one Gateway impulse pilot result.');

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

assert(string(info.type) == "GATEWAY_IMPULSE", ...
    'The saved truth is not a Gateway impulse trajectory.');
assert(abs(info.deltaV_m_s - 10) <= 1e-12, ...
    'The saved impulse magnitude is not 10 m/s.');
assert(string(info.direction) == "PROGRADE", ...
    'The saved impulse direction is not PROGRADE.');
assert(abs(info.duration_TU - 1.5) <= 1e-12, ...
    'The saved tracking duration is not 1.5 TU.');
assert(norm(info.statePostImpulse(1:3) - ...
    info.statePreImpulse(1:3)) <= 1e-14, ...
    'Position changed across the instantaneous impulse.');
assert(norm(info.statePostImpulse(4:6) - ...
    info.statePreImpulse(4:6) - ...
    info.deltaVVector_LU_TU) <= 1e-14, ...
    'Saved pre/post states do not match the impulse.');

result = struct();
result.truthAudit = truthAudit;
result.summary = summary;
result.truthInfo = info;
result.bestJ = runState.bestJ;
result.runDir = string(summary.runDir(1));

fprintf('\n--- Gateway impulse pilot results ---\n');
fprintf('Perilune epoch:      %.12f TU\n', ...
    info.periluneEpochNominal_TU);
fprintf('Impulse:             %.3f m/s %s\n', ...
    info.deltaV_m_s, info.direction);
fprintf('Tracking duration:   %.3f TU\n', ...
    info.duration_TU);
fprintf('Search evaluations:  %d\n', ...
    summary.searchFE(1));
fprintf('Best objective:      %.12g\n', ...
    runState.bestJ);
fprintf('Run directory:\n  %s\n', result.runDir);
fprintf('\nGateway impulse pilot passed.\n');
end
