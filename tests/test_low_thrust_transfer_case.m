function summary = test_low_thrust_transfer_case()
% End-to-end 120-FE pilot for the fixed-boundary low-thrust transfer.

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
setup_project();

missionCfg = target_case_config("LOW_THRUST_TRANSFER");
transferCfg = missionCfg.transfer;

assert(numel(transferCfg.fixedDepartureState)==6 && numel(transferCfg.fixedTargetState)==6, ...
    'Fixed low-thrust boundary states must each have six elements.');
assert(all(isfinite(transferCfg.fixedDepartureState)) && all(isfinite(transferCfg.fixedTargetState)), ...
    'Fixed low-thrust boundary states contain nonfinite values.');

fprintf('\n--- Low-thrust transfer pilot ---\n');
fprintf('Target definition: fixed departure and arrival states.\n');
summary = test_small_fe_case("GA", 0, "LOW_THRUST_TRANSFER");
assert(height(summary) == 1, 'Expected exactly one pilot result.');

checkNames = ["searchBudgetOK";"solverCallPatternOK";"historyOK";"bestOK";"recheckOK"];
for k = 1:numel(checkNames)
    assert(all(logical(summary.(char(checkNames(k))))), 'Pilot check failed: %s.', checkNames(k));
end
assert(string(summary.termination(1)) == "budget_reached", 'The pilot did not terminate at the FE budget.');

resultPath = fullfile(char(summary.runDir(1)), 'data', 'optimization_run.mat');
assert(isfile(resultPath), 'Pilot result file was not found: %s', resultPath);
saved = load(resultPath, 'runState'); runState = saved.runState; info = runState.truthInfo;

assert(string(info.type) == "LOW_THRUST_TRANSFER", 'The saved truth is not a low-thrust transfer.');
assert(string(info.endpointDefinition) == "FIXED_STATES", 'The transfer truth is not identified as a fixed-state case.');
assert(string(info.departureStateSource) == "FIXED_STATE" && string(info.arrivalStateSource) == "FIXED_STATE", 'The transfer solver did not use fixed boundary states.');
assert(norm(info.x_dep(:)-transferCfg.fixedDepartureState(:)) <= 1e-12, 'The solver departure state differs from TargetCaseDatabase.');
assert(norm(info.x_arr(:)-transferCfg.fixedTargetState(:)) <= 1e-12, 'The solver arrival state differs from TargetCaseDatabase.');
assert(info.exitflag > 0, 'The low-thrust fsolve call did not converge.');
assert(isfinite(info.finalResidualNorm) && info.finalResidualNorm <= 1e-8, 'Low-thrust final residual norm is too large: %.6e', info.finalResidualNorm);
assert(isfinite(info.tf) && info.tf > 0, 'Low-thrust transfer time is invalid.');
assert(isfinite(info.mass_final) && info.mass_final > 0, 'Low-thrust final mass is invalid.');

fprintf('\n--- Low-thrust results ---\n');
fprintf('Solver exit flag:       %d\n', info.exitflag);
fprintf('Final residual norm:     %.6e\n', info.finalResidualNorm);
fprintf('Transfer time:           %.6f TU\n', info.tf);
fprintf('Final mass:              %.6f\n', info.mass_final);
fprintf('Search evaluations:      %d\n', summary.searchFE(1));
fprintf('Best objective:          %.12g\n', runState.bestJ);
fprintf('\nLow-thrust transfer pilot passed.\n');
end
