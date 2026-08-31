function test_evaluation_tracker()
% No orbit catalog, EKF, or optimization solver required.

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
setup_project;

calls = 0;
tracker = create_evaluation_tracker(@toy_objective, 5);

values = [3, 1, 1, 4, 2];
rngBefore = rng;

for k = 1:numel(values)
    tracker.evaluate([values(k), 1]);
end

s = tracker.snapshot();

assert(calls == 5 && s.nEvaluations == 5);
assert(isequal([s.history.fe], 1:5));
assert(isequal([s.history.J_total], [9,1,1,16,4]));
assert(isequal([s.history.bestJ], [9,1,1,1,1]));
assert(isequal(s.bestX, [1,1]) && s.bestJ == 1);
assert(tracker.shouldStop());
assert(isequal(rngBefore, rng));

% A sixth request must not execute the objective.
check_error(@() tracker.evaluate([0,1]), ...
    'EvaluationTracker:BudgetReached');

s = tracker.snapshot();

assert(calls == 5);
assert(s.nEvaluations == 5);
assert(numel(s.history) == 5);

% A new tracker must start empty.
fresh = create_evaluation_tracker(@toy_objective, 5);
s = fresh.snapshot();

assert(s.nEvaluations == 0 && isempty(s.bestX));

% An unexpected failure must remain an error.
badCalls = 0;
bad = create_evaluation_tracker(@bad_objective, 5);

check_error(@() bad.evaluate([2,1]), ...
    'Test:ExpectedFailure');

s = bad.snapshot();

assert(s.nEvaluations == 1);
assert(s.history(1).status == "failed");
assert(strcmp(s.failure.identifier, 'Test:ExpectedFailure'));
assert(bad.shouldStop());

% Even if a solver catches the error, further calls cannot resume.
check_error(@() bad.evaluate([2,1]), ...
    'Test:ExpectedFailure');

assert(badCalls == 1);

% Nonfinite costs cannot become the incumbent.
invalid = create_evaluation_tracker(@nonfinite_objective, 2);
didThrow = false;

try
    invalid.evaluate([1,1]);
catch
    didThrow = true;
end

s = invalid.snapshot();

assert(didThrow && s.nEvaluations == 1);
assert(isempty(s.bestX));
assert(s.history(1).status == "failed");

fprintf('Evaluation tracker tests passed.\n');

    function [J, entry] = toy_objective(x)
        calls = calls + 1;
        J = x(1)^2;

        entry = struct( ...
            'x',x, ...
            'J1_rmse',J, ...
            'J2_det',0, ...
            'J3_stab',0);
    end

    function [J, entry] = bad_objective(~)
        J = NaN;
        entry = struct(); %#ok<NASGU>

        badCalls = badCalls + 1;
        error('Test:ExpectedFailure', 'Deliberate test failure.');
    end

    function [J, entry] = nonfinite_objective(x)
        J = NaN;
        entry = struct('x',x);
    end
end

function check_error(fcn, expectedId)

caught = false;

try
    fcn();
catch ME
    caught = true;
    assert(strcmp(ME.identifier, expectedId), ...
        'Unexpected error: %s', ME.identifier);
end

assert(caught, 'Expected an error, but none was raised.');
end