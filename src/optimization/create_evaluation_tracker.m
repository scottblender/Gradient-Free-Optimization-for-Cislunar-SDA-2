function tracker = create_evaluation_tracker(rawObjFcn, maxEvals)
% Serial, single-objective evaluation counter and history.
% rawObjFcn returns [J, details], with details.x the evaluated design.

validateattributes(maxEvals, {'numeric'}, ...
    {'scalar','real','finite','integer','positive'});

nEvaluations = 0;
bestJ = Inf;
bestX = [];
failure = [];
runTimer = tic;

template = struct( ...
    'fe',0, ...
    'x',[], ...
    'orbit_indices',[], ...
    'slot_indices',[], ...
    'J_total',NaN, ...
    'J1_rmse',NaN, ...
    'J2_det',NaN, ...
    'J3_stab',NaN, ...
    'bestJ',Inf, ...
    'elapsed_s',0, ...
    'evaluation_s',0, ...
    'status',"", ...
    'error_id',"", ...
    'error_message',"");

history = repmat(template, maxEvals, 1);
details = cell(maxEvals, 1);

tracker.evaluate = @evaluate;
tracker.snapshot = @snapshot;
tracker.shouldStop = @should_stop;
tracker.getFailure = @get_failure;

    function J = evaluate(x)

        % Do not resume evaluations after an objective error.
        if ~isempty(failure)
            rethrow(failure);
        end

        % Check BEFORE executing the objective.
        if nEvaluations >= maxEvals
            error('EvaluationTracker:BudgetReached', ...
                'The objective evaluation budget has been reached.');
        end

        if istable(x)
            x = table2array(x);
        end
        x = x(:).';

        nEvaluations = nEvaluations + 1;
        evalTimer = tic;

        entry = template;
        entry.fe = nEvaluations;
        entry.x = round(x);

        try
            [J, info] = rawObjFcn(x);

            validateattributes(J, {'numeric'}, ...
                {'scalar','real','finite'});

            validateattributes(info.x, {'numeric'}, ...
                {'vector','real','finite','nonempty'});

            entry.x = info.x(:).';
            entry.J_total = J;

            names = {'J1_rmse','J2_det','J3_stab'};
            for j = 1:numel(names)
                if isfield(info, names{j})
                    entry.(names{j}) = info.(names{j});
                end
            end

            details{nEvaluations} = info;

            if J < bestJ
                bestJ = J;
                bestX = entry.x;
            end

            entry.status = "ok";

        catch ME
            failure = ME;
            entry.status = "failed";
            entry.error_id = string(ME.identifier);
            entry.error_message = string(ME.message);
        end

        entry.orbit_indices = entry.x(1:2:end);
        entry.slot_indices = entry.x(2:2:end);
        entry.bestJ = bestJ;
        entry.elapsed_s = toc(runTimer);
        entry.evaluation_s = toc(evalTimer);

        history(nEvaluations) = entry;

        if ~isempty(failure)
            rethrow(failure);
        end
    end

    function state = snapshot()

        state.maxEvaluations = maxEvals;
        state.nEvaluations = nEvaluations;
        state.bestX = bestX;
        state.bestJ = bestJ;
        state.history = history(1:nEvaluations);
        state.details = details(1:nEvaluations);
        state.failure = [];

        if ~isempty(failure)
            state.failure = struct( ...
                'identifier',failure.identifier, ...
                'message',failure.message, ...
                'report',getReport(failure, ...
                    'extended','hyperlinks','off'));
        end
    end

    function stop = should_stop()
        stop = nEvaluations >= maxEvals || ~isempty(failure);
    end

    function ME = get_failure()
        ME = failure;
    end
end