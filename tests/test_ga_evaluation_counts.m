function audit = test_ga_evaluation_counts(budget)
% Trace actual GA objective calls without orbit propagation or EKF work.
% Run: audit = test_ga_evaluation_counts(120)
% Mirrors the integer GA options in run_opt, using a cheap deterministic
% objective and serial evaluation so call ordering is unambiguous.
% Creates no figures/files. Restores the caller's RNG state.
if nargin < 1, budget = 120; end
validateattributes(budget,{'numeric'}, ...
    {'scalar','finite','integer','>=',120});
pop = 60;
assert(mod(budget,pop) == 0,'Budget must be divisible by 60.');

oldRng = rng;
cleanup = onCleanup(@() rng(oldRng)); %#ok<NASGU>
rng(0,'twister');

lb = repmat([1 1],1,3);
ub = repmat([450 50],1,3); % Synthetic search bounds, not a catalog load.
goal = [75 12 225 28 375 42];
calls = 0;
stage = "before_init";
X = zeros(0,6);
J = zeros(0,1);
phase = strings(0,1);
stackNames = strings(0,1);
callbackFlag = strings(0,1);
callbackGeneration = zeros(0,1);
callbackFE = zeros(0,1);
callbackCalls = zeros(0,1);
lastSearchCalls = 0;
lastSearchFE = 0;

options = optimoptions('ga', ...
    'UseParallel',false,'UseVectorized',false, ...
    'Display','off','PopulationSize',pop,'EliteCount',0, ...
    'MaxGenerations',budget/pop-1,'MaxStallGenerations',Inf, ...
    'FunctionTolerance',0,'ConstraintTolerance',0, ...
    'FitnessLimit',-Inf,'OutputFcn',@record_callback,'PlotFcn',[]);

[bestX,bestJ,exitFlag,output] = ga(@record_objective,6, ...
    [],[],[],[],lb,ub,[],1:6,options);

% Identify actual repeated candidate vectors, without changing the search.
duplicateOf = nan(calls,1);
for k = 2:calls
    prior = find(all(X(1:k-1,:) == X(k,:),2),1,'first');
    if ~isempty(prior), duplicateOf(k) = prior; end
end

audit = struct();
audit.requestedBudget = budget;
audit.lastSearchCallbackFE = lastSearchFE;
audit.actualCallsAtLastSearchCallback = lastSearchCalls;
audit.actualObjectiveCalls = calls;
audit.solverReportedCalls = output.funccount;
audit.callsAfterLastSearchCallback = calls-lastSearchCalls;
audit.reportedMinusActualCalls = output.funccount-calls;
audit.exitFlag = exitFlag;
audit.bestX = bestX;
audit.bestJ = bestJ;
audit.matlabVersion = string(version);
audit.callbacks = table(callbackFlag,callbackGeneration,callbackFE, ...
    callbackCalls,'VariableNames', ...
    {'flag','generation','reportedFE','actualCalls'});
audit.calls = table((1:calls)',phase,J,duplicateOf,X,stackNames, ...
    'VariableNames',{'call','phase','J','duplicateOf','x','stack'});
audit.trailingCalls = audit.calls(lastSearchCalls+1:end,:);

fprintf('\n--- GA evaluation-count audit (cheap serial objective) ---\n');
fprintf('Requested search budget:                 %d\n',budget);
fprintf('Last search callback FE:                 %d\n',lastSearchFE);
fprintf('Actual calls at last search callback:    %d\n',lastSearchCalls);
fprintf('Actual objective calls:                  %d\n',calls);
fprintf('GA-reported objective calls:             %d\n',output.funccount);
fprintf('Calls after last search callback:        %d\n',calls-lastSearchCalls);
fprintf('Reported minus actual calls:             %d\n',output.funccount-calls);
fprintf('\nCallback sequence:\n');
disp(audit.callbacks);
fprintf('\nLast objective calls (duplicateOf refers to an earlier call):\n');
disp(audit.calls(max(1,calls-4):end,{'call','phase','J','duplicateOf'}));
if ~isempty(audit.trailingCalls)
    fprintf('\nCalls after the last search callback:\n');
    disp(audit.trailingCalls(:,{'call','phase','J','duplicateOf','stack'}));
end
fprintf('Full candidate vectors and call stacks are in the returned audit struct.\n');

    function value = record_objective(x)
        calls = calls+1;
        value = sum(((x-goal)./(ub-lb)).^2);
        X(calls,:) = x;
        J(calls,1) = value;
        phase(calls,1) = stage;
        stack = dbstack;
        stackNames(calls,1) = strjoin(string({stack.name})," > ");
    end

    function [state,opts,optchanged] = record_callback(opts,state,flag)
        optchanged = false;
        callbackFlag(end+1,1) = string(flag);
        callbackGeneration(end+1,1) = state.Generation;
        callbackFE(end+1,1) = state.FunEval;
        callbackCalls(end+1,1) = calls;
        if strcmp(flag,'init') || strcmp(flag,'iter')
            lastSearchCalls = calls;
            lastSearchFE = state.FunEval;
        end
        stage = "after_"+string(flag)+"_"+string(state.Generation);
        if state.FunEval >= budget
            state.StopFlag = 'Function evaluation budget reached';
        end
    end
end
