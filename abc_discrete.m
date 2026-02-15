function [xval, fval] = abc_discrete(ObjFcn, LB, UB, opts, stopper)
% --- abc_discrete.m --- %
% Discrete Artificial Bee Colony (ABC) optimizer for INTEGER decision vectors.
%
% Assumption (your design encoding):
%   x is ALWAYS orbit/slot PAIRS:
%       x = [orb1 slot1 orb2 slot2 ...]
%   so numel(x) must be even.
%
% Parallel-safe pattern:
%   - Build candidate batches on the CLIENT (serial).
%   - Evaluate candidates with PARFOR if opts.UseParallel = true.
%   - Apply greedy accept/reject on the CLIENT (serial).
%
% Stopper usage:
%   - Stopper is updated ONLY on the CLIENT using a true global eval counter.
%   - ObjFcn MUST be side-effect free when running in parallel
%     (do not call stopper.eval, do not mutate shared state inside ObjFcn).
%
% Inputs:
%   ObjFcn  : function handle, J = ObjFcn(x)
%   LB/UB   : 1xn bounds (integer-valued after rounding)
%   opts    : struct with fields below
%   stopper : Stopper handle (optional)
%
% opts fields (defaults):
%   ColonySize   (60)  -> must be even; nFood = ColonySize/2
%   MaxIters     (80)  -> number of ABC iterations
%   Limit        (20)  -> scout trigger (trial counter threshold)
%   UseParallel  (true)-> use parfor for batch evaluations
%
% Outputs:
%   xval : best solution found
%   fval : best objective value found

% --- set defaults --- %
if nargin < 4 || isempty(opts), opts = struct(); end
if ~isfield(opts,'ColonySize') || isempty(opts.ColonySize), opts.ColonySize = 60; end
if ~isfield(opts,'MaxIters')   || isempty(opts.MaxIters),   opts.MaxIters   = 80; end
if ~isfield(opts,'Limit')      || isempty(opts.Limit),      opts.Limit      = 20; end
if ~isfield(opts,'UseParallel')|| isempty(opts.UseParallel),opts.UseParallel= true; end

useStopper = (nargin >= 5) && ~isempty(stopper);

assert(mod(opts.ColonySize,2)==0, 'ColonySize must be even.');
nFood = opts.ColonySize/2;

nVars = numel(LB);
assert(numel(UB) == nVars, 'LB and UB must be the same length.');
assert(mod(nVars,2)==0, 'Decision vector must be even length: [orb1 slot1 orb2 slot2 ...].');

% round + clamp helper (keeps decisions integer-feasible)
roundClamp = @(x) max(LB, min(UB, round(x)));

% ==========================
% initialize food sources
% ==========================
Foods = zeros(nFood, nVars);
for i = 1:nFood
    Foods(i,:) = roundClamp(LB + rand(1,nVars).*(UB-LB));
    Foods(i,:) = repair_pairs(Foods(i,:), LB, UB); % enforce orbit/slot pairing constraints (if any)
end

% --- evaluate initial foods (batch) --- %
costs = batch_eval(ObjFcn, Foods, opts.UseParallel);

% --- initialize global best --- %
[fval, idxBest] = min(costs);
xval = Foods(idxBest,:);

% trials(i) = # consecutive failures to improve food i
trials = zeros(nFood,1);

% authoritative eval counter
evalTotal = nFood;

% update stopper after initialization batch
if useStopper
    stopper.update(fval, evalTotal, xval);
    if stopper.stopFlag
        fprintf('ABC stopping early: %s\n', stopper.stopReason);
        return;
    end
end

% ==========================
% main ABC loop
% ==========================
for itr = 1:opts.MaxIters

    % allow external stop (e.g., stall / max evals)
    if useStopper && stopper.stopFlag
        fprintf('ABC stopping early: %s\n', stopper.stopReason);
        break;
    end

    % --------------------------
    % employed bee phase
    % --------------------------
    V = zeros(nFood, nVars);
    for i = 1:nFood
        V(i,:) = abc_neighbor_discrete(Foods(i,:), Foods, LB, UB, i);
        V(i,:) = roundClamp(V(i,:));
        V(i,:) = repair_pairs(V(i,:), LB, UB);
    end

    costV = batch_eval(ObjFcn, V, opts.UseParallel);
    evalTotal = evalTotal + nFood;

    % greedy selection (client-side)
    for i = 1:nFood
        if costV(i) < costs(i)
            Foods(i,:) = V(i,:);
            costs(i)   = costV(i);
            trials(i)  = 0;
        else
            trials(i)  = trials(i) + 1;
        end
    end

    % update global best after employed phase
    [best_cost, idxBest] = min(costs);
    if best_cost < fval
        fval = best_cost;
        xval = Foods(idxBest,:);
    end

    if useStopper
        stopper.update(fval, evalTotal, xval);
        if stopper.stopFlag
            fprintf('ABC stopping early: %s\n', stopper.stopReason);
            break;
        end
    end

    % --------------------------
    % onlooker bee phase
    % --------------------------
    % Convert costs to selection probabilities (lower cost => higher probability)
    w = max(costs) - costs;     % bigger is better
    w = w - min(w);             % ensure nonnegative
    w = w + 1e-12;              % avoid all-zero vector
    prob = w / sum(w);
    cdf  = cumsum(prob);

    % pick nFood foods (with replacement) according to prob
    idxPick = zeros(nFood,1);
    for k = 1:nFood
        r = rand;
        idx = find(cdf >= r, 1, 'first');
        if isempty(idx), idx = nFood; end
        idxPick(k) = idx;
    end

    V2 = zeros(nFood, nVars);
    for k = 1:nFood
        i = idxPick(k);
        V2(k,:) = abc_neighbor_discrete(Foods(i,:), Foods, LB, UB, i);
        V2(k,:) = roundClamp(V2(k,:));
        V2(k,:) = repair_pairs(V2(k,:), LB, UB);
    end

    costV2 = batch_eval(ObjFcn, V2, opts.UseParallel);
    evalTotal = evalTotal + nFood;

    % greedy update (client-side) for the selected foods
    for k = 1:nFood
        i = idxPick(k);
        if costV2(k) < costs(i)
            Foods(i,:) = V2(k,:);
            costs(i)   = costV2(k);
            trials(i)  = 0;
        else
            trials(i)  = trials(i) + 1;
        end
    end

    % update global best after onlooker phase
    [best_cost, idxBest] = min(costs);
    if best_cost < fval
        fval = best_cost;
        xval = Foods(idxBest,:);
    end

    if useStopper
        stopper.update(fval, evalTotal, xval);
        if stopper.stopFlag
            fprintf('ABC stopping early: %s\n', stopper.stopReason);
            break;
        end
    end

    % --------------------------
    % scout bee phase
    % --------------------------
    scoutIdx = find(trials >= opts.Limit);
    nScout = numel(scoutIdx);

    if nScout > 0
        Snew = zeros(nScout, nVars);
        for k = 1:nScout
            Snew(k,:) = roundClamp(LB + rand(1,nVars).*(UB-LB));
            Snew(k,:) = repair_pairs(Snew(k,:), LB, UB);
        end

        costS = batch_eval(ObjFcn, Snew, opts.UseParallel);
        evalTotal = evalTotal + nScout;

        % replace abandoned foods
        for k = 1:nScout
            j = scoutIdx(k);
            Foods(j,:) = Snew(k,:);
            costs(j)   = costS(k);
            trials(j)  = 0;
        end

        % update global best after scouts
        [best_cost, idxBest] = min(costs);
        if best_cost < fval
            fval = best_cost;
            xval = Foods(idxBest,:);
        end

        if useStopper
            stopper.update(fval, evalTotal, xval);
            if stopper.stopFlag
                fprintf('ABC stopping early: %s\n', stopper.stopReason);
                break;
            end
        end
    end

    % --- progress --- %
    fprintf('ABC iter %3d | bestJ = %.6g | evals = %d\n', itr, fval, evalTotal);
end
end % abc_discrete


% =====================================================================
% helpers
% =====================================================================

function J = batch_eval(ObjFcn, X, usePar)
% --- batch_eval.m --- %
% Evaluate each row of X with ObjFcn, optionally using PARFOR.

n = size(X,1);
J = zeros(n,1);

if usePar
    parfor i = 1:n
        J(i) = ObjFcn(X(i,:));
    end
else
    for i = 1:n
        J(i) = ObjFcn(X(i,:));
    end
end
end


function v = abc_neighbor_discrete(x, Foods, LB, UB, selfIdx)
% --- abc_neighbor_discrete.m --- %
% Neighbor generation for INTEGER orbit/slot PAIR decision vectors.
%
% x: 1xn decision vector, interpreted as [orb1 slot1 orb2 slot2 ...]
% Foods: current food population (used to pick a partner solution)
%
% Move types (probabilistic):
%   1) Local slot tweak (same orbit)
%   2) Orbit perturb + random slot (paired)
%   3) Swap two orbit/slot pairs
%   4) ABC-style difference on a whole pair (coupled)
%   5) Random restart of one pair

assert(mod(numel(x),2)==0, 'x must be even length: [orb slot] pairs.');

v = x;
nPairs = numel(x)/2;

% pick a random (orbit,slot) pair
pair = randi(nPairs);
io = 2*pair - 1;   % orbit index
is = 2*pair;       % slot index

% pick a partner solution k != selfIdx
nFood = size(Foods,1);
k = randi(nFood);
if nFood > 1
    while k == selfIdx
        k = randi(nFood);
    end
end
xk = Foods(k,:);

r = rand;

if r < 0.40
    % (1) local slot tweak (same orbit)
    step = randi([-5 5]);
    v(is) = v(is) + step;

elseif r < 0.70
    % (2) orbit perturb + random slot (paired)
    dOrb = randi([-50 50]);
    v(io) = v(io) + dOrb;
    v(is) = randi([LB(is) UB(is)]);

elseif r < 0.85
    % (3) swap two orbit/slot pairs
    p2 = randi(nPairs);
    while p2 == pair
        p2 = randi(nPairs);
    end
    j1 = 2*pair-1;
    j2 = 2*p2-1;
    tmp = v(j1:j1+1);
    v(j1:j1+1) = v(j2:j2+1);
    v(j2:j2+1) = tmp;

elseif r < 0.95
    % (4) ABC-style difference on a WHOLE PAIR (coupled)
    phi = -1 + 2*rand;
    oldOrb = v(io);
    v(io) = v(io) + phi*(v(io) - xk(io));

    % if orbit changes after rounding, resample slot to keep pairing meaningful
    if round(v(io)) ~= round(oldOrb)
        v(is) = randi([LB(is) UB(is)]);
    else
        % otherwise allow mild slot nudge relative to partner
        phiS = -1 + 2*rand;
        v(is) = v(is) + phiS*(v(is) - xk(is));
    end

else
    % (5) restart one pair (uniform in bounds)
    v(io) = LB(io) + rand*(UB(io)-LB(io));
    v(is) = randi([LB(is) UB(is)]);
end

% clamp/round whole vector (keeps everything in bounds and integer)
v = max(LB, min(UB, round(v)));
end


function x = repair_pairs(x, LB, UB)
% --- repair_pairs.m --- %
% Enforce any orbit/slot coupling constraints.
%
% Current implementation:
%   - Just clamp/round to integer bounds.
%   - If you later add orbit-dependent slot rules, implement them here.

x = max(LB, min(UB, round(x)));
end
