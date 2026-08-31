function [xval, fval, nEvals] = abc_discrete(ObjFcn, LB, UB, opts)
% --- abc_discrete.m (VARIABLE # ORBIT/SLOT PAIRS) --- %
% Discrete Artificial Bee Colony (ABC) optimizer for integer decision vectors:
%   x = [orb1 slot1 orb2 slot2 ... orbP slotP], P = nvars/2
%
% The objective can be evaluated in parallel, while nEvals is maintained on
% the MATLAB client. opts.MaxEvals is the universal stopping criterion used
% for fair optimizer comparisons.

% ---------------- defaults ----------------
if nargin < 4 || isempty(opts), opts = struct(); end
if ~isfield(opts,'ColonySize'),      opts.ColonySize = 60; end
if ~isfield(opts,'MaxIters'),        opts.MaxIters = inf; end
if ~isfield(opts,'MaxEvals'),        opts.MaxEvals = inf; end
if ~isfield(opts,'Limit'),           opts.Limit = 20; end
if ~isfield(opts,'StallIters'),      opts.StallIters = inf; end
if ~isfield(opts,'UseParallel'),     opts.UseParallel = true; end
if ~isfield(opts,'UseParallelInit'), opts.UseParallelInit = opts.UseParallel; end
if ~isfield(opts,'Logger') || isempty(opts.Logger)
    opts.Logger = @(varargin) fprintf(varargin{:});
end

validateattributes(opts.MaxEvals, {'numeric'}, ...
    {'scalar','real','positive'});

% ---------------- sizes ----------------
assert(mod(opts.ColonySize,2)==0, 'ColonySize must be even.');
nFood = opts.ColonySize/2;

nvars = numel(LB);
assert(mod(nvars,2)==0, 'Decision vector must be even: [orbit,slot] pairs.');
nPairs = nvars/2;

if opts.MaxEvals < nFood
    error('ABC:InsufficientBudget', ...
        'MaxEvals (%d) must be at least the number of food sources (%d).', ...
        opts.MaxEvals, nFood);
end

roundFcn = @(x) max(LB, min(UB, round(x)));

% ---------------- initialize food sources ----------------
food_sources = zeros(nFood, nvars);
for i = 1:nFood
    food_sources(i,:) = roundFcn(LB + rand(1,nvars).*(UB-LB));
end

costs  = inf(nFood,1);
trials = zeros(nFood,1);
nEvals = 0;

[costInit, nDone] = evaluate_batch_limited( ...
    ObjFcn, food_sources, opts.UseParallelInit, opts.MaxEvals - nEvals);
costs(1:nDone) = costInit;
nEvals = nEvals + nDone;

[fval, idxBest] = min(costs);
xval = food_sources(idxBest,:);
stallCount = 0;

opts.Logger('ABC init     | FE = %5d/%5d | bestJ = %.6g\n', ...
    nEvals, opts.MaxEvals, fval);

% ================= main ABC loop =================
itr = 0;
while itr < opts.MaxIters && nEvals < opts.MaxEvals
    itr = itr + 1;
    fvalAtStart = fval;

    % ---------------------------------------------------------
    % (1) Employed bee phase
    % ---------------------------------------------------------
    V_emp = zeros(nFood, nvars);
    for i = 1:nFood
        v = abc_neighbor_discrete(food_sources(i,:), food_sources, LB, UB, nPairs);
        V_emp(i,:) = roundFcn(v);
    end

    [costV_emp, nDone] = evaluate_batch_limited( ...
        ObjFcn, V_emp, opts.UseParallel, opts.MaxEvals - nEvals);
    nEvals = nEvals + nDone;

    for i = 1:nDone
        if costV_emp(i) < costs(i)
            food_sources(i,:) = V_emp(i,:);
            costs(i) = costV_emp(i);
            trials(i) = 0;
        else
            trials(i) = trials(i) + 1;
        end
    end

    [fval, xval] = update_best(costs, food_sources, fval, xval);

    if nEvals >= opts.MaxEvals
        opts.Logger('ABC iter %3d | FE = %5d/%5d | bestJ = %.6g | phase = employed\n', ...
            itr, nEvals, opts.MaxEvals, fval);
        break;
    end

    % ---------------------------------------------------------
    % (2) Onlooker bee phase
    % ---------------------------------------------------------
    fit  = 1 ./ (1 + max(costs - min(costs), 0));
    prob = fit / sum(fit);
    idx_onl = randsample(nFood, nFood, true, prob);

    V_onl = zeros(nFood, nvars);
    for j = 1:nFood
        i = idx_onl(j);
        v = abc_neighbor_discrete(food_sources(i,:), food_sources, LB, UB, nPairs);
        V_onl(j,:) = roundFcn(v);
    end

    [costV_onl, nDone] = evaluate_batch_limited( ...
        ObjFcn, V_onl, opts.UseParallel, opts.MaxEvals - nEvals);
    nEvals = nEvals + nDone;

    for j = 1:nDone
        i = idx_onl(j);
        if costV_onl(j) < costs(i)
            food_sources(i,:) = V_onl(j,:);
            costs(i) = costV_onl(j);
            trials(i) = 0;
        else
            trials(i) = trials(i) + 1;
        end
    end

    [fval, xval] = update_best(costs, food_sources, fval, xval);

    if nEvals >= opts.MaxEvals
        opts.Logger('ABC iter %3d | FE = %5d/%5d | bestJ = %.6g | phase = onlooker\n', ...
            itr, nEvals, opts.MaxEvals, fval);
        break;
    end

    % ---------------------------------------------------------
    % (3) Scout bee phase
    % ---------------------------------------------------------
    scout_idx = find(trials >= opts.Limit);
    nScout = numel(scout_idx);

    if nScout > 0
        newFoods = zeros(nScout, nvars);
        for s = 1:nScout
            newFoods(s,:) = roundFcn(LB + rand(1,nvars).*(UB-LB));
        end

        [newCosts, nDone] = evaluate_batch_limited( ...
            ObjFcn, newFoods, opts.UseParallel, opts.MaxEvals - nEvals);
        nEvals = nEvals + nDone;

        for s = 1:nDone
            j = scout_idx(s);
            food_sources(j,:) = newFoods(s,:);
            costs(j) = newCosts(s);
            trials(j) = 0;
        end

        [fval, xval] = update_best(costs, food_sources, fval, xval);
    end

    if fval < fvalAtStart
        stallCount = 0;
    else
        stallCount = stallCount + 1;
    end

    opts.Logger('ABC iter %3d | FE = %5d/%5d | bestJ = %.6g | stall = %d | scouts = %d\n', ...
        itr, nEvals, opts.MaxEvals, fval, stallCount, nScout);

    if nEvals >= opts.MaxEvals
        break;
    end

    if stallCount >= opts.StallIters
        opts.Logger('ABC stopping early (stall reached).\n');
        break;
    end
end
end


function [J, nDone] = evaluate_batch_limited(ObjFcn, X, useParallel, remaining)
% Evaluate no more than the remaining FE budget. The counter is deliberately
% updated by the caller on the client, never inside parfor workers.

nRequested = size(X,1);
nDone = min(nRequested, max(0, floor(remaining)));
J = zeros(nDone,1);

if nDone == 0
    return;
end

if useParallel
    parfor i = 1:nDone
        J(i) = ObjFcn(X(i,:));
    end
else
    for i = 1:nDone
        J(i) = ObjFcn(X(i,:));
    end
end
end


function [bestJ, bestX] = update_best(costs, Foods, bestJ, bestX)
[currentBest, idxBest] = min(costs);
if currentBest < bestJ
    bestJ = currentBest;
    bestX = Foods(idxBest,:);
end
end


function v = abc_neighbor_discrete(x, Foods, LB, UB, nPairs)
% Structured neighbor generation for discrete orbit/slot design with variable # pairs.
%
% x:      1 x (2*nPairs) = [orb1 slot1 ... orbP slotP]
% Foods:  nFood x (2*nPairs) population (used to pick partner solution)
%
% Moves:
%   1) Local slot tweak (same orbit)              40%
%   2) Orbit perturb + reset slot                 30%
%   3) Swap two orbit/slot pairs                  15%
%   4) ABC-style difference on pair               10%
%   5) Random restart of one pair                  5%

clampRound = @(z) max(LB, min(UB, round(z)));
v = x;

pair = randi(nPairs);
io = 2*pair - 1;
is = 2*pair;
slotLB = LB(is);
slotUB = UB(is);

nFood = size(Foods,1);
k = randi(nFood);
xk = Foods(k,:);

r = rand;

if r < 0.40
    step = randi([-5 5]);
    v(is) = v(is) + step;

elseif r < 0.70
    dOrb = randi([-50 50]);
    v(io) = v(io) + dOrb;
    v(is) = randi([slotLB slotUB]);

elseif r < 0.85
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
    phi1 = -1 + 2*rand;
    phi2 = -1 + 2*rand;
    v(io) = v(io) + phi1*(v(io) - xk(io));
    v(is) = v(is) + phi2*(v(is) - xk(is));

else
    v(io) = LB(io) + rand*(UB(io)-LB(io));
    v(is) = randi([slotLB slotUB]);
end

v = clampRound(v);
end
