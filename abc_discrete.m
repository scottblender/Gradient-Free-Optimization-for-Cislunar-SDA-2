function [xval, fval] = abc_discrete(ObjFcn, LB, UB, opts)
% --- abc_discrete.m (VARIABLE # ORBIT/SLOT PAIRS) --- %
% Discrete Artificial Bee Colony (ABC) optimizer for integer decision vectors:
%   x = [orb1 slot1 orb2 slot2 ... orbP slotP], P = nvars/2

% ---------------- defaults ----------------
if nargin < 4 || isempty(opts), opts = struct(); end
if ~isfield(opts,'ColonySize'),      opts.ColonySize = 60; end
if ~isfield(opts,'MaxIters'),        opts.MaxIters = 80; end
if ~isfield(opts,'Limit'),           opts.Limit = 20; end
if ~isfield(opts,'StallIters'),      opts.StallIters = 1000; end
if ~isfield(opts,'UseParallel'),     opts.UseParallel = true; end
if ~isfield(opts,'UseParallelInit'), opts.UseParallelInit = opts.UseParallel; end

% ---------------- sizes ----------------
assert(mod(opts.ColonySize,2)==0, 'ColonySize must be even.');
nFood = opts.ColonySize/2;

nvars = numel(LB);
assert(mod(nvars,2)==0, 'Decision vector must be even: [orbit,slot] pairs.');
nPairs = nvars/2;

roundFcn = @(x) max(LB, min(UB, round(x)));

% ---------------- initialize food sources ----------------
food_sources = zeros(nFood, nvars);
for i = 1:nFood
    food_sources(i,:) = roundFcn(LB + rand(1,nvars).*(UB-LB));
end

% ---------------- evaluate initial food sources ----------------
costs  = zeros(nFood,1);
trials = zeros(nFood,1);

if opts.UseParallelInit
    parfor i = 1:nFood
        costs(i) = ObjFcn(food_sources(i,:));
    end
else
    for i = 1:nFood
        costs(i) = ObjFcn(food_sources(i,:));
    end
end

[fval, idxBest] = min(costs);
xval = food_sources(idxBest,:);
stallCount = 0;

% ================= main ABC loop =================
for itr = 1:opts.MaxIters

    % ---------------------------------------------------------
    % (1) Employed bee phase (BATCHED)
    % ---------------------------------------------------------
    V_emp = zeros(nFood, nvars);
    for i = 1:nFood
        v = abc_neighbor_discrete(food_sources(i,:), food_sources, LB, UB, nPairs);
        V_emp(i,:) = roundFcn(v);
    end

    costV_emp = zeros(nFood,1);
    if opts.UseParallel
        parfor i = 1:nFood
            costV_emp(i) = ObjFcn(V_emp(i,:));
        end
    else
        for i = 1:nFood
            costV_emp(i) = ObjFcn(V_emp(i,:));
        end
    end

    for i = 1:nFood
        if costV_emp(i) < costs(i)
            food_sources(i,:) = V_emp(i,:);
            costs(i) = costV_emp(i);
            trials(i) = 0;
        else
            trials(i) = trials(i) + 1;
        end
    end

    % ---------------------------------------------------------
    % (2) Onlooker bee phase (BATCHED)
    % ---------------------------------------------------------
    fit  = 1 ./ (1 + max(costs - min(costs), 0));  % lower cost => higher fitness
    prob = fit / sum(fit);

    % sample nFood indices with replacement by prob
    idx_onl = randsample(nFood, nFood, true, prob);

    V_onl = zeros(nFood, nvars);
    for j = 1:nFood
        i = idx_onl(j);
        v = abc_neighbor_discrete(food_sources(i,:), food_sources, LB, UB, nPairs);
        V_onl(j,:) = roundFcn(v);
    end

    costV_onl = zeros(nFood,1);
    if opts.UseParallel
        parfor j = 1:nFood
            costV_onl(j) = ObjFcn(V_onl(j,:));
        end
    else
        for j = 1:nFood
            costV_onl(j) = ObjFcn(V_onl(j,:));
        end
    end

    for j = 1:nFood
        i = idx_onl(j);
        if costV_onl(j) < costs(i)
            food_sources(i,:) = V_onl(j,:);
            costs(i) = costV_onl(j);
            trials(i) = 0;
        else
            trials(i) = trials(i) + 1;
        end
    end

    % ---------------------------------------------------------
    % (3) Scout bee phase (BATCHED for any trials >= Limit)
    % ---------------------------------------------------------
    scout_idx = find(trials >= opts.Limit);
    nScout = numel(scout_idx);

    if nScout > 0
        newFoods = zeros(nScout, nvars);
        for s = 1:nScout
            newFoods(s,:) = roundFcn(LB + rand(1,nvars).*(UB-LB));
        end

        newCosts = zeros(nScout,1);
        if opts.UseParallel
            parfor s = 1:nScout
                newCosts(s) = ObjFcn(newFoods(s,:));
            end
        else
            for s = 1:nScout
                newCosts(s) = ObjFcn(newFoods(s,:));
            end
        end

        for s = 1:nScout
            j = scout_idx(s);
            food_sources(j,:) = newFoods(s,:);
            costs(j) = newCosts(s);
            trials(j) = 0;
        end
    end

    % ---------------------------------------------------------
    % update best + stall tracking
    % ---------------------------------------------------------
    [best_cost, idxBest] = min(costs);

    if best_cost < fval
        fval = best_cost;
        xval = food_sources(idxBest,:);
        stallCount = 0;
    else
        stallCount = stallCount + 1;
    end

    fprintf('ABC iter %3d | bestJ = %.6g | stall = %d | scouts = %d\n', itr, fval, stallCount, nScout);

    if stallCount >= opts.StallIters
        fprintf('ABC stopping early (stall reached).\n');
        break;
    end
end
end


function v = abc_neighbor_discrete(x, Foods, LB, UB, nPairs)
% Structured neighbor generation for discrete orbit/slot design with variable # pairs.
%
% x:      1 x (2*nPairs) = [orb1 slot1 ... orbP slotP]
% Foods:  nFood x (2*nPairs) population (used to pick partner solution)
%
% Moves (tunable probabilities):
%   1) Local slot tweak (same orbit)              40%
%   2) Orbit perturb + reset slot                 30%
%   3) Swap two orbit/slot pairs                  15%
%   4) ABC-style "difference" on pair             10%
%   5) Random restart of one pair                  5%

clampRound = @(z) max(LB, min(UB, round(z)));

v = x;

% pick random (orbit,slot) pair among nPairs
pair = randi(nPairs);
io = 2*pair - 1;  % orbit index in vector
is = 2*pair;      % slot index in vector

% slot bounds for this pair (allows variable slots per pair if you ever want it)
slotLB = LB(is);
slotUB = UB(is);

% pick partner solution
nFood = size(Foods,1);
k = randi(nFood);
xk = Foods(k,:);

r = rand;

if r < 0.40
    % (1) local slot tweak
    step = randi([-5 5]);
    v(is) = v(is) + step;

elseif r < 0.70
    % (2) orbit perturb + reset slot
    dOrb = randi([-50 50]);
    v(io) = v(io) + dOrb;
    v(is) = randi([slotLB slotUB]);

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
    % (4) ABC-style difference on pair
    phi1 = -1 + 2*rand;
    phi2 = -1 + 2*rand;
    v(io) = v(io) + phi1*(v(io) - xk(io));
    v(is) = v(is) + phi2*(v(is) - xk(is));

else
    % (5) restart one pair
    v(io) = LB(io) + rand*(UB(io)-LB(io));
    v(is) = randi([slotLB slotUB]);
end

v = clampRound(v);
end
