function [xval, fval] = abc_discrete(ObjFcn, LB, UB, opts)
% --- abc_discrete.m --- %
% Discrete Artificial Bee Colony (ABC) optimizer for integer decision vectors.
% Enforces x = round(clamp(x,LB,UB)) before objective evaluation.

% --- set defaults (only if missing) --- %
if nargin < 4, opts = struct(); end
if ~isfield(opts,'ColonySize'),      opts.ColonySize = 60; end
if ~isfield(opts,'MaxIters'),        opts.MaxIters = 80; end
if ~isfield(opts,'Limit'),           opts.Limit = 20; end
if ~isfield(opts,'StallIters'),      opts.StallIters = 10; end
if ~isfield(opts,'SlotsPerOrbit'),   opts.SlotsPerOrbit = UB(2); end
if ~isfield(opts,'UseParallelInit'), opts.UseParallelInit = true; end

% --- basic sizes --- %
assert(mod(opts.ColonySize,2)==0, 'ColonySize must be even.');
nFood = opts.ColonySize/2;
nvars = numel(LB);

roundFcn = @(x) max(LB, min(UB, round(x)));

% --- initialize food sources --- %
food_sources = zeros(nFood, nvars);
for i = 1:nFood
    food_sources(i,:) = roundFcn(LB + rand(1,nvars).*(UB-LB));
end

% --- evaluate initial food sources --- %
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

% --- main ABC loop --- %
for itr = 1:opts.MaxIters

    % --- employed bee phase --- %
    for i = 1:nFood
        v = abc_neighbor_discrete(food_sources(i,:), food_sources, LB, UB, opts.SlotsPerOrbit);
        v = roundFcn(v);

        cost_v = ObjFcn(v);

        if cost_v < costs(i)
            food_sources(i,:) = v;
            costs(i) = cost_v;
            trials(i) = 0;
        else
            trials(i) = trials(i) + 1;
        end
    end

    % --- onlooker bee phase --- %
    fit  = 1 ./ (1 + max(costs - min(costs), 0));  % lower cost => higher fitness
    prob = fit / sum(fit);

    onlookers = 0;
    i = 1;
    while onlookers < nFood
        if rand < prob(i)
            onlookers = onlookers + 1;

            v = abc_neighbor_discrete(food_sources(i,:), food_sources, LB, UB, opts.SlotsPerOrbit);
            v = roundFcn(v);

            cost_v = ObjFcn(v);

            if cost_v < costs(i)
                food_sources(i,:) = v;
                costs(i) = cost_v;
                trials(i) = 0;
            else
                trials(i) = trials(i) + 1;
            end
        end

        i = i + 1;
        if i > nFood, i = 1; end
    end

    % --- scout bee phase --- %
    for j = 1:nFood
        if trials(j) >= opts.Limit
            food_sources(j,:) = roundFcn(LB + rand(1,nvars).*(UB-LB));
            costs(j) = ObjFcn(food_sources(j,:));
            trials(j) = 0;
        end
    end

    % --- update best + stall tracking --- %
    [best_cost, idxBest] = min(costs);

    if best_cost < fval
        fval = best_cost;
        xval = food_sources(idxBest,:);
        stallCount = 0;
    else
        stallCount = stallCount + 1;
    end

    fprintf('ABC iter %3d | bestJ = %.6g | stall = %d\n', itr, fval, stallCount);

    if stallCount >= opts.StallIters
        fprintf('ABC stopping early (stall reached).\n');
        break;
    end
end

end


function v = abc_neighbor_discrete(x, Foods, LB, UB, slots_per_orbit)
% --- abc_neighbor_discrete.m --- %
% Structured neighbor generation for discrete orbit/slot design.
%
% x: 1x6 [orb1 slot1 orb2 slot2 orb3 slot3]
% Foods: nFood x 6 population (used to pick partner solution)
%
% Moves (tunable probabilities):
%   1) Local slot tweak (same orbit)              40%
%   2) Pair change: orbit perturb + random slot   30%
%   3) Swap two orbit/slot pairs                  15%
%   4) ABC-style "difference" on a whole pair     10%
%   5) Random restart of one pair                  5%

clampRound = @(z) max(LB, min(UB, round(z)));

v = x;

% --- pick a random (orbit,slot) pair --- %
pair = randi(3);
io = 2*pair - 1;  % orbit index
is = 2*pair;      % slot index

% --- pick a partner solution --- %
nFood = size(Foods,1);
k = randi(nFood);
xk = Foods(k,:);

r = rand;

if r < 0.40
    % --- (1) local slot tweak --- %
    step = randi([-5 5]);
    v(is) = v(is) + step;

elseif r < 0.70
    % --- (2) orbit perturb + reset slot --- %
    dOrb = randi([-50 50]);
    v(io) = v(io) + dOrb;
    v(is) = randi([1 slots_per_orbit]);

elseif r < 0.85
    % --- (3) swap two orbit/slot pairs --- %
    p2 = randi(3);
    while p2 == pair
        p2 = randi(3);
    end
    j1 = 2*pair-1;
    j2 = 2*p2-1;
    tmp = v(j1:j1+1);
    v(j1:j1+1) = v(j2:j2+1);
    v(j2:j2+1) = tmp;

elseif r < 0.95
    % --- (4) ABC-style difference on pair --- %
    phi1 = -1 + 2*rand;
    phi2 = -1 + 2*rand;
    v(io) = v(io) + phi1*(v(io) - xk(io));
    v(is) = v(is) + phi2*(v(is) - xk(is));

else
    % --- (5) restart one pair --- %
    v(io) = LB(io) + rand*(UB(io)-LB(io));
    v(is) = randi([1 slots_per_orbit]);
end

v = clampRound(v);

end
