function [xval, fval ]= aco_discrete(ObjFcn, LB, UB, opts)
% --- aco_discrete_topk.m --- %
% Discrete ACO for x = [Orb1 Slot1 Orb2 Slot2 Orb3 Slot3] with Top-K sampling.
% Pheromone is stored per-variable per-value (tau{j}), but sampling is restricted
% to top-K values for large domains to improve scaling.

% --- set defaults --- %
if nargin < 4, opts = struct(); end
if ~isfield(opts,'nAnts'),      opts.nAnts = 30; end
if ~isfield(opts,'MaxIters'),   opts.MaxIters = 60; end
if ~isfield(opts,'alpha'),      opts.alpha = 1.0; end
if ~isfield(opts,'beta'),       opts.beta = 1.0; end
if ~isfield(opts,'rho'),        opts.rho = 0.2; end
if ~isfield(opts,'Q'),          opts.Q = 1.0; end
if ~isfield(opts,'StallIters'), opts.StallIters = 10; end
if ~isfield(opts,'TopKOrbit'),  opts.TopKOrbit = 150; end
if ~isfield(opts,'TopKSlot'),   opts.TopKSlot = 60; end

nVars = numel(LB);

% --- initialize pheromone (tau) and heuristic (eta) --- %
tau = cell(nVars,1);
eta = cell(nVars,1);

for j = 1:nVars
    Nj = UB(j) - LB(j) + 1;
    tau{j} = ones(Nj,1);     % uniform start

    % simplest heuristic: uniform
    eta{j} = ones(Nj,1);
end

fval = inf;
xval = zeros(1,nVars);
stallCount = 0;

% --- which indices are "orbit variables"? --- %
orbitDims = [1 3 5];
slotDims  = [2 4 6];

% ==========================
% main ACO loop
% ==========================
for itr = 1:opts.MaxIters

    antX = zeros(opts.nAnts, nVars);
    antJ = zeros(opts.nAnts, 1);

    % --- construct solutions --- %
    for a = 1:opts.nAnts
        x = zeros(1,nVars);

        for j = 1:nVars
            Nj = UB(j) - LB(j) + 1;

            % --- candidate list (Top-K) --- %
            if ismember(j, orbitDims)
                K = min(opts.TopKOrbit, Nj);
            else
                K = min(opts.TopKSlot, Nj);
            end

            % Take top-K indices by pheromone*heuristic
            score = (tau{j}.^opts.alpha) .* (eta{j}.^opts.beta);

            if K < Nj
                [~, idxTop] = maxk(score, K);   % idxTop are 1..Nj
            else
                idxTop = (1:Nj).';
            end

            p = score(idxTop);
            p = p / sum(p);

            chosen = idxTop(roulette_select(p)); % chosen in 1..Nj
            x(j) = LB(j) + (chosen - 1);
        end

        antX(a,:) = x;
        antJ(a) = ObjFcn(x);
    end

    % --- update iteration best --- %
    [iterBestJ, idx] = min(antJ);
    iterBestX = antX(idx,:);

    if iterBestJ < fval
        fval = iterBestJ;
        xval = iterBestX;
        stallCount = 0;
    else
        stallCount = stallCount + 1;
    end

    % --- evaporate pheromone --- %
    for j = 1:nVars
        tau{j} = (1 - opts.rho) * tau{j};
    end

    % --- deposit pheromone (global-best elitist update) --- %
    deposit = opts.Q / (fval + eps);
    for j = 1:nVars
        idxVal = xval(j) - LB(j) + 1;
        tau{j}(idxVal) = tau{j}(idxVal) + deposit;
    end

    fprintf('ACO iter %3d | bestJ = %.6g | stall = %d\n', itr, fval, stallCount);

    if stallCount >= opts.StallIters
        fprintf('ACO stopping early (stall reached).\n');
        break;
    end
end
end


function idx = roulette_select(p)
% --- roulette_select.m --- %
% p must sum to 1

cdf = cumsum(p);
r = rand;
idx = find(r <= cdf, 1, 'first');
end
