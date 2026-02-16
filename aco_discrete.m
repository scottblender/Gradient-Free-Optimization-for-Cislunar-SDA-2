function [xval, fval] = aco_discrete(ObjFcn, LB, UB, opts)
% --- aco_discrete.m --- %
% Ant Colony Optimization for discrete orbit/slot decision vectors.

% --- set defaults --- %
if nargin < 4, opts = struct(); end
if ~isfield(opts,'nAnts'),      opts.nAnts = 40; end
if ~isfield(opts,'MaxIters'),   opts.MaxIters = 60; end
if ~isfield(opts,'alpha'),      opts.alpha = 1.0; end
if ~isfield(opts,'beta'),       opts.beta = 2.0; end
if ~isfield(opts,'rho'),        opts.rho = 0.2; end
if ~isfield(opts,'Q'),          opts.Q = 1.0; end
if ~isfield(opts,'StallIters'), opts.StallIters = 10; end

nVars = numel(LB);

% --- initialize pheromone trails --- %
% tau(j,k) = pheromone for choosing value k at variable j
tau = cell(nVars,1);
for j = 1:nVars
    Nj = UB(j) - LB(j) + 1;
    tau{j} = ones(Nj,1);   % uniform pheromone init
end

% --- optional heuristic desirability --- %
eta = cell(nVars,1);
for j = 1:nVars
    Nj = UB(j) - LB(j) + 1;
    eta{j} = ones(Nj,1);   % simplest: no heuristic bias
end

fval = inf;
xval = [];

stallCount = 0;

% ==========================
% main ACO loop
% ==========================
for itr = 1:opts.MaxIters

    antX = zeros(opts.nAnts, nVars);
    antJ = zeros(opts.nAnts, 1);

    % --- each ant builds a full solution --- %
    for a = 1:opts.nAnts

        x = zeros(1,nVars);

        for j = 1:nVars
            tau_j = tau{j}.^opts.alpha;
            eta_j = eta{j}.^opts.beta;

            p = (tau_j .* eta_j);
            p = p / sum(p);

            idx = roulette_select(p);    % pick an index based on p
            x(j) = LB(j) + (idx-1);
        end

        antX(a,:) = x;
        antJ(a) = ObjFcn(x);
    end

    % --- update best --- %
    [iterBestJ, idx] = min(antJ);
    iterBestX = antX(idx,:);

    if iterBestJ < fval
        fval = iterBestJ;
        xval = iterBestX;
        stallCount = 0;
    else
        stallCount = stallCount + 1;
    end

    % --- pheromone evaporation --- %
    for j = 1:nVars
        tau{j} = (1 - opts.rho) * tau{j};
    end

    % --- pheromone deposit (global-best reinforcement) --- %
    % add pheromone proportional to solution quality
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
