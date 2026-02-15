% ======================= aco_discrete.m =======================
function [xval, fval] = aco_discrete(ObjFcn, LB, UB, opts, stopper)
% --- aco_discrete.m --- %
% Ant Colony Optimization (ACO) for DISCRETE integer decision vectors.
%
% Decision vector:
%   x is an integer row vector with bounds LB <= x <= UB (element-wise).
%
% Parallel-safe usage:
%   - If opts.UseParallel = true, objective evaluations run inside PARFOR.
%   - ObjFcn MUST be side-effect free (no shared-state mutation, no stopper calls).
%   - Stopping is handled ONLY on the client via the Stopper object.
%
% Outputs:
%   xval : best solution found (row vector)
%   fval : best objective value found

% --- set defaults --- %
if nargin < 4 || isempty(opts), opts = struct(); end
if ~isfield(opts,'nAnts'),       opts.nAnts = 40; end
if ~isfield(opts,'MaxIters'),    opts.MaxIters = 60; end
if ~isfield(opts,'alpha'),       opts.alpha = 1.0; end
if ~isfield(opts,'beta'),        opts.beta  = 2.0; end
if ~isfield(opts,'rho'),         opts.rho   = 0.2; end
if ~isfield(opts,'Q'),           opts.Q     = 1.0; end
if ~isfield(opts,'UseParallel'), opts.UseParallel = true; end

nVars = numel(LB);

% --- initialize pheromone trails and heuristic --- %
% tau{j}(k) = pheromone weight for choosing discrete value k at variable j
% eta{j}(k) = heuristic desirability (set to uniform if no heuristic bias)
tau = cell(nVars,1);
eta = cell(nVars,1);
for j = 1:nVars
    Nj = UB(j) - LB(j) + 1;
    tau{j} = ones(Nj,1);   % uniform pheromone initialization
    eta{j} = ones(Nj,1);   % simplest: no heuristic bias
end

% --- global best --- %
fval = inf;
xval = [];

% --- authoritative eval counter (for custom algorithm) --- %
evalTotal = 0;

% ==========================
% main ACO loop
% ==========================
for itr = 1:opts.MaxIters

    % --- optional early stop (set by stopper from previous iteration) --- %
    if nargin >= 5 && ~isempty(stopper) && stopper.stopFlag
        fprintf('ACO stopping early: %s\n', stopper.stopReason);
        break;
    end

    antX = zeros(opts.nAnts, nVars);
    antJ = zeros(opts.nAnts, 1);

    % --- each ant builds a full solution --- %
    for a = 1:opts.nAnts
        x = zeros(1, nVars);

        for j = 1:nVars
            tau_j = tau{j}.^opts.alpha;
            eta_j = eta{j}.^opts.beta;

            p = (tau_j .* eta_j);
            p = p / sum(p);

            idx = roulette_select(p);    % pick discrete index based on p
            x(j) = LB(j) + (idx - 1);
        end

        antX(a,:) = x;
    end

    % --- evaluate ant solutions (optional parallel) --- %
    if opts.UseParallel
        parfor a = 1:opts.nAnts
            antJ(a) = ObjFcn(antX(a,:));
        end
    else
        for a = 1:opts.nAnts
            antJ(a) = ObjFcn(antX(a,:));
        end
    end
    evalTotal = evalTotal + opts.nAnts;

    % --- update global best --- %
    [iterBestJ, idxBest] = min(antJ);
    iterBestX = antX(idxBest,:);

    if isempty(xval) || iterBestJ < fval
        fval = iterBestJ;
        xval = iterBestX;
    end

    % --- update stopping criteria (client-side only) --- %
    if nargin >= 5 && ~isempty(stopper)
        stopper.update(fval, evalTotal, xval);
        if stopper.stopFlag
            fprintf('ACO stopping early: %s\n', stopper.stopReason);
            break;
        end
    end

    % --- pheromone evaporation --- %
    for j = 1:nVars
        tau{j} = (1 - opts.rho) * tau{j};
    end

    % --- pheromone deposit (global-best reinforcement) --- %
    % deposit scaled to remain positive even when fval is negative
    deposit = opts.Q / (abs(fval) + eps);
    for j = 1:nVars
        idxVal = xval(j) - LB(j) + 1;
        tau{j}(idxVal) = tau{j}(idxVal) + deposit;
    end

    fprintf('ACO iter %3d | bestJ = %.6g | evals = %d\n', itr, fval, evalTotal);
end
end

function idx = roulette_select(p)
% --- roulette_select.m --- %
% Roulette wheel selection for a discrete probability vector p (must sum to 1).

cdf = cumsum(p(:));
r = rand;
idx = find(r <= cdf, 1, 'first');
if isempty(idx)
    idx = numel(p);
end
end
