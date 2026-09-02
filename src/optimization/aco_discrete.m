function [xval, fval, nEvals, history] = aco_discrete(ObjFcn, LB, UB, opts)
% --- aco_discrete.m (STRUCTURED, VARIABLE N) --- %
% Ant Colony Optimization for discrete integer decision vectors:
%   x = [orb1 slot1 orb2 slot2 ... orbP slotP]
%
% Objective evaluations may run in parallel. nEvals is maintained on the
% MATLAB client, and opts.MaxEvals is the universal stopping criterion.

% ---------------- defaults ----------------
if nargin < 4 || isempty(opts), opts = struct(); end
if ~isfield(opts,'nAnts'),        opts.nAnts = 40; end
if ~isfield(opts,'MaxEvals'),     opts.MaxEvals = inf; end
if ~isfield(opts,'alpha'),        opts.alpha = 1.0; end
if ~isfield(opts,'beta'),         opts.beta = 2.0; end
if ~isfield(opts,'rho'),          opts.rho = 0.2; end
if ~isfield(opts,'Q'),            opts.Q = 1.0; end
if ~isfield(opts,'StallIters'),   opts.StallIters = inf; end
if ~isfield(opts,'UseParallel'),  opts.UseParallel = true; end
if ~isfield(opts,'TauMin'),       opts.TauMin = 1e-12; end
if ~isfield(opts,'UseIterBestDeposit'), opts.UseIterBestDeposit = true; end
if ~isfield(opts,'IterBestWeight'),     opts.IterBestWeight = 1.0; end
if ~isfield(opts,'UseIterBestDepositSlots'), opts.UseIterBestDepositSlots = true; end
if ~isfield(opts,'Logger') || isempty(opts.Logger)
    opts.Logger = @(varargin) fprintf(varargin{:});
end

validateattributes(opts.MaxEvals, {'numeric'}, ...
    {'scalar','real','positive'});

% ---------------- sizes / structure ----------------
nVars = numel(LB);
assert(mod(nVars,2)==0, 'ACO expects an even number of variables: [orbit,slot] pairs.');
nPairs = nVars/2;

clampRound = @(x) max(LB, min(UB, round(x)));

nOrbits = zeros(nPairs,1);
nSlots  = zeros(nPairs,1);
for p = 1:nPairs
    io = 2*p - 1;
    is = 2*p;
    nOrbits(p) = UB(io) - LB(io) + 1;
    nSlots(p)  = UB(is) - LB(is) + 1;
end

% ---------------- initialize pheromones ----------------
tauOrb = cell(nPairs,1);
etaOrb = cell(nPairs,1);
tauSlot = cell(nPairs,1);
etaSlot = cell(nPairs,1);

for p = 1:nPairs
    tauOrb{p} = ones(nOrbits(p),1);
    etaOrb{p} = ones(nOrbits(p),1);
    tauSlot{p} = cell(nOrbits(p),1);
    etaSlot{p} = cell(nOrbits(p),1);
end

% ---------------- initialize best / FE history ----------------
fval = inf;
xval = clampRound(LB + rand(1,nVars).*(UB-LB));
stallCount = 0;
nEvals = 0;

histFE = zeros(0,1);
histBest = zeros(0,1);

% ==========================
% main ACO loop
% ==========================
itr = 0;
while nEvals < opts.MaxEvals
    itr = itr + 1;

    nThisBatch = min(opts.nAnts, opts.MaxEvals - nEvals);
    antX = zeros(nThisBatch, nVars);
    antJ = zeros(nThisBatch, 1);

    % -------- build all ant solutions on the client --------
    for a = 1:nThisBatch
        x = zeros(1,nVars);

        for p = 1:nPairs
            io = 2*p - 1;
            is = 2*p;

            p_orb = prob_from_tau_eta(tauOrb{p}, etaOrb{p}, opts.alpha, opts.beta);
            oIdx  = roulette_select(p_orb);
            x(io) = LB(io) + (oIdx-1);

            if isempty(tauSlot{p}{oIdx})
                tauSlot{p}{oIdx} = ones(nSlots(p),1);
                etaSlot{p}{oIdx} = ones(nSlots(p),1);
            end

            p_slot = prob_from_tau_eta(tauSlot{p}{oIdx}, etaSlot{p}{oIdx}, opts.alpha, opts.beta);
            sIdx   = roulette_select(p_slot);
            x(is)  = LB(is) + (sIdx-1);
        end

        antX(a,:) = clampRound(x);
    end

    % -------- evaluate batch in parallel --------
    if opts.UseParallel
        parfor a = 1:nThisBatch
            antJ(a) = ObjFcn(antX(a,:));
        end
    else
        for a = 1:nThisBatch
            antJ(a) = ObjFcn(antX(a,:));
        end
    end

    nEvals = nEvals + nThisBatch;

    % -------- update bests --------
    [iterBestJ, idx] = min(antJ);
    iterBestX = antX(idx,:);

    if iterBestJ < fval
        fval = iterBestJ;
        xval = iterBestX;
        stallCount = 0;
    else
        stallCount = stallCount + 1;
    end

    histFE(end+1,1) = nEvals;
    histBest(end+1,1) = fval;

    % -------- evaporation --------
    for p = 1:nPairs
        tauOrb{p} = max((1 - opts.rho) * tauOrb{p}, opts.TauMin);

        for oIdx = 1:nOrbits(p)
            if ~isempty(tauSlot{p}{oIdx})
                tauSlot{p}{oIdx} = max((1 - opts.rho) * tauSlot{p}{oIdx}, opts.TauMin);
            end
        end
    end

    % -------- deposit --------
    dep_best = opts.Q / (fval + eps);
    [tauOrb, tauSlot] = deposit_structured(tauOrb, tauSlot, xval, LB, UB, dep_best);

    if opts.UseIterBestDeposit
        dep_iter = opts.IterBestWeight * (opts.Q / (iterBestJ + eps));
        [tauOrb, tauSlot] = deposit_structured(tauOrb, tauSlot, iterBestX, LB, UB, dep_iter);
    end

    opts.Logger('ACO iter %3d | FE = %5d/%5d | bestJ = %.6g | iterBestJ = %.6g | stall = %d\n', ...
        itr, nEvals, opts.MaxEvals, fval, iterBestJ, stallCount);

    if stallCount >= opts.StallIters
        opts.Logger('ACO stopping early (stall reached).\n');
        break;
    end
end

history = table(histFE, histBest, ...
    'VariableNames', {'fe','bestJ'});
end


% ==========================
% helper functions
% ==========================
function p = prob_from_tau_eta(tau_vec, eta_vec, alpha, beta)
    w = (tau_vec.^alpha) .* (eta_vec.^beta);
    s = sum(w);
    if s <= 0 || ~isfinite(s)
        p = ones(size(w)) / numel(w);
    else
        p = w / s;
    end
end

function [tauOrb, tauSlot] = deposit_structured(tauOrb, tauSlot, x, LB, UB, deposit)
    nVars = numel(LB);
    nPairs = nVars/2;

    for p = 1:nPairs
        io = 2*p - 1;
        is = 2*p;

        oIdx = x(io) - LB(io) + 1;
        sIdx = x(is) - LB(is) + 1;

        oIdx = max(1, min(oIdx, UB(io)-LB(io)+1));
        sIdx = max(1, min(sIdx, UB(is)-LB(is)+1));

        tauOrb{p}(oIdx) = tauOrb{p}(oIdx) + deposit;

        if isempty(tauSlot{p}{oIdx})
            nSlots_p = UB(is) - LB(is) + 1;
            tauSlot{p}{oIdx} = ones(nSlots_p, 1);
        end
        tauSlot{p}{oIdx}(sIdx) = tauSlot{p}{oIdx}(sIdx) + deposit;
    end
end

function idx = roulette_select(p)
    cdf = cumsum(p);
    r = rand;
    idx = find(r <= cdf, 1, 'first');
    if isempty(idx)
        idx = numel(p);
    end
end
