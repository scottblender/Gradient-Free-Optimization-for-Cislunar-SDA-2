function [xval, fval] = aco_discrete(ObjFcn, LB, UB, opts)
% --- aco_discrete.m (STRUCTURED, VARIABLE N) --- %
% Ant Colony Optimization for discrete integer decision vectors with paired variables:
%   x = [orb1 slot1 orb2 slot2 ... orbP slotP], where P = nVars/2

% ---------------- defaults ----------------
if nargin < 4 || isempty(opts), opts = struct(); end
if ~isfield(opts,'nAnts'),        opts.nAnts = 40; end
if ~isfield(opts,'MaxIters'),     opts.MaxIters = 60; end
if ~isfield(opts,'alpha'),        opts.alpha = 1.0; end
if ~isfield(opts,'beta'),         opts.beta = 2.0; end
if ~isfield(opts,'rho'),          opts.rho = 0.2; end
if ~isfield(opts,'Q'),            opts.Q = 1.0; end
if ~isfield(opts,'StallIters'),   opts.StallIters = 1000; end
if ~isfield(opts,'UseParallel'),  opts.UseParallel = true; end
if ~isfield(opts,'TauMin'),       opts.TauMin = 1e-12; end
if ~isfield(opts,'UseIterBestDeposit'), opts.UseIterBestDeposit = true; end
if ~isfield(opts,'IterBestWeight'),     opts.IterBestWeight = 1.0; end
if ~isfield(opts,'UseIterBestDepositSlots'), opts.UseIterBestDepositSlots = true; end

% ---------------- sizes / structure ----------------
nVars = numel(LB);
assert(mod(nVars,2)==0, 'ACO expects an even number of variables: [orbit,slot] pairs.');
nPairs = nVars/2;

clampRound = @(x) max(LB, min(UB, round(x)));

% For each pair p:
%   orbit var index io = 2p-1
%   slot  var index is = 2p
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

% tauSlot{p} is a cell array of length nOrbits(p); each entry holds a [nSlots(p)x1] vector
tauSlot = cell(nPairs,1);
etaSlot = cell(nPairs,1);

for p = 1:nPairs
    tauOrb{p} = ones(nOrbits(p),1);
    etaOrb{p} = ones(nOrbits(p),1);

    tauSlot{p} = cell(nOrbits(p),1); % lazy allocate each orbit's slot pheromones
    etaSlot{p} = cell(nOrbits(p),1);
end

% ---------------- initialize best ----------------
fval = inf;
xval = clampRound(LB + rand(1,nVars).*(UB-LB));
stallCount = 0;

% ==========================
% main ACO loop
% ==========================
for itr = 1:opts.MaxIters

    antX = zeros(opts.nAnts, nVars);
    antJ = zeros(opts.nAnts, 1);

    % -------- build all ant solutions (client) --------
    for a = 1:opts.nAnts
        x = zeros(1,nVars);

        for p = 1:nPairs
            io = 2*p - 1;
            is = 2*p;

            % --- pick orbit index (1..nOrbits(p)) ---
            p_orb = prob_from_tau_eta(tauOrb{p}, etaOrb{p}, opts.alpha, opts.beta);
            oIdx  = roulette_select(p_orb);
            x(io) = LB(io) + (oIdx-1);  % orbit value

            % --- ensure slot pheromone exists for this orbit index ---
            if isempty(tauSlot{p}{oIdx})
                tauSlot{p}{oIdx} = ones(nSlots(p),1);
                etaSlot{p}{oIdx} = ones(nSlots(p),1);
            end

            % --- pick slot index (1..nSlots(p)), conditioned on orbit index ---
            p_slot = prob_from_tau_eta(tauSlot{p}{oIdx}, etaSlot{p}{oIdx}, opts.alpha, opts.beta);
            sIdx   = roulette_select(p_slot);
            x(is)  = LB(is) + (sIdx-1); % slot value
        end

        antX(a,:) = clampRound(x);
    end

    % -------- evaluate all ants (batch, parallel optional) --------
    if opts.UseParallel
        parfor a = 1:opts.nAnts
            antJ(a) = ObjFcn(antX(a,:));
        end
    else
        for a = 1:opts.nAnts
            antJ(a) = ObjFcn(antX(a,:));
        end
    end

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

    % -------- evaporation (orbit + only allocated slot vectors) --------
    for p = 1:nPairs
        tauOrb{p} = max((1 - opts.rho) * tauOrb{p}, opts.TauMin);

        for oIdx = 1:nOrbits(p)
            if ~isempty(tauSlot{p}{oIdx})
                tauSlot{p}{oIdx} = max((1 - opts.rho) * tauSlot{p}{oIdx}, opts.TauMin);
            end
        end
    end

    % -------- deposit (global best) --------
    dep_best = opts.Q / (fval + eps);
    [tauOrb, tauSlot] = deposit_structured(tauOrb, tauSlot, xval, LB, UB, dep_best);

    % -------- deposit (iteration best) --------
    if opts.UseIterBestDeposit
        dep_iter = opts.IterBestWeight * (opts.Q / (iterBestJ + eps));
        [tauOrb, tauSlot] = deposit_structured(tauOrb, tauSlot, iterBestX, LB, UB, dep_iter);
    end

    fprintf('ACO iter %3d | bestJ = %.6g | iterBestJ = %.6g | stall = %d\n', ...
        itr, fval, iterBestJ, stallCount);

    if stallCount >= opts.StallIters
        fprintf('ACO stopping early (stall reached).\n');
        break;
    end
end
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
% Deposit pheromone on each pair's orbit choice and the corresponding slot choice.

    nVars = numel(LB);
    nPairs = nVars/2;

    for p = 1:nPairs
        io = 2*p - 1;
        is = 2*p;

        % convert values -> indices
        oIdx = x(io) - LB(io) + 1;    % 1..nOrbits(p)
        sIdx = x(is) - LB(is) + 1;    % 1..nSlots(p)

        % safety clamp (in case)
        oIdx = max(1, min(oIdx, UB(io)-LB(io)+1));
        sIdx = max(1, min(sIdx, UB(is)-LB(is)+1));

        tauOrb{p}(oIdx) = tauOrb{p}(oIdx) + deposit;

        % ensure slot vector exists, then deposit into it
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
