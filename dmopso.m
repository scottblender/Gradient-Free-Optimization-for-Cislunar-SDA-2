function [x_best, fval] = dmopso(obj_fnc, nvars, LB, UB, pop_size, max_itr, max_stall)
% --- initialize variables --- %
pos = LB + (UB - LB) .* rand(pop_size, nvars); % Continuous positions
vel = zeros(pop_size, nvars);
pbest_x = zeros(pop_size, nvars); % Will store rounded bests
pbest_f = zeros(pop_size, 3);
stall_counter = 0;
max_archive = 200;
best_knee = inf;
ftol = 1e-6;

archive.X = []; 
archive.F = [];

% --- 1. Initial Evaluation ---
fprintf('Initializing Population with %d particles...\n', pop_size);

% Create rounded version for the physics lookup
pos_eval = round(pos); 

parfor i = 1:pop_size
    pbest_f(i,:) = obj_fnc(pos_eval(i,:));
end

% Store rounded discrete values in personal bests and archive
pbest_x = pos_eval;
archive = update_archive(archive, pos_eval, pbest_f, max_archive);

for itr = 1:max_itr
    % Randomly pick a leader from the discrete archive
    narch = size(archive.F, 1);
    leader_idx = randi(narch, pop_size, 1);
    leaders = archive.X(leader_idx, :);
    
    w = 0.4 + 0.9*rand(pop_size, 1); 
    c1 = 1.5; c2 = 1.5;
    r1 = rand(pop_size, nvars);
    r2 = rand(pop_size, nvars);
    
    % --- 2. Continuous Swarm Update ---
    % Use decimals to maintain search "direction"
    vel = w .* vel + c1 .* r1 .* (pbest_x - pos) + c2 .* r2 .* (leaders - pos);
    pos = pos + vel;
    pos = max(LB, min(UB, pos)); % Keep within database bounds
    
    % --- 3. Discrete Evaluation ---
    % Round decimals to integers for the EKF
    pos_eval = round(pos); 
    
    new_fval = zeros(pop_size, 3);
    parfor i = 1:pop_size
        new_fval(i,:) = obj_fnc(pos_eval(i,:));
    end
    
    % --- 4. Personal Best & Archive Update ---
    for i = 1:pop_size
        if is_dominant(new_fval(i,:), pbest_f(i,:))
            % Save the rounded integer configuration
            pbest_x(i,:) = pos_eval(i,:);
            pbest_f(i,:) = new_fval(i,:);
        elseif ~is_dominant(pbest_f(i,:), new_fval(i,:))
            if rand > 0.5
                pbest_x(i,:) = pos_eval(i,:);
                pbest_f(i,:) = new_fval(i,:);
            end
        end
    end
    
    % Add rounded positions to the Pareto front
    archive = update_archive(archive, pos_eval, new_fval, max_archive);
    
    % Check for stall
    current_knee = find_knee_distance(archive.F);
    improvement = best_knee - current_knee;
    if improvement > ftol
        best_knee = current_knee;
        stall_counter = 0;
    else
        stall_counter = stall_counter + 1;
    end
    
    if stall_counter == round(max_stall * 0.5)
        fprintf('<<< STALL DETECTED: Re-randomizing 30%% of swarm >>>\n');
        
        % Re-randomize 30% of the worst particles
        shaken_idx = randperm(pop_size, round(pop_size * 0.3));
        
        % Give them completely new random positions
        pos(shaken_idx, :) = LB + (UB - LB) .* rand(length(shaken_idx), nvars);
        vel(shaken_idx, :) = 0; 
        
        % GRACE PERIOD: Reduce stall counter to give the swarm time to use the new info
        stall_counter = max(0, stall_counter - 5); 
    end

    fprintf('Iter %d/%d | Archive: %d | Best Knee Dist: %.4f | Stall: %d/%d\n', ...
        itr, max_itr, size(archive.F, 1), best_knee, stall_counter, max_stall);
        
    if stall_counter >= max_stall
        fprintf('\nOptimization converged.\n');
        break;
    end
end

x_best = archive.X; % Returns integers
fval = archive.F;
end

% -------------------------------------------------------------------------
% HELPER FUNCTIONS
% -------------------------------------------------------------------------

function d = is_dominant(f1, f2)
    % Returns true if f1 dominates f2 (Minimization)
    d = all(f1 <= f2) && any(f1 < f2);
end

function arch = update_archive(arch, newX, newF, limit)
    % Vectorized dominance sorting
    allX = [arch.X; newX];
    allF = [arch.F; newF];
    
    N = size(allF, 1);
    isDominated = false(N, 1);
    
    for i = 1:N
        % Check if row i is dominated by ANY other row in the current set
        better = all(allF <= allF(i,:), 2) & any(allF < allF(i,:), 2);
        if any(better)
            isDominated(i) = true;
        end
    end
    
    arch.X = allX(~isDominated, :);
    arch.F = allF(~isDominated, :);
    
    % Pruning via Crowding Distance if limit exceeded
    if size(arch.F, 1) > limit
        arch = prune_crowding(arch, limit);
    end
end

function arch = prune_crowding(arch, limit)
    % Maintains Pareto diversity using crowding distance
    F = arch.F;
    [N, M] = size(F);
    dist = zeros(N, 1);
    
    for m = 1:M
        [sortedF, idx] = sort(F(:, m));
        dist(idx(1)) = Inf; 
        dist(idx(end)) = Inf;
        
        valRange = sortedF(end) - sortedF(1);
        if valRange > 1e-9
            dist(idx(2:end-1)) = dist(idx(2:end-1)) + ...
                (sortedF(3:end) - sortedF(1:end-2)) / valRange;
        end
    end
    
    [~, sortIdx] = sort(dist, 'descend');
    keepIdx = sortIdx(1:limit);
    
    arch.X = arch.X(keepIdx, :);
    arch.F = arch.F(keepIdx, :);
end

function dist = find_knee_distance(F)
    % Distance to Utopia point (0,0,0) in normalized space
    f_min = min(F, [], 1);
    f_max = max(F, [], 1);
    range = f_max - f_min;
    range(range < 1e-9) = 1; 
    
    F_norm = (F - f_min) ./ range;
    dists = sqrt(sum(F_norm.^2, 2));
    dist = min(dists);
end