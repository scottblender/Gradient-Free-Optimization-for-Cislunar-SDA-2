% ======================= Stopper.m =======================
classdef Stopper < handle
    % Stopper: evaluation-based stall stopping (client-side).
    %
    % Key idea:
    %  - Built-in optimizers (GA/PSO/bayesopt) may run objectives on workers.
    %    Do NOT count evals inside the objective; instead update from OutputFcn
    %    which runs on the client.
    %
    %  - Custom algorithms (ABC/ACO loops) can count evals in the loop and call
    %    update(bestNow, evalTotal, xBestNow) once per batch.

    properties
        % -------- configuration --------
        isMinimize (1,1) logical = true

        m        (1,1) double = 200    % patience window in evals
        epsAbs   (1,1) double = 1e-3   % abs improvement threshold over m evals (O(1) objective -> ~1e-3 is sane)
        nMin     (1,1) double = 50     % warmup evals before checking
        maxEvals (1,1) double = inf    % hard cap on evals

        % -------- state --------
        evalCount (1,1) double = 0
        Jbest     (1,1) double
        xBest                 = []

        % -------- full history (best-so-far) --------
        histFunccount double = []   % eval count at each update
        histBest      double = []   % best-so-far at each update

        stopFlag   (1,1) logical = false
        stopReason (1,:) char    = ''
    end

    methods
        function obj = Stopper(m, epsAbs, nMin, maxEvals, isMinimize)
            if nargin >= 1 && ~isempty(m),          obj.m = m; end
            if nargin >= 2 && ~isempty(epsAbs),     obj.epsAbs = epsAbs; end
            if nargin >= 3 && ~isempty(nMin),       obj.nMin = nMin; end
            if nargin >= 4 && ~isempty(maxEvals),   obj.maxEvals = maxEvals; end
            if nargin >= 5 && ~isempty(isMinimize), obj.isMinimize = isMinimize; end

            if obj.isMinimize
                obj.Jbest = +inf;
            else
                obj.Jbest = -inf;
            end
        end

        function reset(obj)
            obj.evalCount = 0;
            obj.xBest = [];
            obj.histFunccount = [];
            obj.histBest = [];
            obj.stopFlag = false;
            obj.stopReason = '';
            if obj.isMinimize
                obj.Jbest = +inf;
            else
                obj.Jbest = -inf;
            end
        end

        function update(obj, bestNow, funccountNow, xBestNow)
            % Update stall logic from client-side best + funccount.
            % bestNow must be the best objective value known at funccountNow.
            % xBestNow optional.

            if obj.stopFlag
                return;
            end

            % Make scalars if history vectors were passed by mistake
            if isnumeric(funccountNow) && ~isempty(funccountNow)
                funccountNow = funccountNow(end);
            end
            if isnumeric(bestNow) && ~isempty(bestNow)
                bestNow = bestNow(end);
            end

            obj.evalCount = funccountNow;

            % Update best-so-far (and xBest only if improved)
            improved = false;
            if obj.isMinimize
                if bestNow < obj.Jbest
                    obj.Jbest = bestNow;
                    improved = true;
                end
            else
                if bestNow > obj.Jbest
                    obj.Jbest = bestNow;
                    improved = true;
                end
            end

            if improved && nargin >= 4 && ~isempty(xBestNow)
                obj.xBest = xBestNow;
            end

            % Store FULL history (best-so-far each update)
            obj.histFunccount(end+1,1) = obj.evalCount;
            obj.histBest(end+1,1)      = obj.Jbest;

            % Hard cap
            if obj.evalCount >= obj.maxEvals
                obj.stopFlag = true;
                obj.stopReason = sprintf('Max evals reached: %d', obj.maxEvals);
                return;
            end

            % Warmup
            if obj.evalCount < obj.nMin
                return;
            end

            % Compare current best to best at (evalCount - m)
            target = obj.evalCount - obj.m;
            if target <= 0
                return;
            end

            idx = find(obj.histFunccount <= target, 1, 'last');
            if isempty(idx)
                return;
            end

            olderBest = obj.histBest(idx);
            newerBest = obj.Jbest;

            if obj.isMinimize
                impr = olderBest - newerBest;
            else
                impr = newerBest - olderBest;
            end

            if impr < obj.epsAbs
                obj.stopFlag = true;
                obj.stopReason = sprintf('Stall: impr < %.3g over last %d evals', obj.epsAbs, obj.m);
            end
        end

        function J = eval(obj, baseObjFcn, x)
            % Client-side objective wrapper (custom loops ONLY).
            % Counts every evaluation and applies stopping.
            if obj.stopFlag
                J = obj.Jbest;
                return;
            end

            J = baseObjFcn(x);

            fc = obj.evalCount + 1;

            if obj.isMinimize
                bestNow = min(obj.Jbest, J);
            else
                bestNow = max(obj.Jbest, J);
            end

            if (obj.isMinimize && J < obj.Jbest) || (~obj.isMinimize && J > obj.Jbest)
                obj.xBest = x;
            end

            obj.update(bestNow, fc, obj.xBest);
        end

        % ---------------- OutputFcn hooks ----------------
        function [state, options, optchanged] = gaOutputFcn(obj, options, state, flag)
            optchanged = false;

            if ~strcmp(flag,'iter')
                return;
            end

            % funccount
            fc = NaN;
            if isfield(state,'FunEval')
                fc = state.FunEval;
            elseif isfield(state,'Funccount')
                fc = state.Funccount;
            elseif isfield(state,'funccount')
                fc = state.funccount;
            elseif isfield(state,'Generation') && isfield(options,'PopulationSize')
                fc = (state.Generation+1) * options.PopulationSize;
            end
            if isnumeric(fc) && ~isempty(fc), fc = fc(end); end

            % best of current generation from Score (most reliable)
            bf = NaN;
            if isfield(state,'Score') && isnumeric(state.Score) && ~isempty(state.Score)
                bf = min(state.Score(:));
            elseif isfield(state,'Best') && isnumeric(state.Best) && ~isempty(state.Best)
                bf = state.Best(end);
            elseif isfield(state,'BestScore') && isnumeric(state.BestScore) && ~isempty(state.BestScore)
                bf = state.BestScore(end);
            end
            if isnumeric(bf) && ~isempty(bf), bf = bf(end); end

            if isfinite(fc) && isfinite(bf)
                obj.update(bf, fc);
            end

            if obj.stopFlag
                state.StopFlag = obj.stopReason;
            end
        end

        function stop = psoOutputFcn(obj, optimValues, state) %#ok<INUSD>
            fc = NaN; bf = NaN;
            if isfield(optimValues,'funccount'), fc = optimValues.funccount; end
            if isfield(optimValues,'bestfval'),  bf = optimValues.bestfval; end
            if isnan(bf) && isfield(optimValues,'bestf'), bf = optimValues.bestf; end

            if isnumeric(fc) && ~isempty(fc), fc = fc(end); end
            if isnumeric(bf) && ~isempty(bf), bf = bf(end); end

            if isfinite(fc) && isfinite(bf)
                obj.update(bf, fc);
            end
            stop = obj.stopFlag;
        end

        function stop = bayesOutputFcn(obj, results, state) %#ok<INUSD>
            % Runs on client
            y = results.ObjectiveTrace;
            funccountNow = sum(~isnan(y));   % # evaluated points
            bestNow = results.MinObjective;  % best observed objective
            obj.update(bestNow, funccountNow);
            stop = obj.stopFlag;
        end
    end
end
