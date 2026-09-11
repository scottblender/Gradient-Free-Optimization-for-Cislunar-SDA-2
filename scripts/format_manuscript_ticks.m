function format_manuscript_ticks(ax)
%FORMAT_MANUSCRIPT_TICKS Reduce crowded tick labels on perspective 3-D axes.
% Construction-time only; retain data limits and all ordinary 2-D ticks.
%
% Two cases are treated centrally:
%   1) symmetric three-tick axes such as [-0.05 0 0.05] omit the center 0;
%   2) very small symmetric axes with five or more ticks (the LG y-axis is
%      the motivating case) retain only their two endpoint labels. Perspective
%      projection makes those dense near-zero labels overlap even when the
%      underlying numerical tick spacing is valid.

if abs(ax.View(2)-90) < 1e-8
    return;
end

for name = ["X","Y","Z"]
    ticks = ax.(name+"Tick");
    if numel(ticks) < 3
        continue;
    end

    span = ticks(end)-ticks(1);
    tolerance = max(1e-12,1e-10*max(1,max(abs(ticks))));
    hasZero = any(abs(ticks) <= tolerance);
    isSymmetric = ticks(1) < 0 && ticks(end) > 0 && hasZero && ...
        abs(ticks(1)+ticks(end)) <= tolerance;

    keep = [];
    if numel(ticks) == 3 && isSymmetric
        % Prior manuscript preference: remove the crowded center zero.
        keep = [1 3];
    elseif numel(ticks) >= 5 && isSymmetric && span <= 0.15
        % Small projected axes (e.g., approximately +/-0.04 LU) cannot carry
        % five bold labels legibly at manuscript size. Keep the range visible
        % with the two endpoint labels rather than stacking values near zero.
        keep = [1 numel(ticks)];
    end

    if isempty(keep)
        continue;
    end

    labels = ax.(name+"TickLabel");
    manual = strcmp(ax.(name+"TickLabelMode"),'manual');
    ax.(name+"Tick") = ticks(keep);

    if manual
        labels = cellstr(labels);
        if numel(labels) == numel(ticks)
            ax.(name+"TickLabel") = labels(keep);
        end
    end
end
end
