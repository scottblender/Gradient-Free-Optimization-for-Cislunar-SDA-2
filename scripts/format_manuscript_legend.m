function format_manuscript_legend(ax,lgd,style,plotPosition)
%FORMAT_MANUSCRIPT_LEGEND Format legends without wasting manuscript space.
%
% Two-dimensional result plots keep MATLAB's working north-outside layout so
% axes labels remain inside the canvas. Three-dimensional trajectory plots use
% a dedicated centered legend immediately above the centered fixed plot box.
% No export-time reformatting is performed.

labels = string(lgd.String);
labels = contextual_legend_labels(labels);
labels = abbreviate_manuscript_text(labels);
lgd.String = cellstr(labels);
lgd.FontName = style.fontName;

% MATLAB reports [0 90] for ordinary 2-D axes. Perspective/3-D axes get a
% slightly larger dedicated legend font; 2-D figures retain their compact font.
is3D = abs(ax.View(2)-90) > 1e-8;
if is3D && isfield(style,'geometryLegendFontSize')
    lgd.FontSize = style.geometryLegendFontSize;
elseif isfield(style,'legendFontSize')
    lgd.FontSize = style.legendFontSize;
else
    lgd.FontSize = style.fontSize;
end
lgd.FontWeight = 'bold';
lgd.Box = 'off';

% Preserve in-axes legends such as the DRO panel. For north-outside legends,
% prefer one row and wrap only when the row is physically too wide.
isNorthOutside = strcmpi(string(lgd.Location),"northoutside");
if isNorthOutside
    lgd.Orientation = 'horizontal';
    columns = min(numel(labels),style.legendMaxColumns);
    lgd.NumColumns = max(columns,1);
    lgd.Units = 'normalized';
    drawnow;

    % Comparison legends read best with all optimizer names adjacent and the
    % long-run GA reference last. Shorten/shrink that row before wrapping it.
    hasGaReference = any(labels == "GA (6000 FE)");
    if hasGaReference
        while lgd.Position(3) > 0.94 && lgd.FontSize > 12
            lgd.FontSize = lgd.FontSize-1;
            drawnow;
        end
    end

    while lgd.Position(3) > 0.94 && columns > 1
        columns = columns-1;
        lgd.NumColumns = columns;
        drawnow;
    end
end

if is3D && isNorthOutside
    lgd.Units = 'normalized';
    drawnow;
    pos = lgd.Position;

    % Center the legend exactly in the exported canvas, independent of the
    % legend width, then place it immediately above the centered 3-D plot box.
    pos(1) = 0.5-pos(3)/2;
    legendBottom = plotPosition(2)+plotPosition(4)+style.geometryLegendGap;
    pos(2) = min(legendBottom,0.98-pos(4));
    lgd.Position = pos;
    lgd.AutoUpdate = 'off';

    % Restore the centered manuscript plot box after MATLAB creates/moves the
    % legend. This prevents northoutside from shifting the 3-D axes off-center.
    ax.Units = 'normalized';
    ax.PositionConstraint = 'innerposition';
    ax.Position = plotPosition;
    drawnow;
end

% Remove only the crowded central zero from symmetric three-tick 3-D axes.
format_manuscript_ticks(ax);
end


function labels = contextual_legend_labels(labels)
%CONTEXTUAL_LEGEND_LABELS Use compact mission-specific trajectory labels.
labels = string(labels);

hasEndpoints = any(labels == "Endpoint orbits");
hasTargetTrajectory = any(labels == "Target trajectory");

% Result-geometry legends.
if hasTargetTrajectory
    if hasEndpoints
        labels(labels == "Target trajectory") = "LT";
    elseif any(labels == "Nominal Gateway")
        labels(labels == "Target trajectory") = "GI";
        labels(labels == "Nominal Gateway") = "Nominal LG";
    else
        labels(labels == "Target trajectory") = "LG";
    end
end

% Study-definition trajectory legends.
if any(labels == "Transfer")
    labels(labels == "Transfer") = "LT";
end
if any(labels == "Post-impulse")
    labels(labels == "Post-impulse") = "GI";
    labels(labels == "Nominal Gateway") = "Nominal LG";
elseif any(labels == "Nominal Gateway") && ~hasTargetTrajectory
    labels(labels == "Nominal Gateway") = "LG";
end

labels(labels == "Observer orbits") = "Observers";
labels(labels == "Endpoint orbits") = "Endpoints";
labels(labels == "Candidate slots") = "Slots";
labels(labels == "Excluded endpoint") = "Endpoint";

% Do not abbreviate "L1 point" or "L2 point" in the orbit-database plots;
% the family labels L1/L2 and the physical equilibrium points must be distinct.
end
