function [isVisible, details] = calc_visibility( ...
    r_target, r_observer, r_sun, mu, LU, visibilityCfg)
%CALC_VISIBILITY Earth/Moon/Sun occlusion and optional limb avoidance.
% Positions are 3-element vectors in the Earth-Moon barycentric rotating
% frame [LU]. LU is in km/LU. Row and column input vectors are accepted.
%
% Required configuration:
%   limbMargins_rad : [Earth Moon Sun], each in [0,pi]. These are margins
%                     beyond the apparent limb, NOT body-center angles.
% Optional configuration:
%   radii_km        : [Earth Moon Sun], default [6378.1366 1737.1 695700].
%   distanceTol_km  : nonnegative occlusion tolerance, default 0.
%
% For each body, theta is the observer's target/body-center separation and
% alpha = asin(R/d) is the body's angular radius. A positive margin rejects
% theta < alpha + margin, including targets in front of the body (angular
% avoidance). A zero margin disables that extra avoidance and retains exact
% finite-segment occlusion, so a foreground target is not falsely occulted.
% Tangency counts as occlusion; equality at the exclusion boundary passes.
% An endpoint on/inside a body is blocked. Coincident observer/target states
% are invalid because no line-of-sight direction exists.
%
% details fields are 1x3 arrays in body order Earth, Moon, Sun, except
% bodyNames and modelVersion. Occlusion/exclusion flags may overlap; use
% details.blocked (their union) when counting rejected measurements.
% The nominal solar radius follows IAU 2015 Resolution B3:
% https://arxiv.org/abs/1510.07674

validateattributes(r_target, {'numeric'}, ...
    {'real','finite','vector','numel',3}, mfilename, 'r_target');
validateattributes(r_observer, {'numeric'}, ...
    {'real','finite','vector','numel',3}, mfilename, 'r_observer');
validateattributes(r_sun, {'numeric'}, ...
    {'real','finite','vector','numel',3}, mfilename, 'r_sun');
validateattributes(mu, {'numeric'}, ...
    {'real','finite','scalar','>',0,'<',1}, mfilename, 'mu');
validateattributes(LU, {'numeric'}, ...
    {'real','finite','scalar','positive'}, mfilename, 'LU');

if nargin < 6 || ~isstruct(visibilityCfg) || ~isscalar(visibilityCfg) || ...
        ~isfield(visibilityCfg, 'limbMargins_rad')
    error('calc_visibility:MissingMargins', ...
        'Specify visibilityCfg.limbMargins_rad = [Earth Moon Sun] in radians.');
end
validateattributes(visibilityCfg.limbMargins_rad, {'numeric'}, ...
    {'real','finite','vector','numel',3,'>=',0,'<=',pi}, ...
    mfilename, 'visibilityCfg.limbMargins_rad');
if ~isfield(visibilityCfg, 'radii_km')
    visibilityCfg.radii_km = [6378.1366 1737.1 695700];
end
if ~isfield(visibilityCfg, 'distanceTol_km')
    visibilityCfg.distanceTol_km = 0;
end
validateattributes(visibilityCfg.radii_km, {'numeric'}, ...
    {'real','finite','vector','numel',3,'positive'}, ...
    mfilename, 'visibilityCfg.radii_km');
validateattributes(visibilityCfg.distanceTol_km, {'numeric'}, ...
    {'real','finite','scalar','nonnegative'}, ...
    mfilename, 'visibilityCfg.distanceTol_km');

r_target = double(r_target(:));
r_observer = double(r_observer(:));
r_sun = double(r_sun(:));
mu = double(mu);
LU = double(LU);
los = r_target - r_observer;
targetRange = norm(los);
if targetRange == 0
    error('calc_visibility:CoincidentPositions', ...
        'Observer and target must have different positions.');
end
u_los = los / targetRange;
centers = [-mu, 1-mu, r_sun(1); 0, 0, r_sun(2); 0, 0, r_sun(3)];
radii = double(visibilityCfg.radii_km(:).') / LU;
margins = double(visibilityCfg.limbMargins_rad(:).');
distanceTol = double(visibilityCfg.distanceTol_km) / LU;

details.modelVersion = 'finite_segment_limb_v1';
details.bodyNames = {'Earth','Moon','Sun'};
details.centerSeparation_rad = nan(1,3);
details.angularRadius_rad = nan(1,3);
details.limbClearance_rad = nan(1,3);
details.limbMargins_rad = margins;
details.bodyRange_LU = zeros(1,3);
details.closestDistance_LU = zeros(1,3);
details.occluded = false(1,3);
details.excluded = false(1,3);

for b = 1:3
    d = centers(:,b) - r_observer;
    bodyRange = norm(d);
    details.bodyRange_LU(b) = bodyRange;

    % Closest point on the finite observer-target segment. The shared d
    % and LOS direction also supply the angular test below.
    alongLOS = dot(d, u_los);
    closestRange = min(max(alongLOS, 0), targetRange);
    closestDistance = norm(d - closestRange*u_los);
    details.closestDistance_LU(b) = closestDistance;
    details.occluded(b) = closestDistance <= radii(b) + distanceTol;

    if bodyRange <= radii(b)
        % An apparent limb is undefined for an observer on/inside a body.
        details.occluded(b) = true;
        continue;
    end

    u_body = d / bodyRange;
    theta = atan2(norm(cross(u_los, u_body)), dot(u_los, u_body));
    alpha = asin(min(1, radii(b)/bodyRange));
    details.centerSeparation_rad(b) = theta;
    details.angularRadius_rad(b) = alpha;
    details.limbClearance_rad(b) = theta - alpha;
    details.excluded(b) = margins(b) > 0 && theta < alpha + margins(b);
end

details.blocked = details.occluded | details.excluded;
isVisible = ~any(details.blocked);
end
