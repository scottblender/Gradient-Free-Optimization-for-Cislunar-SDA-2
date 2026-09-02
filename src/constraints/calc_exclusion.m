function [vis, alpha_sun, alpha_moon] = calc_exclusion( ...
    r_target, r_observer, r_sun, mu, ...
    sun_exclusion, moon_exclusion)
    % calc_exclusion - Legacy center-referenced exclusion reference
    %
    % This function is retained only for visibility regression tests.
    % Optimization uses calc_visibility.
    %
    % Inputs:
    %   r_target - target position [LU]
    %   r_observer - observer position [LU]
    %   r_sun - Sun position [LU]
    %   mu - Earth-Moon mass ratio
    %   sun_exclusion - minimum Sun-center separation [rad]
    %   moon_exclusion - minimum Moon-center separation [rad]
    %
    % Outputs:
    %   vis - true when both exclusion constraints are satisfied
    %   alpha_sun - LOS/Sun-center separation [rad]
    %   alpha_moon - LOS/Moon-center separation [rad]

    los = r_target-r_observer;
    u_los = los/norm(los);

    r_moon = [1-mu;0;0];
    u_moon = (r_moon-r_observer)/norm(r_moon-r_observer);
    u_sun = (r_sun-r_observer)/norm(r_sun-r_observer);

    alpha_sun = acos(clamp(dot(u_los,u_sun),-1,1));
    alpha_moon = acos(clamp(dot(u_los,u_moon),-1,1));

    vis = (alpha_moon >= moon_exclusion) && ...
          (alpha_sun >= sun_exclusion);
end

function y = clamp(x,a,b)
    y = min(max(x,a),b);
end
