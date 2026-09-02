function [vis, theta, theta_keepout] = calc_visibility( ...
    r_target, r_observer, r_sun, mu, LU, ...
    sun_exclusion, moon_exclusion, earth_exclusion)
    % calc_visibility - Unified Earth, Moon, and Sun visibility screening
    %
    % Positions are in LU. Exclusion angles are in radians and are
    % absolute minimum LOS separations measured from each body's CENTER.
    %
    % For each body b,
    %   theta_keepout,b = max(theta_occ,b, theta_exclusion,b).
    % Therefore zero exclusion reproduces physical occultation.
    %
    % Outputs theta and theta_keepout are ordered:
    %   [Earth, Moon, Sun]

    r_target   = r_target(:);
    r_observer = r_observer(:);
    r_sun      = r_sun(:);

    % Line of sight.
    los = r_target-r_observer;
    rho = norm(los);

    if rho == 0
        error('Observer and target positions must be different.');
    end

    u_los = los/rho;

    % Body centers and radii.
    r_bodies = [-mu, 1-mu, r_sun(1);
                  0,    0, r_sun(2);
                  0,    0, r_sun(3)];

    radii = [6378.1366,1737.1,695700]/LU;
    exclusion_angles = [ ...
        earth_exclusion,moon_exclusion,sun_exclusion];

    theta = nan(1,3);
    theta_keepout = nan(1,3);
    ok_body = false(1,3);

    for b = 1:3
        body_vec = r_bodies(:,b)-r_observer;
        d = norm(body_vec);
        R = radii(b);

        % An observer on or inside a body cannot provide visibility.
        if d <= R
            theta_keepout(b) = pi;
            continue;
        end

        % LOS/body-center angular separation.
        u_body = body_vec/d;
        cos_theta = dot(u_los,u_body);
        theta(b) = acos(min(max(cos_theta,-1),1));

        % Physical occultation threshold for a finite target range.
        tangent_range = sqrt((d-R)*(d+R));

        if rho < d-R
            theta_occ = -Inf;

        elseif rho < tangent_range
            cos_occ = (d^2+rho^2-R^2)/(2*d*rho);
            theta_occ = acos(min(max(cos_occ,-1),1));

        else
            theta_occ = asin(min(R/d,1));
        end

        % Unified center-referenced keep-out boundary.
        theta_keepout(b) = max(theta_occ,exclusion_angles(b));

        % Physical tangency is blocked. Equality at a configured exclusion
        % boundary is allowed because it satisfies the minimum separation.
        ok_body(b) = (theta(b) > theta_occ) && ...
                     (theta(b) >= exclusion_angles(b));
    end

    vis = all(ok_body);
end
