function [vis, theta, theta_min] = calc_visibility( ...
    r_target, r_observer, r_sun, mu, LU, ...
    sun_min, moon_min, earth_min)
    % calc_visibility - Earth, Moon, and Sun angular screening
    %
    % Positions are in LU; minimum angles are in radians.
    % Minimum angles are measured from each body's CENTER.
    %
    % Outputs theta and theta_min are ordered:
    %   [Earth, Moon, Sun]

    r_target   = r_target(:);
    r_observer = r_observer(:);
    r_sun      = r_sun(:);

    % line of sight
    los = r_target - r_observer;
    rho = norm(los);

    if rho == 0
        error('Observer and target positions must be different.');
    end

    u_los = los/rho;

    % body centers and radii
    r_bodies = [-mu, 1-mu, r_sun(1);
                  0,    0, r_sun(2);
                  0,    0, r_sun(3)];

    radii = [6378.1366, 1737.1, 695700]/LU;
    min_angles = [earth_min, moon_min, sun_min];

    theta     = nan(1,3);
    theta_min = nan(1,3);
    ok_body   = false(1,3);

    for b = 1:3
        body_vec = r_bodies(:,b) - r_observer;
        d = norm(body_vec);
        R = radii(b);

        % observer on or inside a body
        if d <= R
            theta_min(b) = pi;
            continue;
        end

        % observer-target/body-center angular separation
        u_body = body_vec/d;
        cos_theta = dot(u_los, u_body);
        theta(b) = acos(min(max(cos_theta, -1), 1));

        % physical occlusion threshold for a finite target range
        tangent_range = sqrt((d-R)*(d+R));

        if rho < d-R
            theta_occ = -Inf;

        elseif rho < tangent_range
            cos_occ = (d^2 + rho^2 - R^2)/(2*d*rho);
            theta_occ = acos(min(max(cos_occ, -1), 1));

        else
            theta_occ = asin(min(R/d, 1));
        end

        % common minimum angular separation
        theta_min(b) = max(theta_occ, min_angles(b));

        % tangency is blocked; equality at sensor threshold is allowed
        ok_body(b) = (theta(b) > theta_occ) && ...
                     (theta(b) >= min_angles(b));
    end

    vis = all(ok_body);
end