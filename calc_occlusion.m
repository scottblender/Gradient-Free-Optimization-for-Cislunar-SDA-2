function [occE, occM] = calc_occlusion(r_target, r_observer, mu, LU, pad_km)
    % calc_occlusion - Computes whether or not the earth or moon occlude
    % the LOS between the observer and target using a spherical model
    %
    % Inputs:
    %   r_target - position of the target spacecraft [rtx,rty,rtz] LU
    %   r_observer - position of the observer spacecraft [rox, roy, roz] LU
    %   mu - EM graviational constant
    %   LU - length unit
    %   pad_km - amount of padding to the sphere
    %
    % Output:
    %   occE - Earth occludes LOS observer->target
    %   occM - Moon occludes LOS observer->target
    if nargin < 5 || isempty(pad_km)
        pad_km = 0;
    end

    % body centers (LU)
    r_earth = [-mu; 0; 0];
    r_moon = [1-mu; 0; 0];

    % radii (LU)
    Re_LU = (6378.1363 + pad_km) / LU;
    Rm_LU = (1737.4    + pad_km) / LU;

    % determine occlusion
    occE = segment_intersects_sphere(r_observer, r_target, r_earth, Re_LU);
    occM = segment_intersects_sphere(r_observer, r_target, r_moon, Rm_LU);
end

function hit = segment_intersects_sphere(p0, p1, c, R)
% True if segment p0->p1 intersects sphere (center c, radius R)
% p0 is observer position, p1 is target position, c is body center
    v  = p1 - p0;
    vv = dot(v,v);
    if vv < 1e-14
        hit = false;
        return;
    end

    % closest point on segment to center
    t = dot(c - p0, v) / vv;
    t = min(max(t, 0), 1);
    p = p0 + t*v;

    d2 = dot(p - c, p - c);
    hit = (d2 <= R^2);
end