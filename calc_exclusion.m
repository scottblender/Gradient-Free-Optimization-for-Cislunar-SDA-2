function [vis, alpha_sun, alpha_moon] = calc_exclusion(r_target, r_observer, r_sun, mu, sun_min, moon_min)
    % calc_exclusion - Computes lunar and sun exclusion angles and if
    % target is visible
    %
    % Inputs:
    %   r_target - position of the target spacecraft [rtx,rty,rtz] LU
    %   r_observer - position of the observer spacecraft [rox, roy, roz] LU
    %   r_sun - position of the sun in LU
    %   mu - EM graviational constant
    %   sun_min - minimum separation angle of the sun
    %   moon_min - minimum separation angle of the moon
    %
    % Output:
    %   vis - boolean if state is visible or not
    %   alpha_sun - sun exclusion angle in rad
    %   alpha_moon - lunar exclusion angle in rad
    
    % line of sight 
    los = r_target - r_observer;
    nlos = norm(los);
    u_los = los/nlos;

    % moon direction (CR3BP)
    r_moon = [1-mu; 0; 0];
    u_moon = (r_moon - r_observer)/norm((r_moon - r_observer));
    
    % sun direction
    u_sun = (r_sun - r_observer)/norm((r_sun - r_observer));

    % angles 
    alpha_sun = acos(clamp(dot(u_los, u_sun),-1,1));
    alpha_moon = acos(clamp(dot(u_los, u_moon),-1,1));

    vis = (alpha_moon >= moon_min) && (alpha_sun >= sun_min);
end

function y = clamp(x,a,b)
    y = min(max(x,a),b);
end