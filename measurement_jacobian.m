function [H] = measurement_jacobian(r_target, r_observer)
% r_target is the position of the target spacecraft
% r_observer is the position of the ith observer spacecraft
% This function returns the Jacobian of an angles only measurement.
rho_vec = r_target - r_observer;
rho_x = rho_vec(1);
rho_y = rho_vec(2);
rho_z = rho_vec(3);

% Pre-compute common terms
rho_xy_sq = rho_x^2 + rho_y^2;
rho_xy = sqrt(rho_xy_sq);
rho_sq = rho_xy_sq + rho_z^2;
rho = sqrt(rho_sq);

% Partial derivatives for Right Ascension (RA)
dRA_dx = -rho_y / rho_xy_sq;
dRA_dy =  rho_x / rho_xy_sq;
dRA_dz =  0;

% Partial derivatives for Declination (Dec)
% Using: d(asin(u))/dx = (1/sqrt(1-u^2)) * du/dx, where u = rho_z / rho
dDec_dx = (rho_xy * (-rho_x * rho_z)) / (rho_sq * rho_xy);
dDec_dy = (rho_xy * (-rho_y * rho_z)) / (rho_sq * rho_xy);
dDec_dz = rho_xy / rho_sq;

% Handle potential singularity if rho_xy is near zero (poles)
if rho_xy < 1e-10 
    dDec_dx = 0;
    dDec_dy = 0;
else
    dDec_dx = -rho_x * rho_z / (rho_sq * rho_xy);
    dDec_dy = -rho_y * rho_z / (rho_sq * rho_xy);
end

H = [dRA_dx, dRA_dy, dRA_dz, 0, 0, 0;
     dDec_dx, dDec_dy, dDec_dz, 0, 0, 0];
end