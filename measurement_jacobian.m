function [H] = measurement_jacobian(r_target, r_observer)
% r_target is the position of the target spacecraft
% r_observer is the position of the ith observer spacecraft
% This function returns the Jacobian of an angles only measurement.
rho_vec = r_target - r_observer;
rho_x = rho_vec(1);
rho_y = rho_vec(2);
rho_z = rho_vec(3);

% Pre-compute common terms

rho = sqrt(rho_x^2 + rho_y^2 + rho_z^2);
q = rho_x^2 + rho_y^2; s = sqrt(q); 

% Compute relative Jacobian
Hp = zeros(2,3);
Hp(1,1) = -rho_y/q; Hp(1,2) = rho_x/q;
Hp(2,1) = - (rho_z*rho_x)/(rho^2*s); Hp(2,2) = -(rho_z*rho_y)/(rho^2*s); 
Hp(2,3) = s/rho^2;

% translate to ekf state
H = [Hp zeros(2,3)];
end