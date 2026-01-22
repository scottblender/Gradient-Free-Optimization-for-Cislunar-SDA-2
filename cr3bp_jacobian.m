function [A] = cr3bp_jacobian(s, mu)
% s - state vector, [x, y, z, vx, vy, vz]
% mu - mass ratio (mass2/(mass1 + mass2))
% This function calculates the 6x6 dynamics jacobian (A matrix)


% position and velocity
x = s(1); y = s(2); z = s(3);

% define common terms
x1 = x + mu; x2 = x - 1 + mu;
r1 = sqrt(x1^2+y^2+z^2); r2 = sqrt(x2^2+y^2+z^2);
alpha1 = (1-mu)/r1^5; alpha2 = mu/r2^5;
beta1 = (1-mu)/r1^3; beta2 = mu/r2^3;
S = beta1 + beta2;

% diagonal entries 
Omega_xx = 1 - S + 3*(alpha1*x1^2 + alpha2*x2^2);
Omega_yy = 1 - S + 3*(alpha1 + alpha2)*y^2;
Omega_zz = -S + 3*(alpha1 + alpha2)*z^2;

% off-diagonal entries
Omega_xy = 3*(alpha1*x1 + alpha2*x2)*y;
Omega_xz = 3*(alpha1*x1 + alpha2*x2)*z;
Omega_yz = 3*(alpha1 + alpha2)*y*z;

% define pseudo-potential hessian
Omega_dd = zeros(3); % Initialize the Jacobian matrix
Omega_dd(1, 1) = Omega_xx; Omega_dd(1, 2) = Omega_xy; Omega_dd(1, 3) = Omega_xz;
Omega_dd(2, 1) = Omega_xy; Omega_dd(2, 2) = Omega_yy; Omega_dd(2, 3) = Omega_yz;
Omega_dd(3, 1) = Omega_xz; Omega_dd(3, 2) = Omega_yz; Omega_dd(3, 3) = Omega_zz;

% define coriolis 
C = 2*[0 1 0; -1 0 0; 0 0 0];

% define A matrix
A = [zeros(3) eye(3); Omega_dd C];
end
