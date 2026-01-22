function [A] = cr3bp_jacobian(s, pi2)
% s - state vector, [x, y, z, vx, vy, vz]
% pi2 - mass2/(mass1 + mass2)
% This function calculates the 6x6 dynamics jacobian (A matrix)

% position and velocity
x = s(1); y = s(2); z = s(3);
% vx, vy, vz are not needed for this calculation

% mass properties
pi1 = 1 - pi2;

% position components relative to primaries
x_p_pi2 = x + pi2;         % x-component to m1
x_m_1_p_pi2 = x - 1 + pi2; % x-component to m2

% distances squared
r13_sq = x_p_pi2^2 + y^2 + z^2;
r23_sq = x_m_1_p_pi2^2 + y^2 + z^2;

% pre-compute powers of distances
r13_3 = r13_sq^(3/2);
r13_5 = r13_sq^(5/2);
r23_3 = r23_sq^(3/2);
r23_5 = r23_sq^(5/2);

% --- Build A(2,1) sub-matrix (the "gravity gradient") ---
% These are the second partial derivatives (Uxx, Uxy, etc.)
% of the pseudo-potential function.
Uxx = 1 - (pi1/r13_3 - (3*pi1*x_p_pi2^2)/r13_5) - ...
          (pi2/r23_3 - (3*pi2*x_m_1_p_pi2^2)/r23_5);

Uxy = (3*pi1*x_p_pi2*y)/r13_5 + (3*pi2*x_m_1_p_pi2*y)/r23_5;

Uxz = (3*pi1*x_p_pi2*z)/r13_5 + (3*pi2*x_m_1_p_pi2*z)/r23_5;

Uyy = 1 - (pi1/r13_3 - (3*pi1*y^2)/r13_5) - ...
          (pi2/r23_3 - (3*pi2*y^2)/r23_5);

Uyz = (3*pi1*y*z)/r13_5 + (3*pi2*y*z)/r23_5;

Uzz = - (pi1/r13_3 - (3*pi1*z^2)/r13_5) - ...
        (pi2/r23_3 - (3*pi2*z^2)/r23_5);

% Jacobian is symmetric (Uyx = Uxy, etc.)
A21 = [Uxx, Uxy, Uxz;
       Uxy, Uyy, Uyz;
       Uxz, Uyz, Uzz];

% --- Build A(2,2) sub-matrix (Coriolis terms) ---
% These are the partials of acceleration w.r.t velocity
A22 = [ 0,  2,  0;
       -2,  0,  0;
        0,  0,  0];

% --- Assemble the full 6x6 A matrix ---
% Top-left is d(f_pos)/d(pos) = zeros(3,3)
% Top-right is d(f_pos)/d(vel) = eye(3,3)
A = [ zeros(3,3),  eye(3,3);
      A21,         A22     ];
end
