function [h] = measurement_model(r_target, r_observer)
% r_target - position of the target spacecraft [rtx,rty,rtz]
% r_observer - position of the observer spacecraft [rox, roy, roz]
% returns h - EKF measurement model [alpha (right ascension), beta (declination)]

% Line of Sight (LOS) vector between s/c
rho_vec = r_target - r_observer;
rho_x = rho_vec(1);
rho_y = rho_vec(2);
rho_z = rho_vec(3); % components
rho = norm(rho_vec);

% Calculate the right ascension and declination
alpha = atan2(rho_y, rho_x); % right ascension
delta = asin(rho_z / rho);   % declination

h = [alpha; delta];
end
