function [h] = measurement_model(r_target, r_observer, measCfg)
% r_target - position of the target spacecraft [rtx,rty,rtz]
% r_observer - position of the observer spacecraft [rox, roy, roz]
% measCfg - measurement configuration structure
% returns h - EKF measurement model:
%   [alpha; delta]             for angles only
%   [alpha; delta; range]      for angles with range

if nargin < 3 || isempty(measCfg)
    measCfg = struct();
    measCfg.type = "ANGLES_ONLY";
end

if ~isfield(measCfg, 'type') || isempty(measCfg.type)
    measCfg.type = "ANGLES_ONLY";
end

measType = upper(string(measCfg.type));

% Line of Sight (LOS) vector between s/c
rho_vec = r_target - r_observer;
rho_x = rho_vec(1);
rho_y = rho_vec(2);
rho_z = rho_vec(3); % components
rho = norm(rho_vec);

% Calculate the right ascension and declination
alpha = atan2(rho_y, rho_x); % right ascension
delta = asin(rho_z / rho);   % declination

switch measType
    case "ANGLES_ONLY"
        h = [alpha; delta];

    case "ANGLES_RANGE"
        h = [alpha; delta; rho];

    otherwise
        error('Unknown measurement model type: %s', measType);
end
end