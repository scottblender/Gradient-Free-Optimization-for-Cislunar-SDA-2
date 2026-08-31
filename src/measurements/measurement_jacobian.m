function [H] = measurement_jacobian(r_target, r_observer, measCfg)
% r_target is the position of the target spacecraft
% r_observer is the position of the ith observer spacecraft
% measCfg is the measurement configuration structure
% This function returns the Jacobian of either:
%   an angles only measurement, or
%   an angles + range measurement.

if nargin < 3 || isempty(measCfg)
    measCfg = struct();
    measCfg.type = "ANGLES_ONLY";
end

if ~isfield(measCfg, 'type') || isempty(measCfg.type)
    measCfg.type = "ANGLES_ONLY";
end

measType = upper(string(measCfg.type));

rho_vec = r_target - r_observer;
rho_x = rho_vec(1);
rho_y = rho_vec(2);
rho_z = rho_vec(3);

% Pre-compute common terms
rho = sqrt(rho_x^2 + rho_y^2 + rho_z^2);
q = rho_x^2 + rho_y^2;
s = sqrt(q);

switch measType
    case "ANGLES_ONLY"
        % Compute relative Jacobian
        Hp = zeros(2,3);
        Hp(1,1) = -rho_y/q;
        Hp(1,2) =  rho_x/q;

        Hp(2,1) = -(rho_z*rho_x)/(rho^2*s);
        Hp(2,2) = -(rho_z*rho_y)/(rho^2*s);
        Hp(2,3) =  s/rho^2;

    case "ANGLES_RANGE"
        % Compute relative Jacobian
        Hp = zeros(3,3);

        % Right ascension partials
        Hp(1,1) = -rho_y/q;
        Hp(1,2) =  rho_x/q;

        % Declination partials
        Hp(2,1) = -(rho_z*rho_x)/(rho^2*s);
        Hp(2,2) = -(rho_z*rho_y)/(rho^2*s);
        Hp(2,3) =  s/rho^2;

        % Range partials
        Hp(3,1) = rho_x/rho;
        Hp(3,2) = rho_y/rho;
        Hp(3,3) = rho_z/rho;

    otherwise
        error('Unknown measurement model type: %s', measType);
end

% translate to ekf state
H = [Hp zeros(size(Hp,1),3)];
end