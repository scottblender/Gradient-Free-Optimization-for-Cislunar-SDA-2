function [ds] = cr3bp_dynamics(t, s, mu)
% t - propagation time [start time, end time]
% s - state vector [x, y, z, vx, vy, vz] or [s, Phi]
% mu - constant, mass2/(mass 1 + mass 2)
% returns ds - derivative of state vector [vx, vy, vz, ax, ay, az] or [ds;
%dpPhi] if STM is in s

% position vectors
x  = s(1);
y  = s(2);
z  = s(3);
vx = s(4);
vy = s(5);
vz = s(6);

r13 = [x + mu,     y, z];     % distance between earth and s/c
r23 = [x + mu - 1, y, z];     % distance between moon and s/c

r13_mag = norm(r13);
r23_mag = norm(r23);

% initialize state vector 
ds_state = zeros(1,6);

% velocity terms
ds_state(1:3) = [vx vy vz];

% acceleration terms
ax =  2*vy + (x - ((1-mu)/r13_mag^3)*(x+mu) - (mu/r23_mag^3)*(x-1+mu));
ay = -2*vx + (y - ((1-mu)/r13_mag^3)*y     - (mu/r23_mag^3)*y);
az = -((1-mu)/r13_mag^3)*z - (mu/r23_mag^3)*z;

ds_state(4:6) = [ax ay az];
ds_state = ds_state'; % ode45 expeects columns

% STM propagation
if length(s) == 42
        % Extract the current STM (Phi) from the vector
        Phi = reshape(s(7:42), 6, 6);
        
        % Calculate Jacobian A at the CURRENT state s
        A = cr3bp_jacobian(s, mu); 
        
        % Calculate Phi_dot = A * Phi
        Phi_dot = A * Phi;
        
        % Reshape Phi_dot back to a vector
        dPhi = reshape(Phi_dot, 36, 1);
        
        % Combine derivatives
        ds = [ds_state; dPhi];
    else
        % Just return state derivatives if no STM propagation needed
        ds = ds_state;
end

end
