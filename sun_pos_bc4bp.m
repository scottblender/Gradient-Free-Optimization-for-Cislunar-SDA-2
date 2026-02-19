function r_sun = sun_pos_bc4bp(t_lg, LU, TU, theta0, i_sun)
    % sun_pos_bc4bp - Computes sun vector using bicircular 4 body problem
    % dynamics
    %
    % Inputs:
    %   t_lg - propagation time of target (lunar gateway) in TU
    %   LU - length unit in EM system
    %   TU - time unit in EM system
    %   theta0 - initial sun phase angle (rad)
    %   i_sun - initial sun inclination angle out of EM plane (rad)
    %   
    %
    % Output:
    %   r_sun - 3x1 Sun position vector in LU 
    
    % initial distance of sun
    AU_km = 149597870.7;
    a_sun = AU_km*LU;
    
    % sun mean motion
    n_sun = (2*pi/(365.256363004 * 86400))*TU;
    
    % relative rotation in EM frame, sun drifts backwards
    omega = n_sun - 1;
    
    theta = theta0 + omega*t_lg;
    
    % planar sun vector
    r0 = a_sun * [cos(theta); sin(theta); 0];
    
    % incline out of plane if desired
    Rx = [1 0 0;
          0 cos(i_sun) -sin(i_sun);
          0 sin(i_sun)  cos(i_sun)];
    
    r_sun = Rx * r0;

end