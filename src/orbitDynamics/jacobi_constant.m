function C = jacobi_constant(state,mu)
%JACOBI_CONSTANT Jacobi constant for Earth-Moon CR3BP rotating-frame states.
%
% state may be 1x6, 6x1, or Nx6. The returned C is Nx1.

validateattributes(mu,{'numeric'}, ...
    {'scalar','real','finite','>',0,'<',1});

state = double(state);
if isvector(state)
    state = reshape(state,1,[]);
end
assert(size(state,2)==6, ...
    'State must contain [x y z vx vy vz].');

x = state(:,1);
y = state(:,2);
z = state(:,3);
vx = state(:,4);
vy = state(:,5);
vz = state(:,6);

r1 = sqrt((x+mu).^2 + y.^2 + z.^2);
r2 = sqrt((x-(1-mu)).^2 + y.^2 + z.^2);

Omega = 0.5*(x.^2+y.^2) + (1-mu)./r1 + mu./r2;
C = 2*Omega - (vx.^2+vy.^2+vz.^2);
end
