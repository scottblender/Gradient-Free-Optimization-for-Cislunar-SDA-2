function test_geometry_and_slots
%TEST_GEOMETRY_AND_SLOTS Geometry regression checks, no catalog/toolboxes.
% From the repository root: addpath('tests'); test_geometry_and_slots

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(rootDir);

% Sun units, phase, inclination, and rotating-frame angular speed.
LU = 384400;
TU = 375695;
a = 149597870.7 / LU;
assert(norm(sun_pos_bc4bp(0, LU, TU, 0, 0) - [a;0;0]) < 1e-10);
assert(norm(sun_pos_bc4bp(0, LU, TU, pi/2, pi/2) - [0;0;a]) < 1e-10);
omega = (2*pi/(365.256363004*86400))*TU - 1;
dt = 0.25;
r = sun_pos_bc4bp(dt, LU, TU, 0, 0);
assert(abs(norm(r) - a) < 1e-10);
assert(abs(atan2(r(2), r(1)) - omega*dt) < 1e-12);

% Equal temporal spacing, cyclic closure, and no duplicated phase endpoint.
for N = [1, 2, 50, 100]
    T = 1.51110546287394;
    t = orbit_slot_times(T, N);
    assert(isequal(size(t), [N,1]));
    assert(t(1) == 0 && all(t >= 0) && all(t < T));
    assert(numel(unique(t)) == N);
    assert(max(abs(diff([t; T]) - T/N)) < 1e-14);
end
t = orbit_slot_times(50,50);
assert(t(50) == 49);

% Synthetic bodies with analytically known geometry.
mu = 0.1;
sun = [0;5;0];
centers = [-mu, 1-mu, sun(1); 0,0,sun(2); 0,0,sun(3)];
cfg.limbMargins_rad = [0 0 0];
cfg.radii_km = [0.1 0.1 0.1];

for b = 1:3
    c = centers(:,b);
    observer = c + [0;-1;0];

    % Target behind the body: finite segment crosses its diameter.
    [visible, d] = calc_visibility(c+[0;1;0], observer, sun, mu, 1, cfg);
    assert(~visible && d.occluded(b) && ~d.excluded(b));

    % Foreground target aligned with body center: not physically occulted.
    target = c + [0;-0.5;0];
    [visible, d] = calc_visibility(target, observer, sun, mu, 1, cfg);
    assert(visible && ~d.occluded(b));
    assert(d.limbClearance_rad(b) < 0);

    % The same foreground alignment fails positive angular avoidance.
    positiveCfg = cfg;
    positiveCfg.limbMargins_rad(b) = 0.05;
    [visible, d] = calc_visibility(target, observer, sun, mu, 1, positiveCfg);
    assert(~visible && ~d.occluded(b) && d.excluded(b));

    % Exact segment tangent and nearby clear line, parallel to the y axis.
    [~, d] = calc_visibility(c+[0.1;1;0], c+[0.1;-1;0], sun, mu, 1, cfg);
    % Permit only floating-point roundoff in the constructed tangent.
    assert(abs(d.closestDistance_LU(b) - 0.1) < 1e-14);
    tangentCfg = cfg;
    tangentCfg.distanceTol_km = 1e-12;
    [~, d] = calc_visibility(c+[0.1;1;0], c+[0.1;-1;0], sun, mu, 1, tangentCfg);
    assert(d.occluded(b));
    [~, d] = calc_visibility(c+[0.11;1;0], c+[0.11;-1;0], sun, mu, 1, cfg);
    assert(~d.occluded(b));

    % Target/observer inside a body always fails physical visibility.
    [visible, d] = calc_visibility(c, observer, sun, mu, 1, cfg);
    assert(~visible && d.occluded(b));
    [visible, d] = calc_visibility(observer, c, sun, mu, 1, cfg);
    assert(~visible && d.occluded(b) && isnan(d.angularRadius_rad(b)));

    % Threshold is measured from the limb, and varies with observer range.
    for distance = [1, 2]
        observer = c + [0;-distance;0];
        alpha = asin(0.1/distance);
        for offset = [-1e-6, 1e-6]
            angle = alpha + 0.2 + offset;
            target = observer + 2*distance*[sin(angle);cos(angle);0];
            marginCfg = cfg;
            marginCfg.limbMargins_rad(b) = 0.2;
            [~, d] = calc_visibility(target, observer, sun, mu, 1, marginCfg);
            assert(~d.occluded(b));
            assert(d.excluded(b) == (offset < 0));
            assert(abs(d.angularRadius_rad(b) - alpha) < 1e-12);
        end
    end
end

% Zero-margin Earth/Moon results agree with the previous implementation.
% Random external positions plus targeted occultations exercise the check.
oldRng = rng;
restoreRng = onCleanup(@() rng(oldRng)); %#ok<NASGU>
rng(17, 'twister');
mu = 1.215058560962404E-2;
sun = sun_pos_bc4bp(0, LU, TU, 0, 0);
cfg = struct('limbMargins_rad', [0 0 0]);
for k = 1:200
    observer = randn(3,1);
    target = randn(3,1);
    [occE, occM] = calc_occlusion(target, observer, mu, LU);
    [visible, d] = calc_visibility(target, observer, sun, mu, LU, cfg);
    assert(isequal(d.occluded(1:2), [occE occM]));
    [visibleRow, dRow] = calc_visibility(target.', observer.', sun.', mu, LU, cfg);
    assert(visible == visibleRow && isequal(d.blocked, dRow.blocked));
end

% Equivalent radius conversion: km/LU versus already nondimensional radii.
observer = [0.2;0.2;0.1];
target = [0.9;0.3;0.2];
[v1,d1] = calc_visibility(target, observer, sun, mu, LU, cfg);
% Keep the same nondimensional centers and express radii directly using LU=1.
scaledCfg = cfg;
scaledCfg.radii_km = [6378.1366 1737.1 695700] / LU;
[v2,d2] = calc_visibility(target, observer, sun, mu, 1, scaledCfg);
assert(v1 == v2 && isequal(d1.blocked, d2.blocked));
assert(max(abs(d1.angularRadius_rad-d2.angularRadius_rad)) < 1e-12);

mustError(@() calc_visibility(target,target,sun,mu,LU,cfg), ...
    'calc_visibility:CoincidentPositions');
mustError(@() calc_visibility(target,observer,sun,mu,LU,struct()), ...
    'calc_visibility:MissingMargins');

fprintf('All Sun, visibility, and orbit-slot checks passed.\n');
end

function mustError(fcn, identifier)
try
    fcn();
catch ME
    assert(strcmp(ME.identifier, identifier), ...
        'Expected %s; received %s.', identifier, ME.identifier);
    return;
end
error('test_geometry_and_slots:MissingError', 'Expected %s.', identifier);
end
