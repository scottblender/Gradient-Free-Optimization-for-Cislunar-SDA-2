function test_visibility_keepout_definition()
%TEST_VISIBILITY_KEEPOUT_DEFINITION Verify the unified angular definition.

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
setup_project();

mu = 1.215058560962404E-2;
LU = 384400;

earthCenter = [-mu;0;0];
earthRadius = 6378.1366/LU;
observerDistance = 0.10;
targetRange = 0.20;

rObserver = earthCenter+[observerDistance;0;0];
rSun = [5;3;0.5];

thetaOccEarth = asin(earthRadius/observerDistance);
targetAtAngle = @(angle) rObserver+targetRange* ...
    [-cos(angle);sin(angle);0];

% Zero exclusion must reduce to physical occultation.
rBlocked = targetAtAngle(0.5*thetaOccEarth);
visBlocked = calc_visibility( ...
    rBlocked,rObserver,rSun,mu,LU,0,0,0);
assert(~visBlocked, ...
    'A sightline inside the Earth limb must be blocked.');

rClear = targetAtAngle(thetaOccEarth+deg2rad(2));
[visZero,~,physicalKeepout] = calc_visibility( ...
    rClear,rObserver,rSun,mu,LU,0,0,0);
assert(visZero, ...
    'A sightline outside all physical limbs must be visible.');
assert(abs(physicalKeepout(1)-thetaOccEarth) < 1e-12, ...
    'Zero Earth exclusion did not recover the occultation boundary.');

% An exclusion angle below the limb must not enlarge the keep-out region.
earthExclusionSmall = 0.5*thetaOccEarth;
[visSmall,~,keepoutSmall] = calc_visibility( ...
    rClear,rObserver,rSun,mu,LU,0,0,earthExclusionSmall);
assert(visSmall == visZero);
assert(abs(keepoutSmall(1)-thetaOccEarth) < 1e-12, ...
    'A sub-limb exclusion angle changed the physical boundary.');

% An exclusion angle above the limb must become the keep-out boundary.
earthExclusionLarge = thetaOccEarth+deg2rad(5);
[visLarge,theta,thetaKeepout] = calc_visibility( ...
    rClear,rObserver,rSun,mu,LU,0,0,earthExclusionLarge);
assert(~visLarge, ...
    'The enlarged Earth exclusion region did not reject the sightline.');
assert(abs(thetaKeepout(1)-earthExclusionLarge) < 1e-12, ...
    'The Earth keep-out boundary is not max(occultation,exclusion).');
assert(theta(1) > thetaOccEarth && ...
       theta(1) < earthExclusionLarge);

% Check the ordered vector definition for Earth, Moon, and Sun.
sunExclusion = deg2rad(20);
moonExclusion = deg2rad(10);
exclusionAngles = [ ...
    earthExclusionLarge,moonExclusion,sunExclusion];

[~,thetaConfigured,thetaKeepoutConfigured] = calc_visibility( ...
    rClear,rObserver,rSun,mu,LU, ...
    sunExclusion,moonExclusion,earthExclusionLarge);

expectedKeepout = max(physicalKeepout,exclusionAngles);
assert(max(abs(thetaKeepoutConfigured-expectedKeepout)) < 1e-12, ...
    'Keep-out vector does not follow [Earth,Moon,Sun] ordering.');
assert(max(abs(thetaConfigured-theta)) < 1e-12, ...
    'Exclusion settings changed the underlying separation angles.');

fprintf(['Visibility keep-out definition passed: ' ...
    'theta_keepout = max(theta_occ,theta_exclusion).\n']);
end
