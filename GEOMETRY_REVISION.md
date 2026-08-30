# Geometry and phase-slot revision

This revision fixes the Sun distance in `sun_pos_bc4bp.m`, adds
`calc_visibility.m`, and changes all orbit-slot builders to equal-time
samples on `[0,T)`. It does not change stochastic optimization, evaluation
budgets, or the EKF call signature. The EKF still calls the legacy screening
functions until the next integration step.

## Sun

`a_sun = 149597870.7 / LU`, where LU is km per nondimensional length unit.
The rotating-frame angular speed, initial phase, and inclination convention
are unchanged. This remains an idealized bicircular Sun, not an ephemeris.

## Visibility helper

```matlab
% Geometry-only validation: zero extra angular margins.
visibilityCfg.limbMargins_rad = [0 0 0]; % Earth, Moon, Sun
[ok, details] = calc_visibility( ...
    r_target, r_observer, r_sun, mu, LU, visibilityCfg);

% For sensor avoidance, explicitly supply the chosen limb margins:
% visibilityCfg.limbMargins_rad = deg2rad([earth_deg moon_deg sun_deg]);
```

Positive margins are measured beyond the apparent limb. They are not the
old body-center thresholds (20 degrees Sun, 10 degrees Moon); do not copy
those numbers without choosing the revised sensor assumption. The helper
requires all three margins so there is no invented Earth-exclusion default.

Zero margin means finite-segment occlusion only, including tangency.
A target in front of a body's near surface is not occulted merely because
its apparent direction overlaps the body. Positive margins also impose
angular avoidance for such foreground alignments. The zero-margin switch
is an explicit policy; do not describe its foreground behavior as a
continuous angular-limit identity in the paper.

Default spherical radii are Earth 6378.1366 km, Moon 1737.1 km (retaining the
existing project convention), and Sun 695700 km (IAU 2015 Resolution B3,
https://arxiv.org/abs/1510.07674). Optional `radii_km` overrides all three;
`distanceTol_km` defaults to zero and affects only occlusion. Use a small,
documented tolerance if numerical tangency classification requires it.
Observer/target coincidence raises an error. Observer positions on/inside
a body are blocked, with undefined angular diagnostics recorded as NaN.

`details.occluded`, `details.excluded`, and their union `details.blocked`
are in Earth/Moon/Sun order. Reason flags may overlap; count the union
once per rejected measurement. Separation and limb-clearance diagnostics
are in radians.

## Slots and cache migration

For N slots, slot j is at `(j-1)*T/N`, j=1,...,N. Thus 50 slots sample
`0, T/50, ..., 49*T/50`. Slots are equally spaced in time, not arc length.
This removes the duplicate endpoint but does not eliminate phase-grid
resolution error or change optimizer boundary handling.

`orbit_slot_times.m` is used by optimization, baseline recomputation,
observer-IC reconstruction, and catalog/slot figures. New orbit caches have
`_halfopen_v1` names and
record `slot_definition = "equal_time_half_open_v1"`. New summaries also
record that convention and `slots_per_orbit`.

Transfer cache keys have the same version suffix: departure slot 10 now
means 9*T/50 rather than 9*T/49, so the low-thrust trajectory must be rebuilt.
Old caches are retained and not overwritten. Monte Carlo caches additionally
identify the corrected Sun model, and legacy caches are rejected.

Legacy summary slot indices cannot be reinterpreted on this new grid.
Recomputation/reconstruction rejects summaries without matching slot
metadata; use the original code revision for historical results. Do not add
the new version label to an old summary. Existing historical plots and
summary tables can still be read without recomputing the objective.

## Checks

In MATLAB, from the repository root:

```matlab
addpath('tests');
test_geometry_and_slots
```

The tests need no orbit catalog or optimization toolbox. They check Sun
units/phase, cyclic slot spacing, all three occulting bodies, foreground
targets, tangency, limb thresholds at different ranges, invalid geometry,
row/column inputs, and zero-margin agreement with the legacy Earth/Moon
occlusion function. Full optimization still requires the local JPL catalog.
