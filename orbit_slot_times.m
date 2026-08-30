function t_slots = orbit_slot_times(period, slots_per_orbit)
%ORBIT_SLOT_TIMES Equal-time phase samples over one half-open period [0,T).
% Slot j corresponds to t_j = (j-1)*T/N, j = 1,...,N. These are equal
% time intervals, not equal arc lengths. The periodic endpoint is excluded
% so the final slot does not duplicate the first slot. N=1 returns t=0.
% Cache/result convention: equal_time_half_open_v1.

validateattributes(period, {'numeric'}, ...
    {'real','finite','scalar','positive'}, mfilename, 'period');
validateattributes(slots_per_orbit, {'numeric'}, ...
    {'real','finite','scalar','integer','positive'}, mfilename, 'slots_per_orbit');

t_slots = (0:double(slots_per_orbit)-1).' * ...
    (double(period) / double(slots_per_orbit));
end
