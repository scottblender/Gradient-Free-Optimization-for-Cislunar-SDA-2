function [t_target, s_target, info] = build_truth_periodic_orbit( ...
    cfg, T1, times, states, mu, ode_opts)

iOrb = cfg.orbitIndex;
period = T1.("Period (TU) ")(iOrb);
s0 = states{iOrb}(1,:).';

t_target = (0:cfg.dt:cfg.Nperiods*period).';
if t_target(end) < cfg.Nperiods*period
    t_target = [t_target; cfg.Nperiods*period];
end

[t_target, s_target] = ode45(@(t,s) cr3bp_dynamics(t,s,mu), ...
    t_target, s0, ode_opts);

info = struct();
info.type = "PERIODIC_ORBIT";
info.builder = "build_truth_periodic_orbit";
info.orbitIndex = iOrb;
info.period = period;
info.Nperiods = cfg.Nperiods;
info.dt = cfg.dt;
end