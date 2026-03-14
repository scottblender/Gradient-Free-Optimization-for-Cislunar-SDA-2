function [t_target, s_target, info] = build_truth_gateway(cfg, mu, ode_opts)

t_target = (0:cfg.dt:cfg.Nperiods*cfg.period).';
if t_target(end) < cfg.Nperiods*cfg.period
    t_target = [t_target; cfg.Nperiods*cfg.period];
end

[t_target, s_target] = ode45(@(t,s) cr3bp_dynamics(t,s,mu), ...
    t_target, cfg.s0, ode_opts);

info = struct();
info.type = "LUNAR_GATEWAY";
info.builder = "build_truth_gateway";
info.period = cfg.period;
info.Nperiods = cfg.Nperiods;
info.dt = cfg.dt;
end