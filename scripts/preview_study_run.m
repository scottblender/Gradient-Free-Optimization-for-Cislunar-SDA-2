function preview_study_run(DataDir)
% Display-only diagnostic figures. Never export or save figure files.
S = load(fullfile(DataDir,'optimization_run.mat'),'runState');
D = load(fullfile(DataDir,'tracking_data.mat'),'tracking');
r = S.runState;
d = D.tracking;
err = d.estimate-d.truth;
LU = r.settings.LU;
VU = LU/r.settings.TU;
figure('Color','w','Name','Study preview');
tiledlayout(2,2,'TileSpacing','compact');
nexttile;
stairs(r.history.fe,r.history.bestJ,'LineWidth',1.5);
xlabel('Function evaluations'); ylabel('Best objective'); grid on;
nexttile;
plot(d.t_TU,vecnorm(err(:,1:3),2,2)*LU);
xlabel('Time (TU)'); ylabel('Position error (km)'); grid on;
nexttile;
plot(d.t_TU,vecnorm(err(:,4:6),2,2)*VU);
xlabel('Time (TU)'); ylabel('Velocity error (km/s)'); grid on;
nexttile;
stairs(d.t_TU,d.availableObsCount);
xlabel('Time (TU)'); ylabel('Available observers'); grid on;
end
