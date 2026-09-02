function launch_optimization_gui()
% GUI launcher + live monitor for run_opt.m
% Launches a NEW MATLAB batch process so this GUI is not closed by:
% clear; close all; clc;

    projectPaths = setup_project();

    screen = get(groot,'ScreenSize');
    figW = min(1000, max(760, screen(3)-80));
    figH = min(680, max(520, screen(4)-110));
    figX = max(20, round((screen(3)-figW)/2));
    figY = max(40, round((screen(4)-figH)/2));

    fig = uifigure( ...
        'Name','Optimization Launcher', ...
        'Position',[figX figY figW figH], ...
        'Resize','off');

    % ===================== LEFT: CONTROLS =====================
    leftW = min(450, max(350, round(0.43*figW)));
    leftPanel = uipanel(fig, ...
        'Title','Run Configuration', ...
        'Position',[10 10 leftW figH-20], ...
        'Scrollable','on');

    % Keep the controls inside the visible panel on normal laptop screens.
    % The panel remains scrollable for unusually short displays.
    contentH = max(600, figH-60);
    y  = contentH - 42;
    dy = 36;
    labelX = 18;
    fieldX = 165;
    rightPad = 38;  % reserve space for the vertical scroll bar
    fieldW = leftW - fieldX - rightPad;

    lblMission = uilabel(leftPanel, 'Position',[labelX y 140 22], 'Text','Mission Type');
    ddMission = uidropdown(leftPanel, ...
        'Position',[fieldX y fieldW 22], ...
        'Items',{'LOW_THRUST_TRANSFER','LUNAR_GATEWAY','GATEWAY_IMPULSE','PERIODIC_ORBIT'}, ...
        'Value','LOW_THRUST_TRANSFER');

    y = y - dy;
    uilabel(leftPanel, 'Position',[labelX y 140 22], 'Text','Optimizer');
    ddOpt = uidropdown(leftPanel, ...
        'Position',[fieldX y fieldW 22], ...
        'Items',{'GA','PSO','BAYESIAN','ABC','ACO'}, ...
        'Value','GA');

    y = y - dy;
    uilabel(leftPanel, 'Position',[labelX y 140 22], 'Text','Num Observers');
    efNumObs = uieditfield(leftPanel, 'text', ...
        'Position',[fieldX y fieldW 22], 'Value','3');

    y = y - dy;
    uilabel(leftPanel, 'Position',[labelX y 140 22], 'Text','Max Evaluations');
    efMaxEvals = uieditfield(leftPanel, 'text', ...
        'Position',[fieldX y fieldW 22], 'Value','6000');

    y = y - dy;
    uilabel(leftPanel, 'Position',[labelX y 140 22], 'Text','EKF Step (TU)');
    efEkfDt = uieditfield(leftPanel, 'text', ...
        'Position',[fieldX y fieldW 22], 'Value','0.01');

    y = y - dy;
    lblNPeriods = uilabel(leftPanel, 'Position',[labelX y 140 22], 'Text','Num Periods');
    efNPeriods = uieditfield(leftPanel, 'text', ...
        'Position',[fieldX y fieldW 22], 'Value','1');

    y = y - dy;
    uilabel(leftPanel, 'Position',[labelX y 140 22], 'Text','Optimizer Seed');
    efSeed = uieditfield(leftPanel, 'text', ...
        'Position',[fieldX y fieldW 22], 'Value','0');

    y = y - dy;
    uilabel(leftPanel, 'Position',[labelX y 140 22], 'Text','Run Directory');
    efRunDir = uieditfield(leftPanel, 'text', ...
        'Position',[fieldX y fieldW-36 22], ...
        'Value', fullfile(projectPaths.runs, ...
            ['gui_' char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))]));
    uibutton(leftPanel, 'push', ...
        'Position',[fieldX+fieldW-31 y 30 22], ...
        'Text','...', ...
        'ButtonPushedFcn', @(~,~) browseRunDir());

    y = y - dy - 2;
    cbScreen = uicheckbox(leftPanel, ...
        'Position',[labelX y 135 22], 'Text','Use Screening', 'Value',true);
    cbJ1 = uicheckbox(leftPanel, ...
        'Position',[labelX+145 y 65 22], 'Text','J1', 'Value',true);
    cbJ2 = uicheckbox(leftPanel, ...
        'Position',[labelX+215 y 65 22], 'Text','J2', 'Value',true);
    cbJ3 = uicheckbox(leftPanel, ...
        'Position',[labelX+285 y 65 22], 'Text','J3', 'Value',true);

    y = y - 112;
    txtInfo = uitextarea(leftPanel, ...
        'Position',[18 y leftW-56 98], ...
        'Editable','off', ...
        'Value',{ ...
            'Starts a separate MATLAB batch process.', ...
            'MAX_EVALS is the only optimizer stopping budget.', ...
            'Measurement-noise seed is fixed at 1001.', ...
            'Saved study runs contain data only; plots are disabled.', ...
            'Gateway impulse: 10 m/s prograde, propagated 1.5 TU.'});

    y = y - 48;
    gap = 8;
    controlW = leftW - 56;
    btnW = floor((controlW-2*gap)/3);
    uibutton(leftPanel, 'push', ...
        'Position',[18 y btnW 34], ...
        'Text','Launch', ...
        'ButtonPushedFcn', @(~,~) launchRun());
    uibutton(leftPanel, 'push', ...
        'Position',[18+btnW+gap y btnW 34], ...
        'Text','Show Command', ...
        'ButtonPushedFcn', @(~,~) showCommand());
    uibutton(leftPanel, 'push', ...
        'Position',[18+2*(btnW+gap) y btnW 34], ...
        'Text','Refresh', ...
        'ButtonPushedFcn', @(~,~) updateMonitor());

    y = y - 46;
    btnW2 = floor((controlW-gap)/2);
    uibutton(leftPanel, 'push', ...
        'Position',[18 y btnW2 34], ...
        'Text','Stop Runs', ...
        'ButtonPushedFcn', @(~,~) stopBatchRuns());
    uibutton(leftPanel, 'push', ...
        'Position',[18+btnW2+gap y btnW2 34], ...
        'Text','Stop + Launch Fresh', ...
        'ButtonPushedFcn', @(~,~) stopAndLaunchFresh());

    % ===================== RIGHT: MONITOR =====================
    rightX = leftW + 20;
    rightW = figW - rightX - 10;
    monitorPanel = uipanel(fig, ...
        'Title','Run Monitor', ...
        'Position',[rightX 10 rightW figH-20]);

    topY = figH - 75;
    uilabel(monitorPanel, 'Position',[15 topY 70 22], 'Text','Latest Log');
    efLogFile = uieditfield(monitorPanel, 'text', ...
        'Position',[85 topY rightW-145 22], ...
        'Editable','off', 'Value','');
    uibutton(monitorPanel, 'push', ...
        'Position',[rightW-50 topY 35 22], ...
        'Text','...', ...
        'ButtonPushedFcn', @(~,~) browseLogFile());

    colW = floor((rightW-45)/2);
    col2X = 25 + colW;
    statusY = topY - 34;
    lblStatus  = uilabel(monitorPanel, 'Position',[15 statusY colW 22], 'Text','Status: idle');
    lblUpdated = uilabel(monitorPanel, 'Position',[col2X statusY colW 22], 'Text','Last update: --');

    statusY = statusY - 26;
    lblBestCost = uilabel(monitorPanel, 'Position',[15 statusY colW 22], 'Text','Best cost: --');
    lblRuntime  = uilabel(monitorPanel, 'Position',[col2X statusY colW 22], 'Text','Runtime: --');

    statusY = statusY - 26;
    lblRmsePos = uilabel(monitorPanel, 'Position',[15 statusY colW 22], 'Text','RMSE pos (km): --');
    lblRmseVel = uilabel(monitorPanel, 'Position',[col2X statusY colW 22], 'Text','RMSE vel (km/s): --');

    statusY = statusY - 26;
    lblSigmaPos = uilabel(monitorPanel, ...
        'Position',[15 statusY rightW-30 22], ...
        'Text','Effective position uncertainty (km): --');

    lblFE = uilabel(monitorPanel, ...
        'Position',[15 statusY-30 rightW-30 22], ...
        'Text','Function evaluations: 0 / --');

    % Draw the endpoint labels separately so the red gauge needle remains
    % below the text instead of crossing the minimum-value label.
    lblFEMin = uilabel(monitorPanel, ...
        'Position',[15 statusY-54 60 22], ...
        'Text','0', ...
        'HorizontalAlignment','left');
    lblFEMax = uilabel(monitorPanel, ...
        'Position',[rightW-75 statusY-54 60 22], ...
        'Text','1', ...
        'HorizontalAlignment','right');
    gaugeFE = uigauge(monitorPanel, 'linear', ...
        'Position',[15 statusY-102 rightW-30 42], ...
        'Limits',[0 1], ...
        'Value',0, ...
        'MajorTicks',[0 1], ...
        'MajorTickLabels',{});

    logTop = statusY - 115;
    txtLog = uitextarea(monitorPanel, ...
        'Position',[15 15 rightW-30 max(180,logTop-20)], ...
        'Editable','off', ...
        'FontName','Consolas', ...
        'Value', {''} );

    isManualLogSelection = false;

    ddMission.ValueChangedFcn = @(~,~) toggleFields();
    toggleFields();
    resetMonitor();

    logTimer = timer( ...
        'ExecutionMode','fixedRate', ...
        'Period',2.0, ...
        'BusyMode','drop', ...
        'TimerFcn', @(~,~) updateMonitor());

    start(logTimer);

    fig.CloseRequestFcn = @(~,~) closeGUI();

    function browseRunDir()
        d = uigetdir(efRunDir.Value, 'Select Run Directory');
        if isequal(d,0)
            return;
        end
        efRunDir.Value = char(d);
        efLogFile.Value = '';
        isManualLogSelection = false;
        resetMonitor();
        updateMonitor();
    end

    function browseLogFile()
        [f,p] = uigetfile({'*.txt;*.log','Log files (*.txt, *.log)'}, 'Select log file', efRunDir.Value);
        if isequal(f,0)
            return;
        end
        efLogFile.Value = fullfile(p,f);
        isManualLogSelection = true;
        updateMonitor();
    end

    function toggleFields()
        usesPeriods = ismember(ddMission.Value, ...
            {'LUNAR_GATEWAY','PERIODIC_ORBIT'});
        if usesPeriods
            efNPeriods.Enable = 'on';
            lblNPeriods.Enable = 'on';
        else
            efNPeriods.Enable = 'off';
            lblNPeriods.Enable = 'off';
        end
    end

    function showCommand()
        try
            params = collectParams();
            cmd = buildLaunchCommand(params);
            disp('Launch command:');
            disp(cmd);
            uialert(fig, 'Command printed to Command Window.', 'Command');
        catch ME
            uialert(fig, ME.message, 'Input Error');
        end
    end

    function launchRun()
        try
            params = collectParams();
            ensureRunDir(params.RUN_DIR);

            efLogFile.Value = '';
            isManualLogSelection = false;
            resetMonitor();
            lblStatus.Text = 'Status: launching...';
            drawnow;

            cmd = buildLaunchCommand(params);

            disp('Launching command:');
            disp(cmd);

            [status, out] = system(cmd);
            if ~isempty(out)
                disp(out);
            end

            if status == 0
                lblStatus.Text = 'Status: launched';
            else
                lblStatus.Text = 'Status: launch failed';
                uialert(fig, sprintf('Launch failed:\n\n%s', out), 'Launch Error');
            end

            pause(1.0);
            efLogFile.Value = '';
            isManualLogSelection = false;
            updateMonitor();

        catch ME
            uialert(fig, ME.message, 'Input Error');
        end
    end

    function stopBatchRuns()
        try
            msg = stop_other_batch_processes();
            lblStatus.Text = 'Status: batch stop requested';
            uialert(fig, msg, 'Stop Batch Runs');
            pause(0.5);
            efLogFile.Value = '';
            isManualLogSelection = false;
            updateMonitor();
        catch ME
            uialert(fig, ME.message, 'Stop Error');
        end
    end

    function stopAndLaunchFresh()
        try
            msg = stop_other_batch_processes();
            disp(msg);
            pause(1.0);
            launchRun();
        catch ME
            uialert(fig, ME.message, 'Fresh Launch Error');
        end
    end

    function params = collectParams()
        params = struct();

        params.MISSION_TYPE   = char(ddMission.Value);
        params.OPTIMIZER_MODE = char(ddOpt.Value);

        params.NUM_OBSERVERS = validatePositiveInteger(efNumObs.Value, 'NUM_OBSERVERS');
        params.MAX_EVALS     = validatePositiveInteger(efMaxEvals.Value, 'MAX_EVALS');
        params.EKF_DT        = validatePositiveScalar(efEkfDt.Value, 'EKF_DT');
        params.SEED          = validateInteger(efSeed.Value, 'SEED');

        if ~ismember(params.MISSION_TYPE, {'LUNAR_GATEWAY','PERIODIC_ORBIT'})
            params.NPERIODS = '1';
        else
            params.NPERIODS = validatePositiveInteger(efNPeriods.Value, 'NPERIODS');
        end

        params.RUN_DIR = strtrim(efRunDir.Value);
        if isempty(params.RUN_DIR)
            error('RUN_DIR cannot be empty.');
        end

        params.USE_SCREENING = boolToEnv(cbScreen.Value);
        params.USE_J1        = boolToEnv(cbJ1.Value);
        params.USE_J2        = boolToEnv(cbJ2.Value);
        params.USE_J3        = boolToEnv(cbJ3.Value);
    end

    function updateMonitor()
        try
            if isManualLogSelection
                manualLog = strtrim(efLogFile.Value);
                if ~isempty(manualLog) && isfile(manualLog)
                    logFile = string(manualLog);
                else
                    lblStatus.Text = 'Status: selected log not found';
                    lblUpdated.Text = ['Last update: ' char(datetime('now','Format','HH:mm:ss'))];
                    txtLog.Value = {''};
                    drawnow limitrate;
                    return;
                end
            else
                logFile = findLatestLog(efRunDir.Value);
                if strlength(logFile) == 0
                    lblStatus.Text = 'Status: no log found yet';
                    lblUpdated.Text = ['Last update: ' char(datetime('now','Format','HH:mm:ss'))];
                    txtLog.Value = {''};
                    drawnow limitrate;
                    return;
                end
                efLogFile.Value = char(logFile);
            end

            txt = fileread(logFile);
            lblUpdated.Text = ['Last update: ' char(datetime('now','Format','HH:mm:ss'))];

            if contains(txt, 'RUN END:')
                lblStatus.Text = 'Status: finished';
            elseif contains(txt, 'RUN START:')
                lblStatus.Text = 'Status: running';
            else
                lblStatus.Text = 'Status: log found';
            end

            maxChars = 14000;
            txtStr = string(txt);
            if strlength(txtStr) > maxChars
                txtTail = extractAfter(txtStr, strlength(txtStr) - maxChars);
            else
                txtTail = txtStr;
            end
            txtLog.Value = cellstr(splitlines(txtTail));

            bestCost = parseLastMatch(txt, 'bestJ\s*=\s*([\-+0-9.eE]+)');
            if isempty(bestCost)
                bestCost = parseLastMatch(txt, 'Cost:\s*([\-+0-9.eE]+)');
            end
            if isempty(bestCost)
                bestCost = parseLastMatch(txt, 'min_cost\s*=\s*([\-+0-9.eE]+)');
            end
            if ~isempty(bestCost)
                lblBestCost.Text = ['Best cost: ' bestCost];
            else
                lblBestCost.Text = 'Best cost: --';
            end

            runtimeVal = parseLastMatch(txt, 'Total Runtime:\s*([\-+0-9.eE]+)\s*seconds');
            if ~isempty(runtimeVal)
                lblRuntime.Text = ['Runtime: ' runtimeVal ' s'];
            else
                lblRuntime.Text = 'Runtime: --';
            end

            rmsePos = parseLastMatch(txt, 'RMSE position \(km\):\s*([\-+0-9.eE]+)');
            if ~isempty(rmsePos)
                lblRmsePos.Text = ['RMSE pos (km): ' rmsePos];
            else
                lblRmsePos.Text = 'RMSE pos (km): --';
            end

            rmseVel = parseLastMatch(txt, 'RMSE velocity \(km/s\):\s*([\-+0-9.eE]+)');
            if ~isempty(rmseVel)
                lblRmseVel.Text = ['RMSE vel (km/s): ' rmseVel];
            else
                lblRmseVel.Text = 'RMSE vel (km/s): --';
            end

            feVal = parseLastMatch(txt, 'Search FE\s*=\s*([0-9]+)');
            if isempty(feVal)
                feVal = parseLastMatch(txt, 'FE\s*=\s*([0-9]+)');
            end
            maxFE = str2double(efMaxEvals.Value);
            currentFE = str2double(feVal);
            if ~isfinite(maxFE) || maxFE < 1
                maxFE = 1;
            end
            if ~isfinite(currentFE) || currentFE < 0
                currentFE = 0;
            end

            % Completed runs retain authoritative metrics in the saved state.
            stateFile = fullfile(efRunDir.Value,'data','optimization_run.mat');
            if isfile(stateFile)
                try
                    saved = load(stateFile,'runState');
                    if isfield(saved,'runState')
                        rs = saved.runState;
                        if isfield(rs,'bestJ') && isfinite(rs.bestJ)
                            lblBestCost.Text = sprintf('Best cost: %.12g',rs.bestJ);
                        end
                        if isfield(rs,'runtime_s') && isfinite(rs.runtime_s)
                            lblRuntime.Text = sprintf('Runtime: %.2f s',rs.runtime_s);
                        end
                        if isfield(rs,'nEvaluations') && isfinite(rs.nEvaluations)
                            currentFE = rs.nEvaluations;
                        end
                        if isfield(rs,'maxEvaluations') && isfinite(rs.maxEvaluations)
                            maxFE = rs.maxEvaluations;
                        end
                        if isfield(rs,'metrics')
                            metrics = rs.metrics;
                            if isfield(metrics,'rmse_pos_km')
                                lblRmsePos.Text = sprintf('RMSE pos (km): %.6g',metrics.rmse_pos_km);
                            end
                            if isfield(metrics,'rmse_vel_kms')
                                lblRmseVel.Text = sprintf('RMSE vel (km/s): %.6g',metrics.rmse_vel_kms);
                            end
                            if isfield(metrics,'mean_effective_sigma_pos_km')
                                lblSigmaPos.Text = sprintf( ...
                                    'Effective position uncertainty (km): %.6g', ...
                                    metrics.mean_effective_sigma_pos_km);
                            end
                        end
                    end
                catch
                    % The batch process may be replacing the MAT-file.
                end
            end

            maxFE = max(1,round(maxFE));
            currentFE = max(0,min(round(currentFE),maxFE));
            gaugeFE.Limits = [0 maxFE];
            gaugeFE.MajorTicks = unique([0 round(maxFE/2) maxFE]);
            gaugeFE.MajorTickLabels = {};
            gaugeFE.Value = currentFE;
            lblFEMax.Text = sprintf('%d',maxFE);
            lblFE.Text = sprintf('Function evaluations: %d / %d',currentFE,maxFE);

            drawnow limitrate;

        catch ME
            lblStatus.Text = 'Status: monitor error';
            txtLog.Value = {ME.message};
            drawnow limitrate;
        end
    end

    function resetMonitor()
        lblStatus.Text   = 'Status: idle';
        lblUpdated.Text  = 'Last update: --';
        lblBestCost.Text = 'Best cost: --';
        lblRuntime.Text  = 'Runtime: --';
        lblRmsePos.Text  = 'RMSE pos (km): --';
        lblRmseVel.Text  = 'RMSE vel (km/s): --';
        lblSigmaPos.Text = 'Effective position uncertainty (km): --';
        lblFE.Text       = 'Function evaluations: 0 / --';
        gaugeFE.Limits   = [0 1];
        gaugeFE.MajorTicks = [0 1];
        gaugeFE.MajorTickLabels = {};
        gaugeFE.Value    = 0;
        lblFEMin.Text    = '0';
        lblFEMax.Text    = '1';
        txtLog.Value     = {''};
        drawnow;
    end

    function closeGUI()
        try
            stop(logTimer);
            delete(logTimer);
        catch
        end
        delete(fig);
    end
end

function cmd = buildLaunchCommand(params)
    scriptPath = fullfile(fileparts(mfilename('fullpath')), 'run_opt.m');
    if ~isfile(scriptPath)
        error('Could not find run_opt.m in the project root.');
    end

    ensureRunDir(params.RUN_DIR);

    matlabExe = 'matlab';
    launchLog = fullfile(params.RUN_DIR, ...
        ['gui_launch_' char(datetime('now','Format','yyyyMMdd_HHmmss')) '.log']);

    batchCmd = sprintf([ ...
        'setenv(''MISSION_TYPE'',''%s'');' ...
        'setenv(''OPTIMIZER_MODE'',''%s'');' ...
        'setenv(''NUM_OBSERVERS'',''%s'');' ...
        'setenv(''MAX_EVALS'',''%s'');' ...
        'setenv(''EKF_DT'',''%s'');' ...
        'setenv(''NPERIODS'',''%s'');' ...
        'setenv(''SEED'',''%s'');' ...
        'setenv(''MEAS_NOISE_SEED'',''1001'');' ...
        'setenv(''MAKE_PLOTS'',''0'');' ...
        'setenv(''RUN_DIR'',''%s'');' ...
        'setenv(''USE_SCREENING'',''%s'');' ...
        'setenv(''USE_J1'',''%s'');' ...
        'setenv(''USE_J2'',''%s'');' ...
        'setenv(''USE_J3'',''%s'');' ...
        'run(''%s'');' ], ...
        escapeSingleQuotes(params.MISSION_TYPE), ...
        escapeSingleQuotes(params.OPTIMIZER_MODE), ...
        escapeSingleQuotes(params.NUM_OBSERVERS), ...
        escapeSingleQuotes(params.MAX_EVALS), ...
        escapeSingleQuotes(params.EKF_DT), ...
        escapeSingleQuotes(params.NPERIODS), ...
        escapeSingleQuotes(params.SEED), ...
        escapeSingleQuotes(params.RUN_DIR), ...
        escapeSingleQuotes(params.USE_SCREENING), ...
        escapeSingleQuotes(params.USE_J1), ...
        escapeSingleQuotes(params.USE_J2), ...
        escapeSingleQuotes(params.USE_J3), ...
        escapeSingleQuotes(scriptPath));

    if ispc
        cmd = sprintf('start "" %s -logfile "%s" -batch "%s"', ...
            matlabExe, launchLog, batchCmd);
    else
        cmd = sprintf('%s -logfile "%s" -batch "%s" &', ...
            matlabExe, launchLog, batchCmd);
    end
end

function ensureRunDir(runDir)
    if ~exist(runDir, 'dir')
        mkdir(runDir);
    end
end

function logFile = findLatestLog(runDir)
    logFile = "";

    if isempty(runDir) || ~isfolder(runDir)
        return;
    end

    files1 = dir(fullfile(runDir, '**', 'safe_output_fallback_*.txt'));
    files2 = dir(fullfile(runDir, '**', 'matlab_diary_*.txt'));
    files3 = dir(fullfile(runDir, '**', 'gui_launch_*.log'));
    files  = [files1; files2; files3];

    if isempty(files)
        return;
    end

    nonEmpty = files([files.bytes] > 0);
    if ~isempty(nonEmpty)
        files = nonEmpty;
    end

    [~, idx] = max([files.datenum]);
    logFile = string(fullfile(files(idx).folder, files(idx).name));
end

function msg = stop_other_batch_processes()
    if ispc
        thisPid = feature('getpid');
        psCmd = sprintf([ ...
            'powershell -NoProfile -Command "', ...
            '$p = Get-CimInstance Win32_Process | Where-Object { ', ...
            '$_.Name -ieq ''MATLAB.exe'' -and $_.CommandLine -match ''-batch'' -and $_.ProcessId -ne %d }; ', ...
            'if ($p) { $ids = $p.ProcessId; Stop-Process -Id $ids -Force; ', ...
            'Write-Output (''Stopped PIDs: '' + ($ids -join '', '')) } ', ...
            'else { Write-Output ''No other MATLAB batch processes found.'' }"', ...
            ], thisPid);
        [status, out] = system(psCmd);
        if status ~= 0
            error('Failed to stop batch runs.\n%s', out);
        end
        msg = strtrim(out);
    else
        killCmd = 'pkill -f "matlab.*-batch"';
        [status, out] = system(killCmd);
        if status == 0
            msg = 'Requested stop for MATLAB batch processes.';
        else
            msg = strtrim(out);
            if isempty(msg)
                msg = 'No MATLAB batch processes found, or pkill returned no matches.';
            end
        end
    end
end

function val = parseLastMatch(txt, expr)
    val = '';
    toks = regexp(txt, expr, 'tokens');
    if ~isempty(toks)
        lastTok = toks{end};
        if iscell(lastTok) && ~isempty(lastTok)
            val = char(lastTok{1});
        end
    end
end

function s = validatePositiveInteger(v, name)
    x = str2double(strtrim(v));
    if isnan(x) || ~isfinite(x) || x < 1 || abs(x - round(x)) > 0
        error('%s must be a positive integer.', name);
    end
    s = sprintf('%d', round(x));
end

function s = validateInteger(v, name)
    x = str2double(strtrim(v));
    if isnan(x) || ~isfinite(x) || abs(x - round(x)) > 0
        error('%s must be an integer.', name);
    end
    s = sprintf('%d', round(x));
end

function s = validatePositiveScalar(v, name)
    x = str2double(strtrim(v));
    if isnan(x) || ~isfinite(x) || x <= 0
        error('%s must be a positive scalar.', name);
    end
    s = sprintf('%.15g', x);
end

function s = boolToEnv(tf)
    if tf
        s = '1';
    else
        s = '0';
    end
end

function s = escapeSingleQuotes(s)
    s = strrep(char(s), '''', '''''');
end
