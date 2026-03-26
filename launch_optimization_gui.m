function launch_optimization_gui()
% GUI launcher + live monitor for run_opt.m
% Launches a NEW MATLAB batch process so this GUI is not closed by:
% clear; close all; clc;

    figW = 1180;
    figH = 800;

    fig = uifigure( ...
        'Name','Optimization Launcher', ...
        'Position',[100 100 figW figH]);

    % ===================== LEFT: CONTROLS =====================
    leftW = 540;
    leftPanel = uipanel(fig, ...
        'Title','Run Configuration', ...
        'Position',[10 10 leftW figH-20], ...
        'Scrollable','on');

    contentH = 1040;
    y  = contentH - 50;
    dy = 42;
    labelX = 20;
    fieldX = 210;
    fieldW = 290;

    % ---------------- Mission Type ----------------
    lblMission = uilabel(leftPanel, 'Position',[labelX y 160 22], 'Text','Mission Type');
    ddMission = uidropdown(leftPanel, ...
        'Position',[fieldX y fieldW 22], ...
        'Items',{'LOW_THRUST_TRANSFER','LUNAR_GATEWAY','PERIODIC_ORBIT'}, ...
        'Value','LOW_THRUST_TRANSFER');

    y = y - dy;

    % ---------------- Optimizer ----------------
    lblOpt = uilabel(leftPanel, 'Position',[labelX y 160 22], 'Text','Optimizer');
    ddOpt = uidropdown(leftPanel, ...
        'Position',[fieldX y fieldW 22], ...
        'Items',{'GA','PSO','BAYESIAN','GAMULTIOBJ','DMOPSO','ABC','ACO'}, ...
        'Value','GA');

    y = y - dy;

    % ---------------- Number of Observers ----------------
    lblNumObs = uilabel(leftPanel, 'Position',[labelX y 160 22], 'Text','Num Observers');
    efNumObs = uieditfield(leftPanel, 'text', ...
        'Position',[fieldX y fieldW 22], ...
        'Value','3');

    y = y - dy;

    % ---------------- Max Iterations ----------------
    lblMaxIters = uilabel(leftPanel, 'Position',[labelX y 160 22], 'Text','Max Iters');
    efMaxIters = uieditfield(leftPanel, 'text', ...
        'Position',[fieldX y fieldW 22], ...
        'Value','5');

    y = y - dy;

    % ---------------- Max Evaluations ----------------
    lblMaxEvals = uilabel(leftPanel, 'Position',[labelX y 160 22], 'Text','Max Evals');
    efMaxEvals = uieditfield(leftPanel, 'text', ...
        'Position',[fieldX y fieldW 22], ...
        'Value','100');

    y = y - dy;

    % ---------------- EKF DT ----------------
    lblEkfDt = uilabel(leftPanel, 'Position',[labelX y 160 22], 'Text','EKF DT');
    efEkfDt = uieditfield(leftPanel, 'text', ...
        'Position',[fieldX y fieldW 22], ...
        'Value','0.01');

    y = y - dy;

    % ---------------- Number of Periods ----------------
    lblNPeriods = uilabel(leftPanel, 'Position',[labelX y 160 22], 'Text','N Periods');
    efNPeriods = uieditfield(leftPanel, 'text', ...
        'Position',[fieldX y fieldW 22], ...
        'Value','1');

    y = y - dy;

    % ---------------- Seed ----------------
    lblSeed = uilabel(leftPanel, 'Position',[labelX y 160 22], 'Text','Seed');
    efSeed = uieditfield(leftPanel, 'text', ...
        'Position',[fieldX y fieldW 22], ...
        'Value','0');

    y = y - dy;

    % ---------------- Run Directory ----------------
    lblRunDir = uilabel(leftPanel, 'Position',[labelX y 160 22], 'Text','Run Directory');
    efRunDir = uieditfield(leftPanel, 'text', ...
        'Position',[fieldX y 250 22], ...
        'Value', pwd);

    btnBrowseRunDir = uibutton(leftPanel, 'push', ...
        'Position',[fieldX+260 y 30 22], ...
        'Text','...', ...
        'ButtonPushedFcn', @(~,~) browseRunDir());

    y = y - dy - 8;

    % ---------------- Checkboxes ----------------
    cbScreen = uicheckbox(leftPanel, ...
        'Position',[labelX y 180 22], ...
        'Text','Use Screening', ...
        'Value',true);

    y = y - dy + 6;

    cbJ1 = uicheckbox(leftPanel, ...
        'Position',[labelX y 100 22], ...
        'Text','Use J1', ...
        'Value',true);

    cbJ2 = uicheckbox(leftPanel, ...
        'Position',[labelX+120 y 100 22], ...
        'Text','Use J2', ...
        'Value',true);

    cbJ3 = uicheckbox(leftPanel, ...
        'Position',[labelX+240 y 100 22], ...
        'Text','Use J3', ...
        'Value',true);

    y = y - 55;

    % ---------------- Info box ----------------
    txtInfo = uitextarea(leftPanel, ...
        'Position',[20 y-140 490 140], ...
        'Editable','off', ...
        'Value',{ ...
            'This launcher sets environment variables and starts a NEW MATLAB batch process.', ...
            'It uses -logfile to make detached runs more robust.', ...
            'Stop Batch Runs kills OTHER MATLAB processes started with -batch.', ...
            'Your current GUI session is left alone.', ...
            'MAX_ITERS applies to most optimizers.', ...
            'MAX_EVALS applies to BAYESIAN.', ...
            'NPERIODS applies to LUNAR_GATEWAY and PERIODIC_ORBIT only.'});

    y = y - 190;

    % ---------------- Buttons row 1 ----------------
    btnLaunch = uibutton(leftPanel, 'push', ...
        'Position',[30 y 140 36], ...
        'Text','Launch Run', ...
        'ButtonPushedFcn', @(~,~) launchRun());

    btnShowCmd = uibutton(leftPanel, 'push', ...
        'Position',[190 y 140 36], ...
        'Text','Show Command', ...
        'ButtonPushedFcn', @(~,~) showCommand());

    btnRefresh = uibutton(leftPanel, 'push', ...
        'Position',[350 y 140 36], ...
        'Text','Refresh Monitor', ...
        'ButtonPushedFcn', @(~,~) updateMonitor());

    y = y - 52;

    % ---------------- Buttons row 2 ----------------
    btnStopAll = uibutton(leftPanel, 'push', ...
        'Position',[30 y 220 36], ...
        'Text','Stop Batch Runs', ...
        'ButtonPushedFcn', @(~,~) stopBatchRuns());

    btnFresh = uibutton(leftPanel, 'push', ...
        'Position',[270 y 220 36], ...
        'Text','Stop Batch Runs + Launch Fresh', ...
        'ButtonPushedFcn', @(~,~) stopAndLaunchFresh());

    % ===================== RIGHT: MONITOR =====================
    rightX = leftW + 20;
    rightW = figW - rightX - 10;

    monitorPanel = uipanel(fig, ...
        'Title','Run Monitor', ...
        'Position',[rightX 10 rightW figH-20]);

    topY = figH - 90;

    uilabel(monitorPanel, 'Position',[15 topY 110 22], 'Text','Latest Log');
    efLogFile = uieditfield(monitorPanel, 'text', ...
        'Position',[90 topY rightW-170 22], ...
        'Editable','off', ...
        'Value','');

    btnBrowseLog = uibutton(monitorPanel, 'push', ...
        'Position',[rightW-65 topY 40 22], ...
        'Text','...', ...
        'ButtonPushedFcn', @(~,~) browseLogFile());

    statusY = topY - 40;
    lblStatus   = uilabel(monitorPanel, 'Position',[15 statusY 250 22], 'Text','Status: idle');
    lblUpdated  = uilabel(monitorPanel, 'Position',[280 statusY 250 22], 'Text','Last update: --');

    statusY = statusY - 28;
    lblBestCost = uilabel(monitorPanel, 'Position',[15 statusY 250 22], 'Text','Best cost: --');
    lblRuntime  = uilabel(monitorPanel, 'Position',[280 statusY 250 22], 'Text','Runtime: --');

    statusY = statusY - 28;
    lblRmsePos = uilabel(monitorPanel, 'Position',[15 statusY 250 22], 'Text','RMSE pos (km): --');
    lblRmseVel = uilabel(monitorPanel, 'Position',[280 statusY 250 22], 'Text','RMSE vel (km/s): --');

    statusY = statusY - 28;
    lblDetP = uilabel(monitorPanel, 'Position',[15 statusY 450 22], 'Text','Mean det(P_pos) (km^6): --');

    txtLog = uitextarea(monitorPanel, ...
        'Position',[15 15 rightW-30 figH-270], ...
        'Editable','off', ...
        'FontName','Consolas', ...
        'Value', {''} );

    isManualLogSelection = false;

    ddOpt.ValueChangedFcn = @(~,~) toggleFields();
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
        % optimizer-specific toggles
        if strcmp(ddOpt.Value, 'BAYESIAN')
            efMaxIters.Enable = 'off';
            efMaxEvals.Enable = 'on';
        else
            efMaxIters.Enable = 'on';
            efMaxEvals.Enable = 'off';
        end

        % mission-specific toggles
        if strcmp(ddMission.Value, 'LOW_THRUST_TRANSFER')
            efNPeriods.Enable = 'off';
            lblNPeriods.Enable = 'off';
        else
            efNPeriods.Enable = 'on';
            lblNPeriods.Enable = 'on';
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
        params.MAX_ITERS     = validatePositiveInteger(efMaxIters.Value, 'MAX_ITERS');
        params.MAX_EVALS     = validatePositiveInteger(efMaxEvals.Value, 'MAX_EVALS');
        params.EKF_DT        = validatePositiveScalar(efEkfDt.Value, 'EKF_DT');
        params.SEED          = validateInteger(efSeed.Value, 'SEED');

        if strcmp(params.MISSION_TYPE, 'LOW_THRUST_TRANSFER')
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

            bestCost = parseLastMatch(txt, 'Cost:\s*([\-+0-9.eE]+)');
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

            detP = parseLastMatch(txt, 'Mean det\(P_pos\) \(km\^6\):\s*([\-+0-9.eE]+)');
            if ~isempty(detP)
                lblDetP.Text = ['Mean det(P_pos) (km^6): ' detP];
            else
                lblDetP.Text = 'Mean det(P_pos) (km^6): --';
            end

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
        lblDetP.Text     = 'Mean det(P_pos) (km^6): --';
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
    scriptPath = fullfile(pwd, 'run_opt.m');
    if ~isfile(scriptPath)
        error('Could not find run_opt.m in the current folder.');
    end

    ensureRunDir(params.RUN_DIR);

    matlabExe = 'matlab';
    launchLog = fullfile(params.RUN_DIR, ...
        ['gui_launch_' char(datetime('now','Format','yyyyMMdd_HHmmss')) '.log']);

    batchCmd = sprintf([ ...
        'setenv(''MISSION_TYPE'',''%s'');' ...
        'setenv(''OPTIMIZER_MODE'',''%s'');' ...
        'setenv(''NUM_OBSERVERS'',''%s'');' ...
        'setenv(''MAX_ITERS'',''%s'');' ...
        'setenv(''MAX_EVALS'',''%s'');' ...
        'setenv(''EKF_DT'',''%s'');' ...
        'setenv(''NPERIODS'',''%s'');' ...
        'setenv(''SEED'',''%s'');' ...
        'setenv(''RUN_DIR'',''%s'');' ...
        'setenv(''USE_SCREENING'',''%s'');' ...
        'setenv(''USE_J1'',''%s'');' ...
        'setenv(''USE_J2'',''%s'');' ...
        'setenv(''USE_J3'',''%s'');' ...
        'run(''%s'');' ], ...
        escapeSingleQuotes(params.MISSION_TYPE), ...
        escapeSingleQuotes(params.OPTIMIZER_MODE), ...
        escapeSingleQuotes(params.NUM_OBSERVERS), ...
        escapeSingleQuotes(params.MAX_ITERS), ...
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

    files1 = dir(fullfile(runDir, '**', 'safe_output_fallback.txt'));
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