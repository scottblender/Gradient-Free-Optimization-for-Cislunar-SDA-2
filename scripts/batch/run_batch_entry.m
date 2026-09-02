%RUN_BATCH_ENTRY Stable MATLAB entry point for PowerShell batch studies.
% Environment variables RUN_DIR and PROJECT_ROOT are populated by the
% PowerShell launchers before this script is invoked.

try
    runDir = getenv('RUN_DIR');
    projectRoot = getenv('PROJECT_ROOT');

    assert(~isempty(runDir), 'RUN_DIR environment variable is not set.');
    assert(~isempty(projectRoot), 'PROJECT_ROOT environment variable is not set.');

    cd(runDir);
    addpath(projectRoot);
    setup_project;
    run(fullfile(projectRoot, 'run_opt.m'));

    % Explicitly release process workers before -batch tears down MATLAB.
    p = gcp('nocreate');
    if ~isempty(p)
        fprintf('Shutting down parallel pool before MATLAB exit...\n');
        delete(p);
        fprintf('Parallel pool shut down successfully.\n');
    end
catch ME
    % Best-effort pool cleanup also applies when run_opt throws normally.
    try
        p = gcp('nocreate');
        if ~isempty(p)
            delete(p);
        end
    catch
    end

    disp(getReport(ME, 'extended'));
    rethrow(ME);
end
