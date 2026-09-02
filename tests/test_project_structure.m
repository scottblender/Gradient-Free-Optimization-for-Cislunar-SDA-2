function test_project_structure()
%TEST_PROJECT_STRUCTURE Check code paths without data or optimization toolboxes.

    projectDir = fileparts(fileparts(mfilename('fullpath')));
    originalPath = path;
    originalFolder = pwd;
    cleanup = onCleanup(@() restoreEnvironment(originalFolder, originalPath)); %#ok<NASGU>

    addpath(projectDir);
    cd(tempdir);
    externalFolder = pwd;
    paths = setup_project();

    assert(strcmp(paths.root, projectDir), 'Incorrect project root.');
    assert(strcmp(pwd, externalFolder), 'setup_project changed the current folder.');
    assert(isfolder(paths.data) && isfolder(paths.results), ...
        'Missing data/ or results/ directory.');
    assert(strcmp(paths.targetCaseDatabase, ...
        fullfile(paths.data,'TargetCaseDatabase.mat')), ...
        'Incorrect target-case database path.');

    expected = {
        'run_opt', 'run_opt.m'
        'launch_optimization_gui', 'launch_optimization_gui.m'
        'cr3bp_dynamics', 'src/orbitDynamics/cr3bp_dynamics.m'
        'cr3bp_jacobian', 'src/orbitDynamics/cr3bp_jacobian.m'
        'jacobi_constant', 'src/orbitDynamics/jacobi_constant.m'
        'sun_pos_bc4bp', 'src/orbitDynamics/sun_pos_bc4bp.m'
        'cr3bp_ekf', 'src/estimation/cr3bp_ekf.m'
        'measurement_model', 'src/measurements/measurement_model.m'
        'measurement_jacobian', 'src/measurements/measurement_jacobian.m'
        'calc_visibility', 'src/constraints/calc_visibility.m'
        'calc_occlusion', 'src/constraints/calc_occlusion.m'
        'calc_exclusion', 'src/constraints/calc_exclusion.m'
        'objective_wrapper', 'src/optimization/objective_wrapper.m'
        'compute_cost', 'src/optimization/compute_cost.m'
        'abc_discrete', 'src/optimization/abc_discrete.m'
        'aco_discrete', 'src/optimization/aco_discrete.m'
        'dmopso', 'src/optimization/dmopso.m'
        'build_target_truth', 'src/targetGeneration/build_target_truth.m'
        'build_truth_gateway', 'src/targetGeneration/build_truth_gateway.m'
        'build_truth_periodic_orbit', 'src/targetGeneration/build_truth_periodic_orbit.m'
        'LowThrustTransferSolver', 'src/targetGeneration/LowThrustTransferSolver.m'
        'load_target_case_database', 'src/targetGeneration/load_target_case_database.m'
        'target_case_config', 'src/targetGeneration/target_case_config.m'
        'build_observer_orbit_catalog', 'scripts/build_observer_orbit_catalog.m'
        'build_target_case_database', 'scripts/build_target_case_database.m'
        'plot_study_definition_figures', 'scripts/plot_study_definition_figures.m'
        'process_baseline_results', 'scripts/process_baseline_results.m'
        'process_comparison_results', 'scripts/process_comparison_results.m'
        'process_fe_convergence', 'scripts/process_fe_convergence.m'
        'print_observer_ics_from_experiment_summary', 'scripts/print_observer_ics_from_experiment_summary.m'
        'test_observer_catalog', 'tests/test_observer_catalog.m'
        'test_fe_study_configuration', 'tests/test_fe_study_configuration.m'
        'test_visibility_keepout_definition', 'tests/test_visibility_keepout_definition.m'
        'test_visibility_trajectories', 'tests/test_visibility_trajectories.m'
    };

    for i = 1:size(expected, 1)
        resolved = which(expected{i,1});
        relativeParts = strsplit(expected{i,2}, '/');
        wanted = fullfile(projectDir, relativeParts{:});
        assert(strcmp(resolved, wanted), ...
            'Wrong file resolved for %s: %s', expected{i,1}, resolved);
    end

    fprintf('Project structure passed: %d entry points resolved.\n', size(expected, 1));
end

function restoreEnvironment(originalFolder, originalPath)
    cd(originalFolder);
    path(originalPath);
end