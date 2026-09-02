function paths = setup_project()
%SETUP_PROJECT Add project code to the MATLAB path and resolve data locations.
% Run from the project root, or add the root to the path before calling.
% Existing function names are unchanged; src uses ordinary folders.

    paths.root = fileparts(mfilename('fullpath'));
    paths.src = fullfile(paths.root, 'src');
    paths.scripts = fullfile(paths.root, 'scripts');
    paths.tests = fullfile(paths.root, 'tests');
    paths.data = fullfile(paths.root, 'data');
    paths.results = fullfile(paths.root, 'results');
    % Compatibility alias used by existing study/processing code. Study
    % folders now live directly under results/ instead of results/runs/.
    paths.runs = paths.results;
    paths.orbitCache = fullfile(paths.data, 'cache', 'orbits');
    paths.transferCache = fullfile(paths.data, 'cache', 'transfers');
    paths.targetCaseDatabase = fullfile(paths.data, 'TargetCaseDatabase.mat');

    addpath(paths.root);
    addpath(genpath(paths.src));
    addpath(paths.scripts);
    addpath(paths.tests);

    % Prefer data/, but keep existing local catalogs usable after git pull.
    catalogName = 'JPL_CR3BP_OrbitCatalog.mat';
    paths.catalog = fullfile(paths.data, catalogName);
    legacyCatalog = fullfile(paths.root, catalogName);
    if ~isfile(paths.catalog) && isfile(legacyCatalog)
        paths.catalog = legacyCatalog;
    end

    paths.rawData = fullfile(paths.data, 'JPL_Data');
    legacyRawData = fullfile(paths.root, 'JPL_Data');
    if ~isfolder(paths.rawData) && isfolder(legacyRawData)
        paths.rawData = legacyRawData;
    end
end
