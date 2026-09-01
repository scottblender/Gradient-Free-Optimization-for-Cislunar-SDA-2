function outputs = plot_study_definition_figures(inspectFigures)
% Generate catalog, slot-definition, and target-case figures.

if nargin<1 || isempty(inspectFigures), inspectFigures = true; end

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
setup_project();

fprintf('\n--- Study-definition figures ---\n');

fprintf('\n1/3 Orbit catalog characteristics\n');
outputs.catalog = plot_orbit_catalog_characteristics(inspectFigures);

fprintf('\n2/3 Equal-time slot definition\n');
outputs.slots = plot_slot_definition(inspectFigures);

fprintf('\n3/3 Tracking cases\n');
fprintf('The low-thrust panel solves the transfer and can take several minutes.\n');
outputs.cases = plot_tracking_cases(inspectFigures);

fprintf('\nAll study-definition figures were generated.\n');
end
