function outputs = plot_reviewer2_figures()
% Generate the pre-study figures added in response to Reviewer 2.

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
setup_project();

fprintf('\n--- Reviewer 2 pre-study figures ---\n');

fprintf('\n1/3 Orbit catalog characteristics\n');
outputs.catalog = plot_orbit_catalog_characteristics();

fprintf('\n2/3 Equal-time slot definition\n');
outputs.slots = plot_slot_definition();

fprintf('\n3/3 Tracking cases\n');
fprintf('The low-thrust panel solves the transfer and can take several minutes.\n');
outputs.cases = plot_tracking_cases();

fprintf('\nAll Reviewer 2 pre-study figures were generated.\n');
end
