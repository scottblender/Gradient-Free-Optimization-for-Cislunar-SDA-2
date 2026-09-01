function caseDatabase = load_target_case_database(databasePath)
%LOAD_TARGET_CASE_DATABASE Load the fixed target study-case definitions.

if nargin < 1 || strlength(string(databasePath)) == 0
    projectDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(projectDir);
    projectPaths = setup_project();
    databasePath = projectPaths.targetCaseDatabase;
end

databasePath = char(databasePath);
assert(isfile(databasePath), ...
    ['Target-case database was not found: %s\n' ...
     'Run scripts/build_target_case_database.m first.'],databasePath);

S = load(databasePath,'caseDatabase');
assert(isfield(S,'caseDatabase') && isstruct(S.caseDatabase), ...
    'Target-case database does not contain caseDatabase.');
caseDatabase = S.caseDatabase;

required = {'gateway','lowThrust','gatewayImpulse','constants'};
for k = 1:numel(required)
    assert(isfield(caseDatabase,required{k}), ...
        'Target-case database is missing %s.',required{k});
end
end
