clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
catalogPath = fullfile(projectRoot,"data", ...
    "JPL_CR3BP_OrbitCatalog.mat");
referencePath = fullfile(projectRoot,"data", ...
    "transfer_reference.mat");

S = load(catalogPath,"T");
T = S.T;
periods = T.("Period (TU) ");
tableNames = string(T.Properties.VariableNames);

% A catalog containing orbitID has already been rebuilt. In that case this
% script validates the saved legacy reference and must not overwrite it
% using obsolete row numbers.
if ismember("orbitID",tableNames)
    assert(isfile(referencePath), ...
        "The catalog is already rebuilt, but transfer_reference.mat is missing.");

    R = load(referencePath,"transferRef");
    transferRef = R.transferRef;
    catalogIds = string(T.orbitID);

    depIndex = find(catalogIds==string(transferRef.dep.orbitID));
    arrIndex = find(catalogIds==string(transferRef.arr.orbitID));

    assert(numel(depIndex)==1, ...
        "The saved departure orbitID did not resolve uniquely.");
    assert(numel(arrIndex)==1, ...
        "The saved arrival orbitID did not resolve uniquely.");

    depStateError = norm( ...
        T.state{depIndex}(1,:)-transferRef.dep.state0);
    arrStateError = norm( ...
        T.state{arrIndex}(1,:)-transferRef.arr.state0);
    depPeriodError = abs(periods(depIndex)-transferRef.dep.period);
    arrPeriodError = abs(periods(arrIndex)-transferRef.arr.period);

    fprintf("Existing transfer reference resolved in the rebuilt catalog.\n\n");
    fprintf("Departure: row %d, slot %d, %s\n", ...
        depIndex,transferRef.dep.slot,transferRef.dep.orbitID);
    fprintf("  Initial-state error: %.6e\n",depStateError);
    fprintf("  Period error:        %.6e TU\n",depPeriodError);
    fprintf("Arrival:   row %d, slot %d, %s\n", ...
        arrIndex,transferRef.arr.slot,transferRef.arr.orbitID);
    fprintf("  Initial-state error: %.6e\n",arrStateError);
    fprintf("  Period error:        %.6e TU\n",arrPeriodError);

    assert(depStateError<=1e-12 && arrStateError<=1e-12, ...
        "A transfer-reference orbit state changed during catalog rebuilding.");
    assert(depPeriodError<=1e-12 && arrPeriodError<=1e-12, ...
        "A transfer-reference orbit period changed during catalog rebuilding.");

    fprintf("\nReference validation passed. No file was overwritten.\n");
    return;
end

% Legacy capture mode. These indices refer only to the catalog that was
% used by the original study, before filtering/reordering.
depIndex = 51;
arrIndex = 400;

trimmedNames = strtrim(tableNames);
idMatch = find(trimmedNames=="Id");

if isempty(idMatch)
    idMatch = find(strcmpi(trimmedNames,"id"));
end

assert(numel(idMatch)==1, ...
    'Expected exactly one source Id column. Found: %s', ...
    strjoin(tableNames(idMatch),', '));

sourceIds = strtrim(string(T.(char(tableNames(idMatch)))));

assert(ismember("sourceFile",tableNames), ...
    "The legacy catalog does not contain sourceFile.");

sourceStem = erase(lower(strtrim(string(T.sourceFile))),".csv");
catalogIds = sourceStem+":"+sourceIds;

assert(numel(unique(catalogIds))==height(T), ...
    "The composite sourceFile + Id identifiers are not unique.");

transferRef = struct();

transferRef.dep.legacyIndex = depIndex;
transferRef.dep.slot = 10;
transferRef.dep.state0 = T.state{depIndex}(1,:);
transferRef.dep.period = periods(depIndex);
transferRef.dep.family = T.orbitFamily(depIndex);
transferRef.dep.orbitID = catalogIds(depIndex);

transferRef.arr.legacyIndex = arrIndex;
transferRef.arr.slot = 1;
transferRef.arr.state0 = T.state{arrIndex}(1,:);
transferRef.arr.period = periods(arrIndex);
transferRef.arr.family = T.orbitFamily(arrIndex);
transferRef.arr.orbitID = catalogIds(arrIndex);

save(referencePath,"transferRef");

fprintf("Saved legacy transfer reference to:\n  %s\n",referencePath);
fprintf("Departure: legacy orbit %d, slot %d, %s\n", ...
    depIndex,transferRef.dep.slot,transferRef.dep.orbitID);
fprintf("Arrival:   legacy orbit %d, slot %d, %s\n", ...
    arrIndex,transferRef.arr.slot,transferRef.arr.orbitID);
