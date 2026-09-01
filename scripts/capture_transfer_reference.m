clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
catalogPath = fullfile(projectRoot, "data", ...
    "JPL_CR3BP_OrbitCatalog.mat");
referencePath = fullfile(projectRoot, "data", ...
    "transfer_reference.mat");

S = load(catalogPath, "T");
T = S.T;

depIndex = 51;
arrIndex = 400;

periods = T.("Period (TU) ");

assert(ismember("Id", string(T.Properties.VariableNames)), ...
    "The catalog does not contain the JPL Id column.");

catalogIds = string(T.Id);
assert(numel(unique(catalogIds)) == height(T), ...
    "The catalog contains duplicate Id values.");

transferRef = struct();

transferRef.dep.legacyIndex = depIndex;
transferRef.dep.slot        = 10;
transferRef.dep.state0      = T.state{depIndex}(1,:);
transferRef.dep.period      = periods(depIndex);
transferRef.dep.family      = T.orbitFamily(depIndex);
transferRef.dep.Id          = catalogIds(depIndex);

transferRef.arr.legacyIndex = arrIndex;
transferRef.arr.slot        = 1;
transferRef.arr.state0      = T.state{arrIndex}(1,:);
transferRef.arr.period      = periods(arrIndex);
transferRef.arr.family      = T.orbitFamily(arrIndex);
transferRef.arr.Id          = catalogIds(arrIndex);

save(referencePath, "transferRef");

fprintf("Saved transfer reference to:\n  %s\n", referencePath);
fprintf("Departure: orbit %d, slot %d\n", ...
    depIndex, transferRef.dep.slot);
fprintf("Arrival:   orbit %d, slot %d\n", ...
    arrIndex, transferRef.arr.slot);