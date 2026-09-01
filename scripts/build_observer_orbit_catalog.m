% ----- build_observer_orbit_catalog.m ----- %
% Build the filtered observer-orbit catalog and enrich it with the
% manuscript-facing geometric and dynamical properties used in review.
%
% The filtering/selection core is kept private so this is the single public
% entry point for rebuilding the observer database.

projectDir = fileparts(fileparts(mfilename('fullpath')));
addpath(projectDir);
projectPaths = setup_project();

coreScript = fullfile(projectPaths.scripts,'private', ...
    'observer_catalog_filter_core.m');
assert(isfile(coreScript), ...
    'Observer catalog filtering core was not found: %s',coreScript);

% The legacy filtering core intentionally clears its workspace. Run it
% first, then reconstruct paths before enriching the saved catalog.
run(coreScript);

projectPaths = setup_project();
catalogPath = projectPaths.catalog;
S = load(catalogPath);
assert(isfield(S,'T') && istable(S.T), ...
    'Observer catalog does not contain table T.');
T = S.T;

mu = 1.215058560962404E-2;
LU = 384400; % km

nOrbit = height(T);
jacobiConstant = zeros(nOrbit,1);
jacobiVariation = zeros(nOrbit,1);
xAmplitude_LU = zeros(nOrbit,1);
zAmplitude_LU = zeros(nOrbit,1);

for k = 1:nOrbit
    state = T.state{k};
    cjHistory = jacobi_constant(state,mu);
    jacobiConstant(k) = cjHistory(1);
    jacobiVariation(k) = max(cjHistory)-min(cjHistory);

    xAmplitude_LU(k) = 0.5*(max(state(:,1))-min(state(:,1)));
    zAmplitude_LU(k) = 0.5*(max(state(:,3))-min(state(:,3)));
end

T.jacobiConstant = jacobiConstant;
T.jacobiVariation = jacobiVariation;
T.xAmplitude_LU = xAmplitude_LU;
T.zAmplitude_LU = zAmplitude_LU;
T.manuscriptFamily = manuscript_family(string(T.orbitFamily));
T.region = manuscript_region(string(T.orbitFamily));

assert(all(isfinite(T.jacobiConstant)), ...
    'One or more selected observer orbits has a nonfinite Jacobi constant.');
assert(all(isfinite(T.jacobiVariation)), ...
    'One or more selected observer orbits has a nonfinite Jacobi variation.');
assert(height(T)==450, ...
    'Expected 450 selected observer orbits, but found %d.',height(T));

% Use a conservative audit threshold: this is not part of the filtering,
% but catches propagation/data errors before manuscript properties are used.
maxJacobiVariation = max(T.jacobiVariation);
assert(maxJacobiVariation <= 1e-8, ...
    ['Jacobi constant variation exceeds the numerical audit tolerance. ' ...
     'Maximum range = %.6e.'],maxJacobiVariation);

% Preserve all variables already saved by the filtering core.
S.T = T;
save(catalogPath,'-struct','S','-v7.3');

write_observer_property_exports(T,projectPaths.data);

fprintf('Enriched observer catalog saved to:\n  %s\n',catalogPath);
fprintf('Maximum Jacobi variation over selected trajectories: %.6e\n', ...
    maxJacobiVariation);


function family = manuscript_family(orbitFamily)

family = strings(size(orbitFamily));
for k = 1:numel(orbitFamily)
    name = upper(orbitFamily(k));
    if name == "DRO"
        family(k) = "DRO";
    elseif startsWith(name,"NNRH")
        family(k) = "NNRHO";
    elseif startsWith(name,"SNRH")
        family(k) = "SNRHO";
    elseif startsWith(name,"NH")
        family(k) = "NHO";
    elseif startsWith(name,"SH")
        family(k) = "SHO";
    else
        error('Unexpected observer family: %s',name);
    end
end
end


function region = manuscript_region(orbitFamily)

region = strings(size(orbitFamily));
for k = 1:numel(orbitFamily)
    name = upper(orbitFamily(k));
    if name == "DRO"
        region(k) = "--";
    elseif endsWith(name,"L1")
        region(k) = "L1";
    elseif endsWith(name,"L2")
        region(k) = "L2";
    else
        error('Unexpected observer family region: %s',name);
    end
end
end


function write_observer_property_exports(T,outputDir)

perOrbit = table( ...
    string(T.orbitID),string(T.orbitFamily),T.manuscriptFamily,T.region, ...
    T.periluneAltitude_km,T.apoluneAltitude_km, ...
    T.xAmplitude_LU,T.zAmplitude_LU,T.jacobiConstant, ...
    T.period_TU,T.stability,T.jacobiVariation, ...
    'VariableNames',{ ...
    'orbitID','catalogFamily','family','region', ...
    'periluneAltitude_km','apoluneAltitude_km', ...
    'xAmplitude_LU','zAmplitude_LU','jacobiConstant', ...
    'period_TU','stabilityIndex','jacobiVariation'});
writetable(perOrbit,fullfile(outputDir,'ObserverOrbitCatalogProperties.csv'));

familyOrder = [ ...
    "NHO","SHO","NNRHO","SNRHO", ...
    "NHO","SHO","NNRHO","SNRHO","DRO"];
regionOrder = ["L1","L1","L1","L1","L2","L2","L2","L2","--"];

n = numel(familyOrder);
N_orb = zeros(n,1);
hPeriMin_km = zeros(n,1);
hPeriMax_km = zeros(n,1);
hApoMin_km = zeros(n,1);
hApoMax_km = zeros(n,1);
AxMin_LU = zeros(n,1);
AxMax_LU = zeros(n,1);
AzMin_LU = zeros(n,1);
AzMax_LU = zeros(n,1);
CJMin = zeros(n,1);
CJMax = zeros(n,1);
periodMin_TU = zeros(n,1);
periodMax_TU = zeros(n,1);
stabilityMin = zeros(n,1);
stabilityMax = zeros(n,1);

for k = 1:n
    use = T.manuscriptFamily==familyOrder(k) & T.region==regionOrder(k);
    N_orb(k) = nnz(use);
    assert(N_orb(k)==50, ...
        'Expected 50 %s %s orbits, found %d.', ...
        familyOrder(k),regionOrder(k),N_orb(k));

    hPeriMin_km(k) = min(T.periluneAltitude_km(use));
    hPeriMax_km(k) = max(T.periluneAltitude_km(use));
    hApoMin_km(k) = min(T.apoluneAltitude_km(use));
    hApoMax_km(k) = max(T.apoluneAltitude_km(use));
    AxMin_LU(k) = min(T.xAmplitude_LU(use));
    AxMax_LU(k) = max(T.xAmplitude_LU(use));
    AzMin_LU(k) = min(T.zAmplitude_LU(use));
    AzMax_LU(k) = max(T.zAmplitude_LU(use));
    CJMin(k) = min(T.jacobiConstant(use));
    CJMax(k) = max(T.jacobiConstant(use));
    periodMin_TU(k) = min(T.period_TU(use));
    periodMax_TU(k) = max(T.period_TU(use));
    stabilityMin(k) = min(T.stability(use));
    stabilityMax(k) = max(T.stability(use));
end

Family = familyOrder(:);
Region = regionOrder(:);

geometric = table( ...
    Family,Region,hPeriMin_km,hPeriMax_km,hApoMin_km,hApoMax_km, ...
    AxMin_LU,AxMax_LU,AzMin_LU,AzMax_LU);
writetable(geometric, ...
    fullfile(outputDir,'ObserverOrbitFamilyGeometric.csv'));

dynamical = table( ...
    Family,Region,N_orb,CJMin,CJMax,periodMin_TU,periodMax_TU, ...
    stabilityMin,stabilityMax);
writetable(dynamical, ...
    fullfile(outputDir,'ObserverOrbitFamilyDynamical.csv'));
end
