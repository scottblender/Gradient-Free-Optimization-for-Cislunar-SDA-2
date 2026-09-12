function report = run_reviewer2_baseline_monte_carlo(varargin)
%RUN_REVIEWER2_BASELINE_MONTE_CARLO Local validation of baseline GA designs.
%
% This reproduces the local Monte Carlo study used in the earlier paper:
%   * uniform integer perturbations about the optimized discrete design;
%   * each orbit index is varied by up to +/-10 orbit IDs;
%   * each slot index is varied by up to +/-5 slot IDs;
%   * 250 samples are evaluated per baseline case by default;
%   * sample 1 is exactly the optimized reference design.
%
% The no-argument default reproduces both baseline AO validation sets used
% in the paper:
%   * LOW_THRUST_TRANSFER: 3/5/7/10 observers, one trajectory each;
%   * LUNAR_GATEWAY:       3/5/7/10 observers x 1/3/5 periods.
% This gives 16 cases total (4000 objective evaluations at 250 samples/case).
% GATEWAY_IMPULSE remains available as an optional mission but is not part
% of the default Monte Carlo validation set.
%
% The reference design for each configuration is the best observed 6000-FE
% GA baseline realization among the 20 seeds. This seed-specific design is
% used only for local-optimum validation; manuscript performance comparisons
% continue to use the 20-run mean +/- sample standard deviation.
%
% Examples:
%   report = run_reviewer2_baseline_monte_carlo;
%   report = run_reviewer2_baseline_monte_carlo('Samples',100);
%   report = run_reviewer2_baseline_monte_carlo( ...
%       'Mission',"LUNAR_GATEWAY",'GatewayPeriods',[1 3 5]);
%   report = run_reviewer2_baseline_monte_carlo( ...
%       'Mission',["LOW_THRUST_TRANSFER","LUNAR_GATEWAY"]);
%
% Name-value options:
%   Mission            string/string vector; default is
%                      [LOW_THRUST_TRANSFER, LUNAR_GATEWAY].
%                      GATEWAY_IMPULSE is also supported.
%   Measurement        ANGLES_ONLY (default) or ANGLES_RANGE
%   ObserverCounts     [3 5 7 10] (default)
%   GatewayPeriods     [1 3 5] (used only for Lunar Gateway)
%   Samples            250
%   OrbitPerturbation  10
%   SlotPerturbation   5
%   Seed               260909
%   UseParallel        true
%   SaveFigures        true

p = inputParser;
addParameter(p,'Mission',["LOW_THRUST_TRANSFER","LUNAR_GATEWAY"]);
addParameter(p,'Measurement',"ANGLES_ONLY");
addParameter(p,'ObserverCounts',[3 5 7 10]);
addParameter(p,'GatewayPeriods',[1 3 5]);
addParameter(p,'Samples',250);
addParameter(p,'OrbitPerturbation',10);
addParameter(p,'SlotPerturbation',5);
addParameter(p,'Seed',260909);
addParameter(p,'UseParallel',true);
addParameter(p,'SaveFigures',true);
parse(p,varargin{:});
opts = p.Results;

opts.Mission = upper(string(opts.Mission(:)'));
opts.Measurement = upper(string(opts.Measurement));
opts.ObserverCounts = double(opts.ObserverCounts(:)');
opts.GatewayPeriods = double(opts.GatewayPeriods(:)');
opts.Samples = double(opts.Samples);
opts.OrbitPerturbation = double(opts.OrbitPerturbation);
opts.SlotPerturbation = double(opts.SlotPerturbation);
opts.Seed = double(opts.Seed);
opts.UseParallel = logical(opts.UseParallel);
opts.SaveFigures = logical(opts.SaveFigures);

supportedMissions = ["LOW_THRUST_TRANSFER","LUNAR_GATEWAY","GATEWAY_IMPULSE"];
assert(~isempty(opts.Mission) && all(ismember(opts.Mission,supportedMissions)), ...
    'Unsupported Monte Carlo mission set: %s',char(strjoin(opts.Mission,', ')));
assert(numel(unique(opts.Mission,'stable')) == numel(opts.Mission), ...
    'Monte Carlo mission list must not contain duplicates.');
assert(ismember(opts.Measurement,["ANGLES_ONLY","ANGLES_RANGE"]), ...
    'Unsupported measurement model: %s',opts.Measurement);
validateattributes(opts.ObserverCounts,{'numeric'},{'integer','positive','finite','nonempty'});
validateattributes(opts.GatewayPeriods,{'numeric'},{'integer','positive','finite','nonempty'});
validateattributes(opts.Samples,{'numeric'},{'scalar','integer','>=',2,'finite'});
validateattributes(opts.OrbitPerturbation,{'numeric'},{'scalar','integer','nonnegative','finite'});
validateattributes(opts.SlotPerturbation,{'numeric'},{'scalar','integer','nonnegative','finite'});
validateattributes(opts.Seed,{'numeric'},{'scalar','integer','nonnegative','finite'});

paths = setup_project();
baselineRoot = fullfile(paths.results,'BASELINE');
assert(isfolder(baselineRoot),'Baseline raw-result root does not exist: %s',baselineRoot);

compiledRoot = fullfile(paths.root,'COMPILED_REVIEWER_2_RESULTS');
if ~isfolder(compiledRoot), mkdir(compiledRoot); end
stamp = string(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
outDir = fullfile(compiledRoot,"baseline_monte_carlo_"+stamp);
assert(~isfolder(outDir),'Monte Carlo output directory already exists: %s',outDir);
mkdir(outDir);

fprintf('\n--- Reviewer 2 baseline local Monte Carlo validation ---\n');
fprintf('Missions:                   %s\n',char(strjoin(opts.Mission,', ')));
fprintf('Measurement:                %s\n',opts.Measurement);
fprintf('Observer counts:            %s\n',mat2str(opts.ObserverCounts));
if ismember("LUNAR_GATEWAY",opts.Mission)
    fprintf('Gateway periods:            %s\n',mat2str(opts.GatewayPeriods));
end
fprintf('Samples per case:           %d\n',opts.Samples);
fprintf('Orbit-ID perturbation:      +/- %d\n',opts.OrbitPerturbation);
fprintf('Slot-ID perturbation:       +/- %d\n',opts.SlotPerturbation);
fprintf('First sample:               optimized reference design\n');
fprintf('Output:                     %s\n\n',outDir);

references = select_reference_runs(baselineRoot,opts);
assert(~isempty(references),'No requested baseline reference cases were found.');

slotsPerOrbit = reference_slots_per_orbit(references.RunFile(1));
for k = 2:height(references)
    assert(reference_slots_per_orbit(references.RunFile(k)) == slotsPerOrbit, ...
        'Requested Monte Carlo references do not share one slots-per-orbit definition.');
end
cacheFile = fullfile(paths.orbitCache, ...
    sprintf('orbit_database_slots_%d.mat',slotsPerOrbit));
assert(isfile(cacheFile),'Missing orbit database cache: %s',cacheFile);
C = load(cacheFile,'orbit_database','cacheMeta');
orbitDatabase = C.orbit_database;
numOrbits = numel(orbitDatabase);

catalog = load(paths.catalog,'T');
stabilities = catalog_stabilities(catalog.T);
assert(numel(stabilities) == numOrbits, ...
    'Orbit cache and catalog stability vectors have different lengths.');

if opts.UseParallel
    pool = gcp('nocreate');
    if isempty(pool), parpool; end
    orbitDbArg = parallel.pool.Constant(orbitDatabase);
    stabilityArg = parallel.pool.Constant(stabilities);
else
    orbitDbArg = orbitDatabase;
    stabilityArg = stabilities;
end

sampleTables = cell(height(references),1);
summaryRows = cell(height(references),1);
for k = 1:height(references)
    fprintf('Monte Carlo case %d/%d: %s, %d observers, %d period(s)\n', ...
        k,height(references),references.Mission(k), ...
        references.NumObservers(k),references.NPeriods(k));
    [sampleTables{k},summaryRows{k}] = evaluate_case( ...
        references(k,:),orbitDbArg,stabilityArg,numOrbits,slotsPerOrbit,opts,k);
end

samples = vertcat(sampleTables{:});
summary = vertcat(summaryRows{:});
writetable(samples,fullfile(outDir,'baseline_monte_carlo_samples.csv'));
writetable(summary,fullfile(outDir,'baseline_monte_carlo_summary.csv'));

figureDetails = plot_reviewer2_baseline_monte_carlo( ...
    samples,summary,outDir,opts.SaveFigures);
writetable(figureDetails,fullfile(outDir,'baseline_monte_carlo_figures.csv'));

report = struct();
report.analysisDirectory = string(outDir);
report.options = opts;
report.references = references;
report.samples = samples;
report.summary = summary;
report.figureDetails = figureDetails;

fprintf('\nBaseline Monte Carlo validation complete.\n');
fprintf('Cases:   %d\n',height(summary));
fprintf('Samples: %d\n',height(samples));
fprintf('Output:  %s\n',outDir);
end


function references = select_reference_runs(root,opts)
files = dir(fullfile(root,'**','optimization_run.mat'));
assert(~isempty(files),'No baseline optimization_run.mat files under %s.',root);
rows = cell(0,1);
for k = 1:numel(files)
    file = fullfile(files(k).folder,files(k).name);
    try
        S = load(file,'runState'); r = S.runState; s = r.settings;
        if string(r.studyID) ~= "reviewer2_baseline_v1" || ...
                string(r.optimizer) ~= "GA" || r.maxEvaluations ~= 6000 || ...
                string(r.status) ~= "completed" || string(r.validationStatus) ~= "passed"
            continue;
        end
        mission = string(s.mission.type);
        measurement = string(s.measurements.type);
        if ~ismember(mission,opts.Mission) || measurement ~= opts.Measurement
            continue;
        end
        nObs = double(s.mission.optimization.numObservers);
        if ~ismember(nObs,opts.ObserverCounts), continue; end
        nPeriods = 1;
        if mission == "LUNAR_GATEWAY", nPeriods = double(s.mission.gateway.Nperiods); end
        if mission == "LUNAR_GATEWAY" && ~ismember(nPeriods,opts.GatewayPeriods), continue; end
        assert(s.useScreening && s.costFlags.J1 && s.costFlags.J2 && s.costFlags.J3, ...
            'Baseline reference does not use screening ON and J111.');
        rows{end+1,1} = table(string(file),mission,measurement,nObs,nPeriods, ...
            double(r.optimizerSeed),double(r.bestJ), ...
            'VariableNames',{'RunFile','Mission','Measurement','NumObservers', ...
            'NPeriods','Seed','BestObjective'});
    catch
        % Ignore incomplete/non-schema files; the selected groups are
        % explicitly checked for all 20 valid seeds below.
    end
end
assert(~isempty(rows),'No valid baseline runs match the requested Monte Carlo study.');
allRuns = vertcat(rows{:});

references = table();
for mission = opts.Mission
    for nObs = opts.ObserverCounts
        periods = 1;
        if mission == "LUNAR_GATEWAY", periods = opts.GatewayPeriods; end
        for nPeriods = periods
            group = allRuns(allRuns.Mission == mission & ...
                allRuns.NumObservers == nObs & allRuns.NPeriods == nPeriods,:);
            assert(height(group) == 20 && numel(unique(group.Seed)) == 20, ...
                'Expected 20 baseline seeds for %s/o%d/p%d; found %d.', ...
                mission,nObs,nPeriods,height(group));
            [~,idx] = min(group.BestObjective);
            references = [references;group(idx,:)];
        end
    end
end
end


function n = reference_slots_per_orbit(runFile)
S = load(runFile,'runState');
n = double(S.runState.settings.slotsPerOrbit);
validateattributes(n,{'numeric'},{'scalar','integer','positive','finite'});
end


function stabilities = catalog_stabilities(T)
names = string(T.Properties.VariableNames);
idx = find(contains(lower(names),'stability'),1);
assert(~isempty(idx),'Observer catalog does not contain a stability-index column.');
stabilities = double(T.(T.Properties.VariableNames{idx}));
stabilities = stabilities(:);
assert(all(isfinite(stabilities)),'Catalog stability vector contains nonfinite values.');
end


function [sampleTable,summaryRow] = evaluate_case( ...
    reference,orbitDbArg,stabilityArg,numOrbits,slotsPerOrbit,opts,caseIndex)
runFile = string(reference.RunFile);
trackingFile = string(fullfile(fileparts(runFile),'tracking_data.mat'));
assert(isfile(trackingFile),'Missing reference tracking data: %s',trackingFile);
S = load(runFile,'runState'); T = load(trackingFile,'tracking');
r = S.runState; s = r.settings; tracking = T.tracking;

x0 = round(double(r.bestX(:)'));
assert(numel(x0) == 2*reference.NumObservers, ...
    'Reference decision vector does not match observer count.');

designs = generate_local_designs(x0,opts.Samples,numOrbits,slotsPerOrbit, ...
    opts.OrbitPerturbation,opts.SlotPerturbation,opts.Seed+caseIndex-1);

sunFcn = @(t) sun_pos_bc4bp(t,s.LU,s.TU,s.theta0,s.i_sun);
objective = @(x) objective_wrapper(x,orbitDbArg,stabilityArg, ...
    tracking.truth,tracking.t_TU,s.P0,s.Q,s.R,s.mu,s.LU, ...
    sunFcn,s.sun_exclusion,s.moon_exclusion,s.earth_exclusion, ...
    'SOO',"MONTE_CARLO",[],s.useScreening,s.costFlags,s.cost,s.measurements);

cost = nan(opts.Samples,1);
if opts.UseParallel
    parfor j = 1:opts.Samples
        cost(j) = objective(designs(j,:));
    end
else
    for j = 1:opts.Samples
        cost(j) = objective(designs(j,:));
    end
end
assert(all(isfinite(cost)),'Monte Carlo produced a nonfinite objective value.');

tol = 1e-8*max(1,abs(r.bestJ));
assert(abs(cost(1)-double(r.bestJ)) <= tol, ...
    ['Reference Monte Carlo sample does not reproduce the saved optimized ' ...
     'objective. Saved %.12g, reproduced %.12g.'],double(r.bestJ),cost(1));

sampleNumber = (1:opts.Samples)';
isReference = sampleNumber == 1;
design = strings(opts.Samples,1);
for j = 1:opts.Samples
    design(j) = strjoin(string(designs(j,:)),',');
end
sampleTable = table( ...
    repmat(string(reference.Mission),opts.Samples,1), ...
    repmat(string(reference.Measurement),opts.Samples,1), ...
    repmat(double(reference.NumObservers),opts.Samples,1), ...
    repmat(double(reference.NPeriods),opts.Samples,1), ...
    repmat(double(reference.Seed),opts.Samples,1),sampleNumber,isReference, ...
    cost,design, ...
    'VariableNames',{'Mission','Measurement','NumObservers','NPeriods', ...
    'ReferenceSeed','Sample','IsReference','TotalCost','Design'});

neighbors = cost(2:end);
referenceCost = cost(1);
minimumNeighborCost = min(neighbors);
medianCost = median(cost);
q25 = prctile(cost,25); q75 = prctile(cost,75);
fractionAtOrAboveReference = mean(neighbors >= referenceCost-tol);
numImprovedNeighbors = nnz(neighbors < referenceCost-tol);
localMinimumPass = numImprovedNeighbors == 0;
summaryRow = table(string(reference.Mission),string(reference.Measurement), ...
    double(reference.NumObservers),double(reference.NPeriods), ...
    double(reference.Seed),opts.Samples,opts.OrbitPerturbation, ...
    opts.SlotPerturbation,referenceCost,minimumNeighborCost,medianCost,q25,q75, ...
    fractionAtOrAboveReference,numImprovedNeighbors,localMinimumPass, ...
    'VariableNames',{'Mission','Measurement','NumObservers','NPeriods', ...
    'ReferenceSeed','Samples','OrbitPerturbation','SlotPerturbation', ...
    'ReferenceObjective','MinimumNeighborObjective','MedianObjective', ...
    'Q25Objective','Q75Objective','FractionNeighborsAtOrAboveReference', ...
    'ImprovedNeighborCount','StrictLocalMinimumPass'});
end


function designs = generate_local_designs( ...
    x0,nSamples,numOrbits,slotsPerOrbit,orbitRadius,slotRadius,seed)
numObs = numel(x0)/2;
assert(numObs == round(numObs),'Decision vector must contain orbit/slot pairs.');
designs = repmat(x0,nSamples,1);
if nSamples == 1, return; end
stream = RandStream('mt19937ar','Seed',seed);
orbitDelta = randi(stream,[-orbitRadius orbitRadius],nSamples-1,numObs);
slotDelta = randi(stream,[-slotRadius slotRadius],nSamples-1,numObs);
for j = 2:nSamples
    orbit = x0(1:2:end) + orbitDelta(j-1,:);
    slot = x0(2:2:end) + slotDelta(j-1,:);
    orbit = max(1,min(numOrbits,orbit));
    slot = max(1,min(slotsPerOrbit,slot));
    designs(j,1:2:end) = orbit;
    designs(j,2:2:end) = slot;
end
% Sample 1 remains exactly x0 by construction.
assert(isequal(designs(1,:),x0));
end
