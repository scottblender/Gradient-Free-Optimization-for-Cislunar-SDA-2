function output = print_manuscript_tables(varargin)
%PRINT_MANUSCRIPT_TABLES Print all 13 tables in manuscript order.
% output = print_manuscript_tables;
% output = print_manuscript_tables('CompiledRoot',folder);
% output = print_manuscript_tables('Directories',struct('baseline',folder));
% No optimizations or propagation are performed. With Reprocess=true,
% existing result processors also create their historical hidden previews.
% The four fixed configuration tables are printed from the manuscript
% template, explicitly distinguished from values read from saved results.
% Dynamic tables also print copyable LaTeX rows. Missing inputs are reported
% table by table; a missing table never silently becomes a zero or stale value.
p = inputParser;
paths = setup_project();
addParameter(p,'CompiledRoot',fullfile(paths.root,'COMPILED_REVIEWER_2_RESULTS'));
addParameter(p,'Directories',struct());
addParameter(p,'Reprocess',false);
addParameter(p,'OutputDirectory',fullfile(paths.root,'MANUSCRIPT_OUTPUT'));
addParameter(p,'CaptureOutput',true);
parse(p,varargin{:}); opts = p.Results;
if opts.CaptureOutput
    if ~isfolder(opts.OutputDirectory), mkdir(opts.OutputDirectory); end
    transcript = evalc('output = print_manuscript_tables(varargin{:},''CaptureOutput'',false);');
    fprintf('%s',transcript);
    txt = fullfile(opts.OutputDirectory,'manuscript_tables.txt');
    tex = fullfile(opts.OutputDirectory,'manuscript_tables.tex');
    write_text(txt,transcript);
    % Keep complete fixed tables and dynamic rows as LaTeX. Other messages
    % become comments, so source paths/diagnostics cannot break compilation.
    lines = splitlines(string(transcript)); inTable = false;
    for line = 1:numel(lines)
        if startsWith(strtrim(lines(line)),"\begin{table}"), inTable = true; end
        keep = inTable || (contains(lines(line)," & ") && endsWith(strtrim(lines(line)),"\\"));
        if ~keep, lines(line) = "% " + lines(line); end
        if startsWith(strtrim(lines(line)),"\end{table}"), inTable = false; end
    end
    write_text(tex,strjoin(lines,newline));
    output.textFile = string(txt); output.latexFile = string(tex);
    fprintf('Table printouts saved in: %s\n',opts.OutputDirectory);
    return;
end
if opts.Reprocess
    reports = run_reviewer2_results("all",false);
    for field = ["runtime","comparison","baseline","objective_screening"]
        opts.Directories.(field) = reports.(field).analysisDirectory;
    end
end
labels = ["orbit_database_ranges","orbit_database_geometric", ...
    "cost_weights_thresholds","gradient_free_algorithms", ...
    "optimization_ekf_parameters","target_ic_summary","comparison_cases", ...
    "baseline_summary_ao","baseline_summary_ar","j111_cost_runtime_tradeoff", ...
    "comparison_summary_ao_j111","screening_events_only","cost_component_metric_winners"];
output = struct(); output.missing = strings(0,1); output.tables = struct();
fixed = fileread(fullfile(paths.scripts,'templates','manuscript_configuration_tables.tex'));
blocks = regexp(fixed,'\\begin\{table\}.*?\\end\{table\}','match');
for k = 1:numel(labels)
    label = labels(k);
    fprintf('\nTABLE %d -- tab:%s\n',k,label);
    try
        idx = find(contains(string(blocks),"tab:"+label+"}"));
        if ~isempty(idx)
            fprintf('Source: fixed manuscript configuration (not measured results).\n');
            fprintf('%s\n',blocks{idx});
            output.tables.(label) = string(blocks{idx});
            continue;
        end
        switch label
            case {"orbit_database_ranges","orbit_database_geometric"}
                [ranges,geometry] = catalog_tables(paths.catalog);
                if label=="orbit_database_ranges", rows = ranges; else, rows = geometry; end
                fprintf('Source: %s\n',paths.catalog);
            case "target_ic_summary"
                rows = target_table(paths,opts);
            case {"baseline_summary_ao","baseline_summary_ar"}
                folder = analysis_directory(opts,'baseline','baseline_6000_results.csv');
                R = read_results(folder,'baseline_6000_results.csv');
                meas = "ANGLES_ONLY";
                if label=="baseline_summary_ar", meas = "ANGLES_RANGE"; end
                rows = strings(0,6);
                for mission = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"]
                    periods = 1; if mission=="LUNAR_GATEWAY", periods = [1 3 5]; end
                    for period = periods
                        for count = [3 5 7 10]
                            r = one(R,R.Mission==mission & R.Measurement==meas & ...
                                R.NPeriods==period & R.NumObservers==count);
                            per = string(period); if mission~="LUNAR_GATEWAY", per = "--"; end
                            rows(end+1,:) = [mission_code(mission),string(count),per, ...
                                metric(r,'BestJ'),metric(r,'RMSEPos','_km'), ...
                                metric(r,'EffectiveSigmaPos','_km')]; %#ok<AGROW>
                        end
                    end
                end
            case "j111_cost_runtime_tradeoff"
                folder = analysis_directory(opts,'runtime','runtime_comparison_1200_results.csv');
                R = read_results(folder,'runtime_comparison_1200_results.csv');
                M = read_results(folder,'final_run_metrics.csv'); rows = strings(0,5);
                for optimizer = ["GA","PSO","BAYESIAN","ABC","ACO"]
                    r = one(R,R.Optimizer==optimizer);
                    m = M(M.optimizer==optimizer,:); check_calls(m,1200);
                    rows(end+1,:) = [optimizer_label(optimizer),"1200", ...
                        range_text(m.solver_calls),metric(r,'BestJ'),metric(r,'BudgetRuntime','_s')]; %#ok<AGROW>
                end
            case "comparison_summary_ao_j111"
                folder = analysis_directory(opts,'comparison','comparison_6000_results.csv');
                R = read_results(folder,'comparison_6000_results.csv');
                M = read_results(folder,'final_run_metrics.csv'); rows = strings(0,7);
                for mission = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"]
                    for optimizer = ["GA","PSO","ABC","ACO"]
                        r = one(R,R.Mission==mission & R.Optimizer==optimizer);
                        % Match the group key; never pool different target cases.
                        keys = string(M.comparison_key(M.optimizer==optimizer));
                        group = M([],:);
                        for key = unique(keys(:)')
                            candidate = M(M.comparison_key==key & M.optimizer==optimizer,:);
                            saved = load(char(candidate.run_file(1)),'runState');
                            if string(saved.runState.settings.mission.type)==mission
                                assert(isempty(group),'Multiple comparison groups for %s/%s.',mission,optimizer);
                                group = candidate;
                            end
                        end
                        check_calls(group,6000);
                        rows(end+1,:) = [mission_code(mission),optimizer_label(optimizer), ...
                            range_text(group.solver_calls),metric(r,'BestJ'), ...
                            metric(r,'RMSEPos','_km'),metric(r,'EffectiveSigmaPos','_km'), ...
                            metric(r,'MeanStability')]; %#ok<AGROW>
                    end
                end
            case {"screening_events_only","cost_component_metric_winners"}
                folder = analysis_directory(opts,'objective_screening','ga_objective_screening_results.csv');
                R = read_results(folder,'ga_objective_screening_results.csv');
                if label=="screening_events_only"
                    configs = ["combined_on","combined_off"]; names = ["ON","OFF"]; rows = strings(0,6);
                else
                    configs = ["combined_on","j1_only","j2_only","j3_only"];
                    names = ["$J_{111}$","$J_{100}$","$J_{010}$","$J_{001}$"]; rows = strings(0,5);
                end
                for mission = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"]
                    for j = 1:numel(configs)
                        r = one(R,R.Mission==mission & string(R.Configuration)==configs(j));
                        row = [mission_code(mission),names(j),metric(r,'RMSEPos','_km'), ...
                            metric(r,'EffectiveSigmaPos','_km'),metric(r,'MeanStability')];
                        if label=="screening_events_only", row(end+1) = metric(r,'Screening'); end
                        rows(end+1,:) = row; %#ok<AGROW>
                    end
                end
        end
        output.tables.(label) = rows;
        for j = 1:size(rows,1)
            if label=="target_ic_summary" && j>6
                fprintf('\\multicolumn{2}{@{}l}{%s} & \\multicolumn{3}{l@{}}{%s} %s\n', ...
                    rows(j,1),rows(j,2),'\\');
            else
                fprintf('%s %s\n',strjoin(rows(j,:),' & '),'\\');
            end
        end
    catch exception
        output.missing(end+1) = label;
        fprintf(2,'UNAVAILABLE: %s\n',exception.message);
    end
end
fprintf('\nPrinted %d of %d tables.\n',numel(labels)-numel(output.missing),numel(labels));
if ~isempty(output.missing), fprintf('Missing tables: %s\n',strjoin(output.missing,', ')); end
end

function R = read_results(folder,name)
file = fullfile(folder,name); assert(isfile(file),'Missing %s. Run the corresponding result processor first.',file);
fprintf('Source: %s\n',file);
R = readtable(file,'TextType','string','VariableNamingRule','preserve');
end

function folder = analysis_directory(opts,field,file)
if isfield(opts.Directories,field)
    folder = string(opts.Directories.(field));
    assert(isfile(fullfile(folder,file)),'Missing %s in %s.',file,folder); return;
end
files = dir(fullfile(opts.CompiledRoot,'**',file));
assert(~isempty(files),'No %s under %s. Run run_manuscript_figures("results") first.',file,opts.CompiledRoot);
[~,idx] = max([files.datenum]); folder = string(files(idx).folder);
end

function r = one(R,mask)
r = R(mask,:); assert(height(r)==1,'Expected one matched result group, found %d.',height(r));
assert(r.NRuns==20,'Expected 20 independent runs; found %d.',r.NRuns);
end

function textValue = metric(r,prefix,suffix)
if nargin<3, suffix=''; end
mu = r.([prefix 'Mean' suffix]); sigma = r.([prefix 'Std' suffix]);
assert(isfinite(mu) && isfinite(sigma),'Nonfinite summary metric %s.',prefix);
textValue = string(sprintf('$%.6g \\pm %.3g$',mu,sigma));
end

function check_calls(M,budget)
assert(height(M)==20 && numel(unique(M.seed))==20,'Expected 20 unique optimizer seeds.');
assert(all(M.search_fe==budget),'Unexpected admitted FE counts.');
assert(all(isfinite(M.solver_calls)) && all(M.solver_calls>=budget),'Invalid solver-call counts.');
end

function value = range_text(x)
assert(all(isfinite(x)) && ~isempty(x),'Missing/nonfinite range data.');
value = string(sprintf('$[%.10g,\\;%.10g]$',min(x),max(x)));
end

function label = mission_code(mission)
labels = ["LG","LT","GI"]; keys = ["LUNAR_GATEWAY","LOW_THRUST_TRANSFER","GATEWAY_IMPULSE"];
label = labels(keys==mission); assert(isscalar(label),'Unknown target case.');
end

function label = optimizer_label(optimizer)
label = string(optimizer); if label=="BAYESIAN", label="BO"; elseif label=="ABC", label="ABCO"; end
end

function [ranges,geometry] = catalog_tables(file)
assert(isfile(file),'Missing observer catalog: %s',file); S=load(file,'T'); T=S.T;
names = strtrim(string(T.Properties.VariableNames));
period = T{:,names=="Period (TU)"}; stability = T{:,names=="Stability index"};
assert(size(period,2)==1 && size(stability,2)==1,'Missing catalog period/stability columns.');
state = T.state; jacobi = zeros(height(T),1); geom = zeros(height(T),4);
for k = 1:height(T)
    s = state{k}; mu=1.215058560962404e-2; LU=384400;
    r = s(:,1:3); d = vecnorm(r-[1-mu 0 0],2,2)*LU-1737.1;
    span = max(r,[],1)-min(r,[],1);
    geom(k,:) = [min(d),max(d),0.5*hypot(span(1),span(2))*LU,0.5*span(3)*LU];
    q=s(1,:); r1=norm(q(1:3)+[mu 0 0]); r2=norm(q(1:3)-[1-mu 0 0]);
    jacobi(k)=q(1)^2+q(2)^2+2*(1-mu)/r1+2*mu/r2-sum(q(4:6).^2);
end
keys=["NHL1","SHL1","NNRHL1","SNRHL1","NHL2","SHL2","NNRHL2","SNRHL2","DRO"];
labels=["NHO","SHO","NNRHO","SNRHO","NHO","SHO","NNRHO","SNRHO","DRO"];
ranges=strings(10,6); geometry=strings(9,6);
for k=1:9
    use=string(T.orbitFamily)==keys(k); assert(any(use),'Missing family %s.',keys(k));
    region="$L_1$"; if k>4, region="$L_2$"; end; if k==9, region="--"; end
    ranges(k,:)=[labels(k),region,string(nnz(use)),range_text(jacobi(use)),range_text(period(use)),range_text(stability(use))];
    geometry(k,:)=[labels(k),region,range_text(geom(use,1)),range_text(geom(use,2)),range_text(geom(use,3)),range_text(geom(use,4))];
end
ranges(10,:)=["Total","--",string(height(T)),"--","--","--"];
end

function rows = target_table(paths,opts)
assert(isfile(paths.targetCaseDatabase),'Missing TargetCaseDatabase.mat; build the fixed target database first.');
S=load(paths.targetCaseDatabase,'caseDatabase'); C=S.caseDatabase;
fprintf('Source: %s\n',paths.targetCaseDatabase);
states=[C.gateway.state0,C.lowThrust.departureState,C.lowThrust.arrivalState,C.gatewayImpulse.postBurnState];
rows=[ ["$x$";"$y$";"$z$";"$v_x$";"$v_y$";"$v_z$"],compose('$%.10f$',states) ];
q=C.lowThrust.lowthrust;
values={q.m0,q.Tmax,q.sigma,q.ve,q.lambda_guess,q.lambda_lb,q.lambda_ub,q.tf_guess,[q.tf_lb q.tf_ub], ...
    [q.w_pos_indirect q.w_vel_indirect q.w_norm_indirect q.w_mass_indirect]};
names=["Initial mass","Maximum thrust parameter","Throttle","Exhaust velocity (LU/TU)", ...
    "Supplied costate guess","Costate lower limits","Costate upper limits", ...
    "Final-time guess (TU)","Final-time interval (TU)","Boundary-residual weights"];
for k=1:numel(values), rows(end+1,:)=[names(k),string(mat2str(values{k},16)),"","",""]; end
rows(end+1,:)=["Shooting solver","fsolve, Levenberg--Marquardt","","",""];
rows(end+1,:)=["ODE relative / absolute tolerance","1e-13 / 1e-13","","",""];
% Read the actual converged transfer from a saved optimization run. Never
% solve a new transfer merely to fill in the reproduction table.
try
    folder=analysis_directory(opts,'baseline','baseline_6000_results.csv');
    R=read_results(folder,'baseline_6000_results.csv');
    M=read_results(folder,'final_run_metrics.csv');
    keys=R.ComparisonKey(R.Mission=="LOW_THRUST_TRANSFER");
    candidates=M(ismember(M.comparison_key,keys),:);
    assert(~isempty(candidates),'No saved LT run found.');
    S=load(char(candidates.run_file(1)),'runState'); info=S.runState.truthInfo;
    assert(isfield(info,'tf') && isfield(info,'lambda0'),'Saved LT solution lacks tf/lambda0.');
    rows(end+1,:)=["Converged duration (TU)",string(sprintf('%.16g',info.tf)),"","",""];
    rows(end+1,:)=["Converged initial costates",string(mat2str(info.lambda0,16)),"","",""];
    fprintf('Converged LT source: %s\n',candidates.run_file(1));
catch exception
    fprintf(2,'Converged LT values unavailable: %s\n',exception.message);
    rows(end+1,:)=["Converged duration / costates","UNAVAILABLE (see diagnostic)","","",""];
end
end

function write_text(file,content)
fid = fopen(file,'w'); assert(fid>=0,'Cannot write %s.',file);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s',content);
end
