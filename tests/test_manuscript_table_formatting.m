function test_manuscript_table_formatting()
% Synthetic runtime summaries verify row order, SD formatting, and call range.
setup_project(); folder=tempname; mkdir(folder);
cleanup=onCleanup(@() rmdir(folder,'s'));
Optimizer=["GA";"PSO";"BAYESIAN";"ABC";"ACO"];
NRuns=20*ones(5,1); SearchFE=1200*ones(5,1);
BestJMean=(1:5)'; BestJStd=0.25*ones(5,1);
BudgetRuntimeMean_s=10*ones(5,1); BudgetRuntimeStd_s=ones(5,1);
R=table(Optimizer,NRuns,SearchFE,BestJMean,BestJStd,BudgetRuntimeMean_s,BudgetRuntimeStd_s);
writetable(R,fullfile(folder,'runtime_comparison_1200_results.csv'));
optimizer=repelem(Optimizer,20); seed=repmat((0:19)',5,1);
search_fe=1200*ones(100,1); solver_calls=search_fe+mod(seed,2);
M=table(optimizer,seed,search_fe,solver_calls);
writetable(M,fullfile(folder,'final_run_metrics.csv'));
[~,output]=evalc('print_manuscript_tables(''CompiledRoot'',folder,''OutputDirectory'',folder)');
assert(isfile(output.textFile) && isfile(output.latexFile));
assert(contains(fileread(output.latexFile),'% TABLE 10'));
rows=output.tables.j111_cost_runtime_tradeoff;
assert(isequal(rows(:,1),["GA";"PSO";"BO";"ABCO";"ACO"]));
assert(rows(1,3)=="$[1200,\;1201]$");
assert(rows(1,4)=="$1 \pm 0.25$");
assert(isfield(output.tables,'optimization_ekf_parameters'));
R.NRuns(1)=19; writetable(R,fullfile(folder,'runtime_comparison_1200_results.csv'));
[~,output]=evalc('print_manuscript_tables(''CompiledRoot'',folder,''OutputDirectory'',folder)');
assert(ismember("j111_cost_runtime_tradeoff",output.missing));
fprintf('Manuscript table formatting checks passed.\n');
end
