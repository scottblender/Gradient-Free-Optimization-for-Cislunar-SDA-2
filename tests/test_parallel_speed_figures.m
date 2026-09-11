function test_parallel_speed_figures()
% Synthetic saved histories verify both exports without running an optimizer.
setup_project(); folder=tempname; mkdir(folder);
cleanup=onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
Mode=["Serial";"Parallel";"Serial";"Parallel"];
Repeat=[1;1;2;2]; SearchFE=6000*ones(4,1);
results=table(Mode,Repeat,SearchFE);
histories=cell(4,1);
for k=1:4
    fe=(60:60:6000)'; bestJ=2+1./sqrt(fe/60);
    elapsed_s=fe/10; if Mode(k)=="Parallel", elapsed_s=elapsed_s/3; end
    histories{k}=table(fe,bestJ,elapsed_s);
end
benchmark=struct('mission',"LUNAR_GATEWAY",'budget',6000,'nRepeats',2, ...
    'results',results,'histories',{histories},'complete',true);
file=fullfile(folder,'benchmark.mat'); save(file,'benchmark');
files=plot_parallel_speed(file,folder);
assert(numel(files)==2 && all(isfile(files)));
assert(all(isfile(replace(files,'.eps','.png'))));
benchmark.budget=120; save(file,'benchmark');
rejected=false;
try, plot_parallel_speed(file,folder); catch, rejected=true; end
assert(rejected,'Legacy 120-FE benchmark must not be plotted as 6000 FE.');
fprintf('Parallel convergence export checks passed.\n');
end
