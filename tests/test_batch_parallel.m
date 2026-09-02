clear;
clc;

fprintf('MATLAB: %s\n', version);
fprintf('Starting process pool...\n');

p = parpool("Processes", 4);

fprintf('Pool started with %d workers.\n', p.NumWorkers);

x = zeros(100,1);

parfor k = 1:100
    x(k) = sin(k)^2 + cos(k)^2;
end

fprintf('parfor completed. sum(x) = %.15f\n', sum(x));

delete(p);

fprintf('TEST PASSED.\n');