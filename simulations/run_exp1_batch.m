function run_exp1_batch
    % Add paths carefully - only top-level, not recursive subdirs
    addpath('D:\R2020b\bin\pnp');
    addpath('D:\R2020b\bin\pnp\oDLT');
    addpath('D:\R2020b\bin\pnp\oDLT\utils');
    addpath('D:\R2020b\bin\pnp\oDLT\PnP\oDLT');

    fprintf('=== exp1 start: %s ===\n', datestr(now));
    run('D:\R2020b\bin\pnp\exp1.m');
    fprintf('=== exp1 end: %s ===\n', datestr(now));
    save('D:\R2020b\bin\pnp\exp1_results.mat');
    fprintf('Results saved.\n');
end
