function run_planar
    addpath('D:/R2020b/bin/pnp');
    addpath(genpath('D:/R2020b/bin/pnp/MLPnP_matlab_toolbox-master'));
    addpath(genpath('D:/R2020b/bin/pnp/CPnP'));
    addpath(genpath('D:/R2020b/bin/pnp/SRPnP'));
    addpath(genpath('D:/R2020b/bin/pnp/lhm'));
    rehash;

    algorithms = {
        'BAPnP-GN', @BAPnP_new;
        'BAPnP', @pnp_linear_only;
        'EPnP-GN', @run_epnp_guass;
        'OPnP', @run_opnp;
        'RPnP', @run_rpnp;
        'SRPnP-GN', @run_srpnp;
        'MLPnP', @run_mlpnp;
        'CPnP-GN', @run_cpnp;
        'SQPnP', @sqpnp;
        'EPnP-GN-Greedy', @run_epnp_with_cpts;
        'oDLT-GN', @run_odlt_planar_gn;
    };
    n_algs = size(algorithms,1);
    algo_names = algorithms(:,1);

    n_points = 20; n_trials = 500; focal = 800; pixel_noise = 1.0;
    z_spread_levels = [1e-1, 1e-2, 1e-3, 1e-5, 1e-7, 1e-10, 1e-12, 0];
    n_levels = numel(z_spread_levels);
    rot_thresh = 2; trans_thresh = 2;

    MedianRot = zeros(n_levels, n_algs);
    MedianTrans = zeros(n_levels, n_algs);
    SuccessRate = zeros(n_levels, n_algs);

    fprintf('=== Planar experiment (%d levels, %d trials) ===\n', n_levels, n_trials);
    for i = 1:n_levels
        z_spread = z_spread_levels(i);
        rot_err = nan(n_trials, n_algs);
        trans_err = nan(n_trials, n_algs);
        success = false(n_trials, n_algs);
        for k = 1:n_trials
            Pw = 2*(rand(3,n_points) - 0.5);
            if z_spread == 0, Pw(3,:) = 0;
            else, Pw(3,:) = Pw(3,:) * z_spread; end
            R_gt = random_rotation();
            t_gt = [0;0;4] + 0.5*(rand(3,1)-0.5);
            Pc = R_gt * Pw + t_gt;
            y = Pc(1:2,:) ./ Pc(3,:);
            y = y + (pixel_noise / focal) * randn(2,n_points);
            y = [y; ones(1,n_points)];
            for a = 1:n_algs
                try
                    [R_est, t_est] = algorithms{a,2}(y, Pw);
                    R_err = R_gt' * R_est;
                    v = (trace(R_err) - 1) / 2;
                    v = max(min(v,1), -1);
                    re = rad2deg(acos(v));
                    te = norm(t_gt - t_est) / norm(t_gt) * 100;
                    rot_err(k,a) = re; trans_err(k,a) = te;
                    success(k,a) = (re < rot_thresh) && (te < trans_thresh);
                catch
                end
            end
        end
        for a = 1:n_algs
            valid = ~isnan(rot_err(:,a));
            if any(valid)
                MedianRot(i,a) = median(rot_err(valid,a));
                MedianTrans(i,a) = median(trans_err(valid,a));
            end
            SuccessRate(i,a) = mean(success(:,a)) * 100;
        end
        fprintf('Level %d/%d done (z=%.0e)\n', i, n_levels, z_spread);
    end

    save('D:/R2020b/bin/pnp/planar_results.mat', ...
        'MedianRot', 'MedianTrans', 'SuccessRate', 'algo_names', 'z_spread_levels', 'n_trials');
    fprintf('Saved planar_results.mat\n');
end

function R = random_rotation()
    q = randn(4,1); q = q / norm(q);
    w=q(1); x=q(2); y=q(3); z=q(4);
    R = [1-2*y^2-2*z^2, 2*x*y-2*z*w, 2*x*z+2*y*w;
         2*x*y+2*z*w, 1-2*x^2-2*z^2, 2*y*z-2*x*w;
         2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x^2-2*y^2];
end
