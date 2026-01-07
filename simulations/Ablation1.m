% =====================================================================
% Ablation1
% =====================================================================
% =====================================================================

clc; clear; close all;
warning('off', 'all'); 


n_list = [6, 10, 20, 50, 80, 100]; 
noise_std = 2.0;                    
n_trials = 2000; 
 
methods = {'BPnP (Greedy)', 'BPnP-PCA', 'BPnP-Random', 'BPnP-Convex'};
colors = {'r', 'b', 'g', 'c'};
markers = {'-o', '-s', '-^', '--d'};
linewidths = [2.5, 1.5, 1.5, 1.5];
 
res_rot_median = zeros(length(n_list), length(methods));
res_trans_median = zeros(length(n_list), length(methods));
 
fprintf('======================================================\n');
fprintf(' Noise = %.1f px, Trials = %d\n', noise_std, n_trials);
fprintf('======================================================\n');

for i = 1:length(n_list)
    npts = n_list(i);
    fprintf(' N = %3d ... ', npts);
    
    err_stats = nan(n_trials, length(methods), 2); 
    
    for k = 1:n_trials
   
        [pts3d, pts2d_noisy, pts2d_norm, K, R_gt, t_gt] = generate_data(npts, noise_std);
        P_world = pts3d;
        y_norm = pts2d_norm; 
        
        
        % --- 1. Greedy (Proposed) ---
        [R1, t1] = pnp_linear_strategy_impl(y_norm, P_world, 'greedy');
        [err_stats(k,1,1), err_stats(k,1,2)] = calc_error(R_gt, t_gt, R1, t1);
        
        % --- 2. PCA-Real ---
        [R2, t2] = pnp_linear_strategy_impl(y_norm, P_world, 'pca_real');
        [err_stats(k,2,1), err_stats(k,2,2)] = calc_error(R_gt, t_gt, R2, t2);
        
        % --- 3. Random ---
        [R3, t3] = pnp_linear_strategy_impl(y_norm, P_world, 'random');
        [err_stats(k,3,1), err_stats(k,3,2)] = calc_error(R_gt, t_gt, R3, t3);

        % --- 4. Convex Hull Oracle ---
        [R4, t4] = pnp_linear_strategy_impl(y_norm, P_world, 'convex_opt');
        [err_stats(k,4,1), err_stats(k,4,2)] = calc_error(R_gt, t_gt, R4, t4);
    end
    

    for m = 1:length(methods)
        valid_data = squeeze(err_stats(:, m, :));
        valid_idx = ~isnan(valid_data(:,1));
        if sum(valid_idx) > 0
            res_rot_median(i, m) = mean(valid_data(valid_idx, 1));
            res_trans_median(i, m) = mean(valid_data(valid_idx, 2));
        else
            res_rot_median(i, m) = nan;
            res_trans_median(i, m) = nan;
        end
    end
    fprintf('OK。\n');
end
 

plot_results(n_list, res_rot_median, res_trans_median, methods, colors, markers, linewidths);


% =========================================================================
% LINEAR SOLVER IMPLEMENTATION (Pure Linear, No Iterative Refinement)
% =========================================================================
function [R, t] = pnp_linear_strategy_impl(y_norm, P_world, strategy)
    N = size(P_world, 2);
    

    cent_3d = mean(P_world, 2);
    P_centered = P_world - cent_3d;
    sq_dists = sum(P_centered.^2, 1);
    rms_dist = sqrt(sum(sq_dists) / N);
    if rms_dist < 1e-6, rms_dist = 1; end
    scale_3d = sqrt(3) / rms_dist; 
    P_n = P_centered * scale_3d;
    

    base_idx = zeros(1, 4);
    
    if strcmp(strategy, 'greedy')

        [~, base_idx(1)] = max(sq_dists);
        p1 = P_n(:, base_idx(1));
        [~, base_idx(2)] = max(sum((P_n - p1).^2, 1));
        p2 = P_n(:, base_idx(2));
        v12 = p2 - p1; vecs = P_n - p1;
        cp = cross(repmat(v12,1,N), vecs);
        [~, base_idx(3)] = max(sum(cp.^2, 1));
        p3 = P_n(:, base_idx(3));
        v13 = p3 - p1; n_plane = cross(v12, v13);
        [~, base_idx(4)] = max((n_plane' * vecs).^2);
        
    elseif strcmp(strategy, 'pca_real')

        [U, S, ~] = svd(P_n * P_n');
        sigmas = sqrt(diag(S) / N);
        targets = [mean(P_n,2), U(:,1)*sigmas(1)*2, U(:,2)*sigmas(2)*2, U(:,3)*sigmas(3)*2];
        mask = true(1, N);
        for k = 1:4
            d = sum((P_n - targets(:,k)).^2, 1);
            d(~mask) = inf;
            [~, idx] = min(d);
            base_idx(k) = idx;
            mask(idx) = false;
        end
        
    elseif strcmp(strategy, 'convex_opt')
        % (Oracle)
        [~, base_idx] = strategy_convex_hull_optimal(P_n);
        
    else % random
        base_idx = randperm(N, 4);
    end
    

    perm = [base_idx, setdiff(1:N, base_idx)];
    P_n_perm = P_n(:, perm);
    y_norm_perm = y_norm(:, perm);
    

    P1=P_n_perm(:,1); P2=P_n_perm(:,2); P3=P_n_perm(:,3); 
    C0 = (P1+P2+P3)/3;
    r1 = P1 - C0; n1 = 1/sqrt(sum(r1.^2)); r1 = r1 * n1;
    v12 = P2 - C0; 
    r3 = [r1(2)*v12(3)-r1(3)*v12(2); r1(3)*v12(1)-r1(1)*v12(3); r1(1)*v12(2)-r1(2)*v12(1)];
    n3 = 1/sqrt(sum(r3.^2)); r3 = r3 * n3;
    r2 = [r3(2)*r1(3)-r3(3)*r1(2); r3(3)*r1(1)-r3(1)*r1(3); r3(1)*r1(2)-r3(2)*r1(1)];
    R0 = [r1'; r2'; r3'];
    
    W_prime = R0 * (P_n_perm - C0);
    B = [W_prime(:,1)-W_prime(:,4), W_prime(:,2)-W_prime(:,4), W_prime(:,3)-W_prime(:,4)];
    
    if rcond(B) < 1e-12
         base_idx = randperm(N, 4);
         perm = [base_idx, setdiff(1:N, base_idx)];
         P_n_perm = P_n(:, perm);
         y_norm_perm = y_norm(:, perm);
         Basis_Mat = P_n_perm(:,1:3) - P_n_perm(:,4);
         B = Basis_Mat; 
    end
    
    Coeffs = B \ (W_prime(:, 5:end) - W_prime(:,4));
    alphas = Coeffs(1,:); betas = Coeffs(2,:); gammas = Coeffs(3,:); deltas = 1 - sum(Coeffs, 1);
    

    y1=y_norm_perm(:,1); y2=y_norm_perm(:,2); y3=y_norm_perm(:,3); y4=y_norm_perm(:,4);
    y_others = y_norm_perm(:, 5:end);
    
    cp1_x = y_others(2,:)*y1(3) - y_others(3,:)*y1(2); cp1_y = y_others(3,:)*y1(1) - y_others(1,:)*y1(3); cp1_z = y_others(1,:)*y1(2) - y_others(2,:)*y1(1);
    cp2_x = y_others(2,:)*y2(3) - y_others(3,:)*y2(2); cp2_y = y_others(3,:)*y2(1) - y_others(1,:)*y2(3); cp2_z = y_others(1,:)*y2(2) - y_others(2,:)*y2(1);
    cp3_x = y_others(2,:)*y3(3) - y_others(3,:)*y3(2); cp3_y = y_others(3,:)*y3(1) - y_others(1,:)*y3(3); cp3_z = y_others(1,:)*y3(2) - y_others(2,:)*y3(1);
    cp4_x = y_others(2,:)*y4(3) - y_others(3,:)*y4(2); cp4_y = y_others(3,:)*y4(1) - y_others(1,:)*y4(3); cp4_z = y_others(1,:)*y4(2) - y_others(2,:)*y4(1);
    
    L = [reshape([alphas.*cp1_x; alphas.*cp1_y; alphas.*cp1_z], [], 1), ...
         reshape([betas .*cp2_x; betas .*cp2_y; betas .*cp2_z], [], 1), ...
         reshape([gammas.*cp3_x; gammas.*cp3_y; gammas.*cp3_z], [], 1), ...
         reshape([deltas.*cp4_x; deltas.*cp4_y; deltas.*cp4_z], [], 1)];
     
    [~, ~, V] = svd(L, 'econ');
    rho = V(:, end);
    if sum(rho) < 0, rho = -rho; end
    if rho(1) < 1e-6, rho(1) = 1e-6; end
    
    Z_others = alphas*rho(1) + betas*rho(2) + gammas*rho(3) + deltas*rho(4);
    Z_all = [rho', Z_others];
    

    P_cam_norm = [y_norm_perm(1,:).*Z_all; y_norm_perm(2,:).*Z_all; Z_all];
    cent_cam = mean(P_cam_norm, 2);
    sq_norm_cam = sum((P_cam_norm - cent_cam).^2, 'all');
    s_cam = sqrt(sq_norm_cam / N);
    true_scale = sqrt(3) / s_cam;
    P_cam_metric = P_cam_norm * true_scale;
    
    Bm = P_cam_metric - mean(P_cam_metric, 2);
    H = P_n_perm * Bm';
    [U, ~, V] = svd(H);
    R_est = V * U';
    if det(R_est) < 0, R_est = V * diag([1 1 -1]) * U'; end
    t_est_norm = mean(P_cam_metric, 2);
    

    P_cam_ref = R_est * P_n_perm + t_est_norm;
    Z_ref = P_cam_ref(3, :);
    P_cam_ref_corr = [y_norm_perm(1,:).*Z_ref; y_norm_perm(2,:).*Z_ref; Z_ref];
    Bm = P_cam_ref_corr - mean(P_cam_ref_corr, 2);
    H = P_n_perm * Bm';
    [U, ~, V] = svd(H);
    R = V * U'; 
    if det(R) < 0, R = V * diag([1 1 -1]) * U'; end
    t_temp = mean(P_cam_ref_corr, 2);
    

    t = t_temp / scale_3d - R * cent_3d;
end

% =========================================================================
% HELPER: Convex Hull Oracle
% =========================================================================
function [vol, base_idx] = strategy_convex_hull_optimal(P_n)
    try
        K = convhull(P_n(1,:), P_n(2,:), P_n(3,:));
    catch
        base_idx = randperm(size(P_n,2), 4); vol=0; return;
    end
    hull_indices = unique(K(:));
    M = numel(hull_indices);
    if M < 4 
        base_idx = randperm(size(P_n,2), 4); vol=0; return;
    end
    max_trials = 200; max_vol = 0; best_idx = hull_indices(1:4);
    for k = 1:max_trials
        if M == 4, idx = hull_indices; else, idx = hull_indices(randperm(M,4)); end
        p1=P_n(:,idx(1)); p2=P_n(:,idx(2)); p3=P_n(:,idx(3)); p4=P_n(:,idx(4));
        V = abs(det([p2-p1, p3-p1, p4-p1])) / 6;
        if V > max_vol
            max_vol = V; best_idx = idx;
        end
    end
    base_idx = best_idx(:)'; vol = max_vol;
end

% =========================================================================
% HELPER: Data Generation
% =========================================================================
function [pts3d, pts2d_noisy, pts2d_norm, K, R, t] = generate_data(npts, noise_std)
    w=640; h=480; f=800; K=[f,0,w/2; 0,f,h/2; 0,0,1];
    min_depth = 5; max_depth = 10;
    d = min_depth + (max_depth - min_depth) * rand(1, npts);
    pts2d_true = [w*rand(1, npts); h*rand(1, npts); ones(1, npts)];
    pts2d_norm_true = K \ pts2d_true;
    [U,~,V] = svd(rand(3)); R = U*diag([1,1,det(U*V')])*V'; t = min_depth/2 * rand(3,1);
    pts3d = R' * (pts2d_norm_true .* d - t * ones(1, npts));
    noise = noise_std * randn(2, npts);
    pts2d_noisy = pts2d_true;
    pts2d_noisy(1:2, :) = pts2d_noisy(1:2, :) + noise;
    pts2d_norm = K \ pts2d_noisy;
end

% =========================================================================
% HELPER: Error Calculation
% =========================================================================
function [err_rot, err_t] = calc_error(R_gt, t_gt, R_est, t_est)
    R_diff = R_est * R_gt';
    val = (trace(R_diff) - 1) / 2;
    val = max(min(val, 1), -1);
    err_rot = acos(val) * 180 / pi;
    err_t = norm(t_est - t_gt) / norm(t_gt) * 100;
end

% =========================================================================
% HELPER: Plotting (Two Separate Figures)
% =========================================================================
function plot_results(n_list, res_rot, res_trans, methods, colors, markers, linewidths)

    % ------------------- Figure 1: Rotation Error -------------------
    figure('Color', 'w', 'Position', [100, 100, 600, 500]);
    hold on; grid on; box on;
    
    for m = 1:length(methods)
        plot(n_list, res_rot(:, m), [colors{m} markers{m}], ...
            'LineWidth', linewidths(m), 'MarkerSize', 8, 'MarkerFaceColor', colors{m});
    end
    xlabel('Number of Points (N)');
    ylabel('Rotation Error (deg)');
    legend(methods, 'Location', 'best');
    xlim([min(n_list), max(n_list)]);
    title('Rotation Error vs Number of Points');


    % ------------------- Figure 2: Translation Error -------------------
    figure('Color', 'w', 'Position', [750, 100, 600, 500]);
    hold on; grid on; box on;
    
    for m = 1:length(methods)
        plot(n_list, res_trans(:, m), [colors{m} markers{m}], ...
            'LineWidth', linewidths(m), 'MarkerSize', 8, 'MarkerFaceColor', colors{m});
    end
    xlabel('Number of Points (N)');
    ylabel('Translation Error (%)');
    legend(methods, 'Location', 'best');
    xlim([min(n_list), max(n_list)]);
    title('Translation Error vs Number of Points');

end

