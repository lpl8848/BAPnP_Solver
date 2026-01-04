
    % =====================================================================
    % PnP 选点策略对比实验 (FIXED VERSION)
    % =====================================================================
    
    clc; clear; close all;
    warning('off', 'all'); 

    % 1. 实验参数设置
    n_list = [6, 10, 20, 50, 80, 100]; 
    noise_std = 2.0;                   
    n_trials = 1000; 
    
    methods = {'BPnP-Greedy', 'BPnP-PCA-Real', 'BPnP-Random', 'BPnP-ConvHull-Oracle', 'EPnP-Gauss'};
    colors = {'r', 'b', 'g', 'c', 'k'};
    markers = {'-o', '-s', '-^', '--d', '--x'};
    linewidths = [2.5, 1.5, 1.5, 1.5, 2.0];
    
    res_rot_median = zeros(length(n_list), length(methods));
    res_trans_median = zeros(length(n_list), length(methods));
    
    fprintf('======================================================\n');
    fprintf('开始对比实验: Noise = %.1f px, Trials = %d\n', noise_std, n_trials);
    fprintf('======================================================\n');

    for i = 1:length(n_list)
        npts = n_list(i);
        fprintf('正在处理点数 N = %3d ... ', npts);
        
        err_stats = nan(n_trials, length(methods), 2); 
        

        for k = 1:n_trials
            % 1. 生成仿真数据
            [pts3d, pts2d_noisy, pts2d_norm, K, R_gt, t_gt] = generate_data(npts, noise_std);
            P_world = pts3d;
            y_norm = pts2d_norm(1:2, :); 
            
            % --- 1. Greedy ---
            [R1, t1] = pnp_proposed_wrapper(y_norm, P_world, 'greedy');
            [err_stats(k,1,1), err_stats(k,1,2)] = calc_error(R_gt, t_gt, R1, t1);
            
            % --- 2. PCA-Real ---
            [R2, t2] = pnp_proposed_wrapper(y_norm, P_world, 'pca_real');
            [err_stats(k,2,1), err_stats(k,2,2)] = calc_error(R_gt, t_gt, R2, t2);
            
            % --- 3. Random ---
            [R3, t3] = pnp_proposed_wrapper(y_norm, P_world, 'random');
            [err_stats(k,3,1), err_stats(k,3,2)] = calc_error(R_gt, t_gt, R3, t3);

            % --- 4. Oracle ---
            [R4, t4] = pnp_proposed_wrapper(y_norm, P_world, 'convex_opt');
            [err_stats(k,4,1), err_stats(k,4,2)] = calc_error(R_gt, t_gt, R4, t4);

            % --- 5. EPnP-Gauss ---
            try
                % EPnP 接口适配
                P_h = [P_world; ones(1, npts)]';     
                y_pix_h = [pts2d_noisy; ones(1, npts)]'; 
                if exist('efficient_pnp_gauss', 'file')
                    [R5, t5] = efficient_pnp_gauss(P_h, y_pix_h, K);
                    [err_stats(k,5,1), err_stats(k,5,2)] = calc_error(R_gt, t_gt, R5, t5);
                end
            catch
                % EPnP 偶尔会因为奇异值挂掉，这里允许它跳过
            end
        end
        
        % 统计中位数
        for m = 1:length(methods)
            valid_data = squeeze(err_stats(:, m, :));
            valid_idx = ~isnan(valid_data(:,1));
            if sum(valid_idx) > 0
                res_rot_median(i, m) = median(valid_data(valid_idx, 1));
                res_trans_median(i, m) = median(valid_data(valid_idx, 2));
            else
                res_rot_median(i, m) = nan;
                res_trans_median(i, m) = nan;
            end
        end
        fprintf('完成。\n');
    end
    
    % 绘图
    plot_results(n_list, res_rot_median, res_trans_median, methods, colors, markers, linewidths);


% =========================================================================
% LOCAL FUNCTION 1: 算法 Wrapper
% =========================================================================
function [R, t] = pnp_proposed_wrapper(y_norm, P_world, strategy)
    % 线性初始化
    [R_init, t_init] = pnp_linear_with_strategy(y_norm, P_world, strategy);
    
    % LHM 优化需要归一化的视线向量
    V_img = [y_norm; ones(1, size(y_norm, 2))];
    norms = sqrt(sum(V_img.^2, 1));
    V_img = V_img ./ norms;
    
    [R, t] = pnp_refine_lhm(R_init, t_init, V_img, P_world);
end

% =========================================================================
% LOCAL FUNCTION 2: 线性求解器 
% =========================================================================
function [R, t] = pnp_linear_with_strategy(y_norm, P_world, strategy)
    N = size(P_world, 2);
    
    % --- 数据预处理 ---
    cent_3d = mean(P_world, 2);
    P_centered = P_world - cent_3d;
    sq_dists = sum(P_centered.^2, 1);
    rms_dist = sqrt(sum(sq_dists) / N);
    if rms_dist < 1e-6, rms_dist = 1; end
    scale_3d = sqrt(3) / rms_dist; 
    P_n = P_centered * scale_3d;
    
    % --- 策略选择 ---
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
        [~, base_idx] = strategy_convex_hull_optimal(P_n);
        
    else % random or fallback
        base_idx = randperm(N, 4);
    end
    

    P_basis = P_n(:, base_idx);
    Basis_Mat = P_basis(:, 1:3) - P_basis(:, 4);
    RHS = P_n - P_basis(:, 4); 
    
    % 防止基底奇异
    if rcond(Basis_Mat) < 1e-12
        base_idx = randperm(N, 4);
        P_basis = P_n(:, base_idx);
        Basis_Mat = P_basis(:, 1:3) - P_basis(:, 4);
        RHS = P_n - P_basis(:, 4);
    end

    ABC = Basis_Mat \ RHS; % 结果是 3xN
    alphas = ABC(1,:); betas = ABC(2,:); gammas = ABC(3,:); deltas = 1 - sum(ABC, 1);
    
    % 构建 L 矩阵
    y_basis_h = [y_norm(:, base_idx); ones(1, 4)];
    y_all_h   = [y_norm;   ones(1, N)];
    L = zeros(3*N, 4);
    
    for i = 1:N
        obs = y_all_h(:, i);
        % 注意：这里的 alphas(i) 现在是安全的，因为它是 1xN
        L(3*i-2:3*i, 1) = alphas(i) * cross(obs, y_basis_h(:,1));
        L(3*i-2:3*i, 2) = betas(i) * cross(obs, y_basis_h(:,2));
        L(3*i-2:3*i, 3) = gammas(i) * cross(obs, y_basis_h(:,3));
        L(3*i-2:3*i, 4) = deltas(i) * cross(obs, y_basis_h(:,4));
    end
    
    [~, ~, V_svd] = svd(L, 'econ');
    rho = V_svd(:, end);
    
    % 恢复深度
    Zc = alphas*rho(1) + betas*rho(2) + gammas*rho(3) + deltas*rho(4);
    

    if mean(Zc) < 0
        rho = -rho;
        Zc = -Zc;
    end
    
    P_cam_metric = [y_norm(1,:).*Zc; y_norm(2,:).*Zc; Zc];
    
    [R, t] = umeyama_alignment(P_world, P_cam_metric);
end

% =========================================================================
% LOCAL FUNCTION 3: LHM
% =========================================================================
function [R_opt, t_opt] = pnp_refine_lhm(R_init, t_init, V, P_world)
    MAX_ITER = 5; TOL = 1e-5;
    R = R_init; t = t_init;
    p_bar = mean(P_world, 2); P_cent = P_world - p_bar;
    current_err = inf;
    for iter = 1:MAX_ITER
        Q = R * P_world + t;
        v_dot_Q = sum(V .* Q, 1);
        Q_opt = V .* v_dot_Q; 
        E_vec = Q - Q_opt;
        new_err = sum(E_vec.^2, 'all');
        if abs(current_err - new_err) < TOL, break; end
        current_err = new_err;
        q_bar = mean(Q_opt, 2); Q_cent = Q_opt - q_bar;
        W = P_cent * Q_cent';
        [U, ~, V_svd] = svd(W);
        R = V_svd * U';
        if det(R) < 0, R = V_svd * diag([1 1 -1]) * U'; end
        t = q_bar - R * p_bar;
    end
    R_opt = R; t_opt = t;
end

% =========================================================================
% LOCAL FUNCTION 4: Umeyama (带尺度恢复)
% =========================================================================
function [R, t] = umeyama_alignment(P_src, P_dst)
    n = size(P_src, 2);
    mu_s = mean(P_src, 2); mu_d = mean(P_dst, 2);
    P_s_cent = P_src - mu_s; P_d_cent = P_dst - mu_d;
    sig_s = sum(sum(P_s_cent.^2)) / n; 
    
    K = P_d_cent * P_s_cent' / n;
    [U, D, V] = svd(K); S = eye(3);
    if det(K) < 0, S(3,3) = -1; end
    R = U * S * V';
    s = trace(D * S) / sig_s;
    
    % 恢复 t (P_dst = s * R * P_src + t)
    % 我们需要 PnP 格式: P_cam = R * P_world + t_metric
    % t_metric = t_aligned / s
    t = mu_d - s * R * mu_s;
    t = t / s; % 消除尺度模糊，对齐到 Ground Truth 尺度进行评估
end

% =========================================================================
% LOCAL FUNCTION 5: 数据生成
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
% LOCAL FUNCTION 6: 误差计算
% =========================================================================
function [err_rot, err_t] = calc_error(R_gt, t_gt, R_est, t_est)
    R_diff = R_est * R_gt';
    val = (trace(R_diff) - 1) / 2;
    val = max(min(val, 1), -1);
    err_rot = acos(val) * 180 / pi;
    err_t = norm(t_est - t_gt) / norm(t_gt) * 100;
end

% =========================================================================
% LOCAL FUNCTION 7: Convex Hull Oracle
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
        idx = hull_indices(randperm(M,4));
        p1=P_n(:,idx(1)); p2=P_n(:,idx(2)); p3=P_n(:,idx(3)); p4=P_n(:,idx(4));
        V = abs(det([p2-p1, p3-p1, p4-p1])) / 6;
        if V > max_vol
            max_vol = V; best_idx = idx;
        end
    end
    base_idx = best_idx(:)'; vol = max_vol;
end

% =========================================================================
% LOCAL FUNCTION 8: 绘图工具
% =========================================================================
function plot_results(n_list, res_rot, res_trans, methods, colors, markers, linewidths)
    figure('Color', 'w', 'Position', [100, 100, 1200, 500]);
    
    subplot(1, 2, 1); hold on; grid on; box on;
    for m = 1:length(methods)
        plot(n_list, res_rot(:, m), [colors{m} markers{m}], ...
            'LineWidth', linewidths(m), 'MarkerSize', 8, 'MarkerFaceColor', colors{m});
    end
    xlabel('Number of Points (N)'); ylabel('Median Rotation Error (deg)');
    title('Rotation Error'); legend(methods);  xlim([min(n_list), max(n_list)]);

    subplot(1, 2, 2); hold on; grid on; box on;
    for m = 1:length(methods)
        plot(n_list, res_trans(:, m), [colors{m} markers{m}], ...
            'LineWidth', linewidths(m), 'MarkerSize', 8, 'MarkerFaceColor', colors{m});
    end
    xlabel('Number of Points (N)'); ylabel('Median Translation Error (%)');
    title('Translation Error'); legend(methods); xlim([min(n_list), max(n_list)]);
end

