function [R, t] = pnp_linear_only(y_norm, P_world)
% PNP_LINEAR_ONLY 仅包含线性求解部分的 BAPnP 算法 (无迭代优化)
%

% 输入:
%   y_norm:  2xN 或 3xN 归一化平面坐标 (x, y) 或 (x, y, 1)
%   P_world: 3xN 世界坐标
%
% 输出:
%   R, t:    线性解算的位姿


    N = size(P_world, 2);
    
    % 1. 3D 数据中心化与归一化 
    cent_3d = mean(P_world, 2);
    P_centered = P_world - cent_3d;
    sq_dists = sum(P_centered.^2, 1);
    rms_dist = sqrt(sum(sq_dists) / N);
    if rms_dist < 1e-6, rms_dist = 1; end
    scale_3d = 1.732050807568877 / rms_dist; 
    P_n = P_centered * scale_3d;
    
    % 2. 极速基底选择
    base_idx = zeros(1, 4);
    [~, base_idx(1)] = max(sq_dists);
    p1 = P_n(:, base_idx(1));
    
    d2 = sum((P_n - p1).^2, 1);
    [~, base_idx(2)] = max(d2);
    p2 = P_n(:, base_idx(2));
    
    v12 = p2 - p1;
    v12_sq = sum(v12.^2); if v12_sq < 1e-8, v12_sq = 1; end
    
    vecs = P_n - p1;
    cp_x = v12(2)*vecs(3,:) - v12(3)*vecs(2,:);
    cp_y = v12(3)*vecs(1,:) - v12(1)*vecs(3,:);
    cp_z = v12(1)*vecs(2,:) - v12(2)*vecs(1,:);
    d2_line = cp_x.^2 + cp_y.^2 + cp_z.^2;
    [~, base_idx(3)] = max(d2_line);
    p3 = P_n(:, base_idx(3));
    
    v13 = p3 - p1;
    nx = v12(2)*v13(3) - v12(3)*v13(2);
    ny = v12(3)*v13(1) - v12(1)*v13(3);
    nz = v12(1)*v13(2) - v12(2)*v13(1);
    d2_plane = (nx*vecs(1,:) + ny*vecs(2,:) + nz*vecs(3,:)).^2;
    [~, base_idx(4)] = max(d2_plane);
    
    % 重排列数据，让基底点在前
    perm = [base_idx, setdiff(1:N, base_idx)];
    P_n_perm = P_n(:, perm);
    y_norm_perm = y_norm(:, perm);
    
    % 3. 线性求解控制点系数
    P1=P_n_perm(:,1); P2=P_n_perm(:,2); P3=P_n_perm(:,3); 
    C0 = (P1+P2+P3)/3;
    
    % 构造局部坐标系 
    r1 = P1 - C0; n1 = 1/sqrt(sum(r1.^2)); r1 = r1 * n1;
    v12 = P2 - C0; 
    r3 = [r1(2)*v12(3)-r1(3)*v12(2); r1(3)*v12(1)-r1(1)*v12(3); r1(1)*v12(2)-r1(2)*v12(1)];
    n3 = 1/sqrt(sum(r3.^2)); r3 = r3 * n3;
    r2 = [r3(2)*r1(3)-r3(3)*r1(2); r3(3)*r1(1)-r3(1)*r1(3); r3(1)*r1(2)-r3(2)*r1(1)];
    R0 = [r1'; r2'; r3'];
    
    W_prime = R0 * (P_n_perm - C0);
    B = [W_prime(:,1)-W_prime(:,4), W_prime(:,2)-W_prime(:,4), W_prime(:,3)-W_prime(:,4)];
    Coeffs = B \ (W_prime(:, 5:end) - W_prime(:,4));
    
    alphas = Coeffs(1,:); betas = Coeffs(2,:); gammas = Coeffs(3,:); deltas = 1 - sum(Coeffs, 1);
    
    % 4. 构造 MX = 0 线性系统求解深度
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
    rho = V(:, end); % 最小特征向量
    if sum(rho) < 0, rho = -rho; end
    if rho(1) < 1e-6, rho(1) = 1e-6; end
    
    Z_others = alphas*rho(1) + betas*rho(2) + gammas*rho(3) + deltas*rho(4);
    Z_all = [rho', Z_others];
    
    % 5. 绝对定向 (Procrustes) - 恢复旋转和平移
    P_cam_norm = [y_norm_perm(1,:).*Z_all; y_norm_perm(2,:).*Z_all; Z_all];
    cent_cam = mean(P_cam_norm, 2);
    sq_norm_cam = sum((P_cam_norm - cent_cam).^2, 'all');
    s_cam = sqrt(sq_norm_cam / N);
    true_scale = 1.732050807568877 / s_cam;
    P_cam_metric = P_cam_norm * true_scale;
    
    Bm = P_cam_metric - mean(P_cam_metric, 2);
    H = P_n_perm * Bm';
    [U, ~, V] = svd(H);
    R_est = V * U';
    if det(R_est) < 0, R_est = V * diag([1 1 -1]) * U'; end
    t_est_norm = mean(P_cam_metric, 2);
    
    % 6. 单步代数对齐 (One-Step Analytic Alignment)
    P_cam_ref = R_est * P_n_perm + t_est_norm;
    Z_ref = P_cam_ref(3, :);
    P_cam_ref_corr = [y_norm_perm(1,:).*Z_ref; y_norm_perm(2,:).*Z_ref; Z_ref];
    Bm = P_cam_ref_corr - mean(P_cam_ref_corr, 2);
    H = P_n_perm * Bm';
    [U, ~, V] = svd(H);
    R = V * U'; 
    if det(R) < 0, R = V * diag([1 1 -1]) * U'; end
    t_temp = mean(P_cam_ref_corr, 2);
    
    % 7. 还原尺度和中心
    t = t_temp / scale_3d - R * cent_3d;
    
end


