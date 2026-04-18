function [R, t] = BAPnP_new(y_norm, P_world)
% BAPnP: Unified Barycentric Affine-Invariant PnP Solver

    N = size(P_world, 2);
    
    % =====================================================================
    % Stage I: (Geometry-Guided Base Selection)
    % =====================================================================
    cent_3d = mean(P_world, 2);
    P_centered = P_world - cent_3d;
    sq_dists = sum(P_centered.^2, 1);
    rms_dist = sqrt(sum(sq_dists) / N);
    if rms_dist < 1e-6, rms_dist = 1; end
    scale_3d = 1.732050807568877 / rms_dist; 
    P_n = P_centered * scale_3d;
    
   
    [~, b1] = max(sq_dists); p1 = P_n(:, b1);
    [~, b2] = max(sum((P_n - p1).^2, 1)); p2 = P_n(:, b2);
    
    v12 = p2 - p1;
    vecs = P_n - p1;
    cp_x = v12(2)*vecs(3,:) - v12(3)*vecs(2,:);
    cp_y = v12(3)*vecs(1,:) - v12(1)*vecs(3,:);
    cp_z = v12(1)*vecs(2,:) - v12(2)*vecs(1,:);
    d2_line = cp_x.^2 + cp_y.^2 + cp_z.^2;
    [~, b3] = max(d2_line); p3 = P_n(:, b3);
    
    
    v13 = p3 - p1;
    nx = v12(2)*v13(3) - v12(3)*v13(2);
    ny = v12(3)*v13(1) - v12(1)*v13(3);
    nz = v12(1)*v13(2) - v12(2)*v13(1);
    d2_plane = (nx*vecs(1,:) + ny*vecs(2,:) + nz*vecs(3,:)).^2;
    [max_d2_plane, b4] = max(d2_plane); 
    p4 = P_n(:, b4);
    
    % =====================================================================
    % Stage II: = (Dimension-Adaptive Branching)
    % =====================================================================
    is_coplanar = (max_d2_plane < 1e-16); 
    
    if ~is_coplanar
        % --------------------------------------------------
        % Branch A: Non-coplanar (General 3D) - 4x4 System
        % --------------------------------------------------
        base_idx = [b1, b2, b3, b4];
        perm = [base_idx, setdiff(1:N, base_idx)];
        P_n_perm = P_n(:, perm);
        y_norm_perm = y_norm(:, perm);
        
        P1=P_n_perm(:,1); P2=P_n_perm(:,2); P3=P_n_perm(:,3); 
        C0 = (P1+P2+P3)/3;
        
        r1 = P1 - C0; n1 = 1/sqrt(sum(r1.^2)); r1 = r1 * n1;
        v12_C0 = P2 - C0; 
        r3 = [r1(2)*v12_C0(3)-r1(3)*v12_C0(2); r1(3)*v12_C0(1)-r1(1)*v12_C0(3); r1(1)*v12_C0(2)-r1(2)*v12_C0(1)];
        n3 = 1/sqrt(sum(r3.^2)); r3 = r3 * n3;
        r2 = [r3(2)*r1(3)-r3(3)*r1(2); r3(3)*r1(1)-r3(1)*r1(3); r3(1)*r1(2)-r3(2)*r1(1)];
        R0 = [r1'; r2'; r3'];
        
        W_prime = R0 * (P_n_perm - C0);
        B_mat = [W_prime(:,1)-W_prime(:,4), W_prime(:,2)-W_prime(:,4), W_prime(:,3)-W_prime(:,4)];
        Coeffs = B_mat \ (W_prime(:, 5:end) - W_prime(:,4));
        
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
         
        [~, ~, V_svd] = svd(L, 'econ');
        rho = V_svd(:, end);
        if sum(rho) < 0, rho = -rho; end
        if rho(1) < 1e-6, rho(1) = 1e-6; end
        
        Z_others = alphas*rho(1) + betas*rho(2) + gammas*rho(3) + deltas*rho(4);
        Z_all = [rho', Z_others];
        
        % Procrustes & 1-Step Refinement
        P_cam_norm = [y_norm_perm(1,:).*Z_all; y_norm_perm(2,:).*Z_all; Z_all];
        cent_cam = mean(P_cam_norm, 2);
        s_cam = sqrt(sum((P_cam_norm - cent_cam).^2, 'all') / N);
        P_cam_metric = P_cam_norm * (1.732050807568877 / s_cam);
        
        Bm = P_cam_metric - mean(P_cam_metric, 2);
        [U, ~, V_svd] = svd(P_n_perm * Bm');
        R_est = V_svd * U';
        if det(R_est) < 0, R_est = V_svd * diag([1 1 -1]) * U'; end
        t_est_norm = mean(P_cam_metric, 2);
        
        P_cam_ref = R_est * P_n_perm + t_est_norm;
        Z_ref = P_cam_ref(3, :);
        P_cam_ref_corr = [y_norm_perm(1,:).*Z_ref; y_norm_perm(2,:).*Z_ref; Z_ref];
        Bm = P_cam_ref_corr - mean(P_cam_ref_corr, 2);
        [U, ~, V_svd] = svd(P_n_perm * Bm');
        R_init = V_svd * U'; 
        if det(R_init) < 0, R_init = V_svd * diag([1 1 -1]) * U'; end
        t_init = mean(P_cam_ref_corr, 2) / scale_3d - R_init * cent_3d;
        
    else
        % --------------------------------------------------
        % Branch B: Strictly Coplanar - 3x3 System
        % --------------------------------------------------
        nw = cross(p2 - p1, p3 - p1); nw = nw / norm(nw);
        p4_virtual = p1 + nw; 
        
        B_mat = [p2 - p1, p3 - p1, p4_virtual - p1];
        Coeffs = B_mat \ vecs;
        betas = Coeffs(1, :); gammas = Coeffs(2, :); deltas = Coeffs(3, :); 
        alphas = 1 - betas - gammas - deltas;
        
        perm = [b1, b2, b3, setdiff(1:N, [b1, b2, b3])];
        y_norm_perm = y_norm(:, perm);
        alphas_perm = alphas(perm); betas_perm = betas(perm); gammas_perm = gammas(perm);
        
        y1 = y_norm_perm(:, 1); y2 = y_norm_perm(:, 2); y3 = y_norm_perm(:, 3);
        y_oth = y_norm_perm(:, 4:end); N_oth = size(y_oth, 2);
        
        cp1 = cross(y_oth, repmat(y1, 1, N_oth), 1);
        cp2 = cross(y_oth, repmat(y2, 1, N_oth), 1);
        cp3 = cross(y_oth, repmat(y3, 1, N_oth), 1);
        
        L3 = [reshape(alphas_perm(4:end) .* cp1, [], 1), ...
              reshape(betas_perm(4:end)  .* cp2, [], 1), ...
              reshape(gammas_perm(4:end) .* cp3, [], 1)];
              
        [~, ~, V_svd] = svd(L3, 'econ');
        rho3 = V_svd(:, end); 
        if sum(rho3) < 0, rho3 = -rho3; end
        if rho3(1) < 1e-6, rho3(1) = 1e-6; end
        
        Z_all = alphas_perm * rho3(1) + betas_perm * rho3(2) + gammas_perm * rho3(3);
        P_cam_unscaled = y_norm_perm .* Z_all;
        cent_cam_unscaled = mean(P_cam_unscaled, 2);
        s_cam = sqrt(sum((P_cam_unscaled - cent_cam_unscaled).^2, 'all') / N);
        
        cent_world_norm = mean(P_n(:, perm), 2);
        s_world = sqrt(sum((P_n(:, perm) - cent_world_norm).^2, 'all') / N);
        P_cam_metric = P_cam_unscaled * (s_world / s_cam);
        
        X1 = P_cam_metric(:, 1); X2 = P_cam_metric(:, 2); X3 = P_cam_metric(:, 3);
        nc = cross(X2 - X1, X3 - X1); nc = nc / norm(nc); 
        X4 = X1 + nc;       
        
        P_bases = [P_n(:, b1), P_n(:, b2), P_n(:, b3), p4_virtual];
        X_bases = [X1, X2, X3, X4];
        
        Bm_world = P_bases - mean(P_bases, 2);
        Bm_cam = X_bases - mean(X_bases, 2);
        [U, ~, V_svd] = svd(Bm_world * Bm_cam');
        
        R_init = V_svd * U';
        if det(R_init) < 0, R_init = V_svd * diag([1 1 -1]) * U'; end
        t_est_norm = mean(X_bases, 2) - R_init * mean(P_bases, 2);
        t_init = t_est_norm / scale_3d - R_init * cent_3d;
    end
    
    % =====================================================================
    % Stage III: Single-Step Gauss-Newton Refinement
    % =====================================================================
    [R, t] = pnp_refine_gn(R_init, t_init, y_norm, P_world);
end

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
function [R_opt, t_opt] = pnp_refine_gn(R_init, t_init, y_norm, P_world)
    MAX_ITER = 1; 
    MIN_DELTA = 1e-6; 
    
    if size(y_norm, 1) == 3
        pts_obs = y_norm(1:2, :) ./ y_norm(3, :);
    else
        pts_obs = y_norm(1:2, :);
    end
    
    N = size(P_world, 2);
    R = R_init;
    t = t_init;
    
    for iter = 1:MAX_ITER
        P_cam = R * P_world + t;
        X = P_cam(1, :); Y = P_cam(2, :); Z = P_cam(3, :);
        
        Z(abs(Z) < 1e-6) = 1e-6; 
        inv_Z = 1 ./ Z;
        u_proj = X .* inv_Z; v_proj = Y .* inv_Z;
        
        res_u = u_proj - pts_obs(1, :);
        res_v = v_proj - pts_obs(2, :);
        residual = [res_u'; res_v']; 
        
        if norm(residual) < MIN_DELTA, break; end
        
        u_invZ = u_proj .* inv_Z;
        v_invZ = v_proj .* inv_Z;
        
        zeros_row = zeros(1, N);
        J_u_trans = [inv_Z; zeros_row; -u_invZ; -u_proj.*v_proj; 1+u_proj.^2; -v_proj];
        J_v_trans = [zeros_row; inv_Z; -v_invZ; -1-v_proj.^2; u_proj.*v_proj; u_proj];
        
        J = [J_u_trans, J_v_trans]'; 
        delta = - (J \ residual);
        
        if norm(delta) < MIN_DELTA, break; end
        
        d_rho = delta(1:3); 
        d_phi = delta(4:6); 
        
        theta = norm(d_phi);
        if theta < 1e-10
            dR = eye(3);
        else
            axis = d_phi / theta;
            K = [0 -axis(3) axis(2); axis(3) 0 -axis(1); -axis(2) axis(1) 0];
            dR = eye(3) + sin(theta)*K + (1-cos(theta))*K^2;
        end
        
        R = dR * R;
        t = dR * t + d_rho;
    end
    
    R_opt = R;
    t_opt = t;
end
