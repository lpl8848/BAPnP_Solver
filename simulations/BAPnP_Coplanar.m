function [R_opt, t_opt] = BAPnP_Coplanar(y_norm, P_world)
    [R_init, t_init] = pnp_linear_coplanar_core(y_norm, P_world);
    [R_opt, t_opt] = pnp_refine_gn(R_init, t_init, y_norm, P_world);
end

function [R, t] = pnp_linear_coplanar_core(y_norm, P_world)
    N = size(P_world, 2);
    cent_3d = mean(P_world, 2);
    P_centered = P_world - cent_3d;
    sq_dists = sum(P_centered.^2, 1);
    rms_dist = sqrt(sum(sq_dists) / N);
    if rms_dist < 1e-6, rms_dist = 1; end
    scale_3d = 1.732050807568877 / rms_dist; 
    P_n = P_centered * scale_3d;
    
    [~, b1] = max(sum(P_n.^2, 1)); p1 = P_n(:, b1);
    [~, b2] = max(sum((P_n - p1).^2, 1)); p2 = P_n(:, b2);
    v12 = p2 - p1; v12_u = v12 / norm(v12);
    vecs = P_n - p1; proj = vecs - v12_u * (v12_u' * vecs);
    [~, b3] = max(sum(proj.^2, 1)); p3 = P_n(:, b3);
    
    nw = cross(p2 - p1, p3 - p1); nw = nw / norm(nw);
    p4 = p1 + nw; 
    
    B = [p2 - p1, p3 - p1, p4 - p1];
    Coeffs = B \ vecs;
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
          
    [~, ~, V] = svd(L3, 'econ');
    rho3 = V(:, end); 
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
    
    P_bases = [P_n(:, b1), P_n(:, b2), P_n(:, b3), p4];
    X_bases = [X1, X2, X3, X4];
    
    Bm_world = P_bases - mean(P_bases, 2);
    Bm_cam = X_bases - mean(X_bases, 2);
    [U, ~, V_svd] = svd(Bm_world * Bm_cam');
    
    R_est = V_svd * U';
    if det(R_est) < 0, R_est = V_svd * diag([1 1 -1]) * U'; end
    t_est_norm = mean(X_bases, 2) - R_est * mean(P_bases, 2);
    
    R = R_est;
    t = t_est_norm / scale_3d - R * cent_3d;
end

% --- Gauss-Newton 优化器 ---
function [R_opt, t_opt] = pnp_refine_gn(R_init, t_init, y_norm, P_world)
    MAX_ITER = 10; MIN_DELTA = 1e-6; 
    pts_obs = y_norm(1:2, :);
    if size(y_norm, 1) == 3, pts_obs = y_norm(1:2, :) ./ y_norm(3, :); end
    N = size(P_world, 2); R = R_init; t = t_init;
    
    for iter = 1:MAX_ITER
        P_cam = R * P_world + t;
        X = P_cam(1, :); Y = P_cam(2, :); Z = P_cam(3, :);
        Z(abs(Z) < 1e-6) = 1e-6; inv_Z = 1 ./ Z;
        u_proj = X .* inv_Z; v_proj = Y .* inv_Z;
        
        res_u = u_proj - pts_obs(1, :); res_v = v_proj - pts_obs(2, :);
        residual = [res_u'; res_v'];
        if norm(residual) < MIN_DELTA, break; end
        
        u_invZ = u_proj .* inv_Z; v_invZ = v_proj .* inv_Z;
        zeros_row = zeros(1, N);
        J_u_trans = [inv_Z; zeros_row; -u_invZ; -u_proj.*v_proj; 1+u_proj.^2; -v_proj];
        J_v_trans = [zeros_row; inv_Z; -v_invZ; -1-v_proj.^2; u_proj.*v_proj; u_proj];
        
        J = [J_u_trans, J_v_trans]'; 
        delta = - (J \ residual);
        if norm(delta) < MIN_DELTA, break; end
        
        d_rho = delta(1:3); d_phi = delta(4:6); 
        theta = norm(d_phi);
        if theta < 1e-10
            dR = eye(3);
        else
            axis = d_phi / theta;
            K = [0 -axis(3) axis(2); axis(3) 0 -axis(1); -axis(2) axis(1) 0];
            dR = eye(3) + sin(theta)*K + (1-cos(theta))*K^2;
        end
        R = dR * R; t = dR * t + d_rho;
    end
    R_opt = R; t_opt = t;
end
