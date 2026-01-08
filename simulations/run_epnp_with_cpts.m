function [R, t] = run_epnp_with_cpts(normalized_pts2d, pts3d)
% RUN_EPNP_WITH_CPTS 
%
% Inputs:
%   normalized_pts2d - 2xN (or 3xN) normalized image coordinates
%   pts3d            - 3xN world coordinates
%
% Outputs:
%   R - 3x3 Estimated Rotation Matrix
%   t - 3x1 Estimated Translation Vector


    if size(pts3d, 1) ~= 3 && size(pts3d, 2) == 3
        pts3d = pts3d'; 
    end
    
  
    if size(normalized_pts2d, 1) == 3
        normalized_pts2d = normalized_pts2d(1:2, :);
    elseif size(normalized_pts2d, 1) ~= 2 && size(normalized_pts2d, 2) == 2
        normalized_pts2d = normalized_pts2d';
    end

 
    % 
    cpts_greedy = select_greedy_bases(pts3d);


    n = size(pts3d, 2);
    x3d_h = [pts3d', ones(n,1)];      % Nx4 Homogeneous 3D
    x2d_h = [normalized_pts2d', ones(n,1)]; % Nx3 Homogeneous 2D
    A_eye = eye(3);                 

    % 4. 
    [RR, tt, ~, ~] = efficient_pnp_custom_cpts(x3d_h, x2d_h, A_eye, cpts_greedy);
    [R, t] = pnp_refine_gn(RR, tt, normalized_pts2d, pts3d);
end

%% =========================================================================
%  Sub-function: Greedy Base Selection
% =========================================================================
function cpts = select_greedy_bases(pts3d)
    % Section III-B 
    N = size(pts3d, 2);
    used_indices = false(1, N);
    
    % 1. 
    centroid = mean(pts3d, 2);
    
    % 2. P_b1: 
    dists = sum((pts3d - centroid).^2, 1);
    [~, idx1] = max(dists);
    Pb1 = pts3d(:, idx1);
    used_indices(idx1) = true;
    
    % 3. P_b2: 
    dists = sum((pts3d - Pb1).^2, 1);
    dists(used_indices) = -1; 
    [~, idx2] = max(dists);
    Pb2 = pts3d(:, idx2);
    used_indices(idx2) = true;
    
    % 4. P_b3:
    u = Pb2 - Pb1; 
    u_norm = norm(u);
    if u_norm < 1e-8, u = [1;0;0]; else, u = u / u_norm; end
    
    max_dist = -1; idx3 = -1;
    for k = 1:N
        if used_indices(k), continue; end
        P = pts3d(:, k);
        v = P - Pb1;
        dist_sq = norm(cross(u, v))^2; 
        if dist_sq > max_dist, max_dist = dist_sq; idx3 = k; end
    end
    
 
    if idx3 == -1
        idx3 = find(~used_indices, 1); 
    end
    Pb3 = pts3d(:, idx3);
    used_indices(idx3) = true;
    
    % 5. P_b4
    v1_vec = Pb2 - Pb1; 
    v2_vec = Pb3 - Pb1;
    n_vec = cross(v1_vec, v2_vec); 
    n_norm = norm(n_vec);
    
    if n_norm < 1e-8
         
         idx4 = find(~used_indices, 1);
    else
        n_vec = n_vec / n_norm;
        max_dist = -1; idx4 = -1;
        for k = 1:N
            if used_indices(k), continue; end
            P = pts3d(:, k);
            dist = abs(dot(n_vec, P - Pb1));
            if dist > max_dist, max_dist = dist; idx4 = k; end
        end
    end
    
    if idx4 == -1, idx4 = find(~used_indices, 1); end
    Pb4 = pts3d(:, idx4);
    
    cpts = [Pb1, Pb2, Pb3, Pb4]'; % 返回 4x3 格式
end

%% =========================================================================
%  Sub-function: Modified EPnP Core (Custom Control Points)
%  NOTE: Requires standard EPnP helper functions (compute_alphas, etc.)
% =========================================================================
function [R,T,Xc,best_solution]=efficient_pnp_custom_cpts(x3d_h,x2d_h,A,Cw_custom)

    Xw=x3d_h(:,1:3);
    U=x2d_h(:,1:2);
    THRESHOLD_REPROJECTION_ERROR=20;

    % CHANGE 1: Use custom control points directly
    Cw = Cw_custom; 

    % CHANGE 2: Compute distances dsq based on custom Cw
    c1=Cw(1,:); c2=Cw(2,:); c3=Cw(3,:); c4=Cw(4,:);
    dsq = [sum((c1-c2).^2); sum((c1-c3).^2); sum((c1-c4).^2); ...
           sum((c2-c3).^2); sum((c2-c4).^2); sum((c3-c4).^2)];

    % Standard EPnP Logic starts here
    % 注意：以下函数依赖于外部 EPnP 库
    Alph=compute_alphas(Xw,Cw);
    M=compute_M_ver2(U,Alph,A);
    Km=kernel_noise(M,4); 
        
    % 1. dim(ker)=1
    X1=Km(:,end);
    [Cc,Xc]=compute_norm_sign_scaling_factor(X1,Cw,Alph,Xw);
    [R,T]=getrotT(Xw,Xc); 
    err(1)=reprojection_error_usingRT(Xw,U,R,T,A);
    sol(1).Xc=Xc; sol(1).Cc=Cc; sol(1).R=R; sol(1).T=T; sol(1).error=err(1);

    % 2. dim(ker)=2
    Km1=Km(:,end-1); Km2=Km(:,end);
    D=compute_constraint_distance_2param_6eq_3unk(Km1,Km2);
    betas_=inv(D'*D)*D'*dsq; % 使用动态计算的 dsq
    beta1=sqrt(abs(betas_(1)));
    beta2=sqrt(abs(betas_(3)))*sign(betas_(2))*sign(betas_(1));
    X2=beta1*Km1+beta2*Km2;
    [Cc,Xc]=compute_norm_sign_scaling_factor(X2,Cw,Alph,Xw);
    [R,T]=getrotT(Xw,Xc);
    err(2)=reprojection_error_usingRT(Xw,U,R,T,A);
    sol(2).Xc=Xc; sol(2).Cc=Cc; sol(2).R=R; sol(2).T=T; sol(2).error=err(2);

    % 3. dim(ker)=3
    if min(err)>THRESHOLD_REPROJECTION_ERROR
        Km1=Km(:,end-2); Km2=Km(:,end-1); Km3=Km(:,end);
        D=compute_constraint_distance_3param_6eq_6unk(Km1,Km2,Km3);
        betas_=inv(D)*dsq; % Use custom dsq
        beta1=sqrt(abs(betas_(1)));
        beta2=sqrt(abs(betas_(4)))*sign(betas_(2))*sign(betas_(1));
        beta3=sqrt(abs(betas_(6)))*sign(betas_(3))*sign(betas_(1));
        X3=beta1*Km1+beta2*Km2+beta3*Km3;
        [Cc,Xc]=compute_norm_sign_scaling_factor(X3,Cw,Alph,Xw);
        [R,T]=getrotT(Xw,Xc);
        err(3)=reprojection_error_usingRT(Xw,U,R,T,A);
        sol(3).Xc=Xc; sol(3).Cc=Cc; sol(3).R=R; sol(3).T=T; sol(3).error=err(3);
    else
        err(3)=inf;
    end

    % 4. dim(ker)=4
    if min(err)>THRESHOLD_REPROJECTION_ERROR
        Km1=Km(:,end-3); Km2=Km(:,end-2); Km3=Km(:,end-1); Km4=Km(:,end);
        D=compute_constraint_distance_orthog_4param_9eq_10unk(Km1,Km2,Km3,Km4);
        lastcolumn=[-dsq',0,0,0]'; % Use custom dsq
        D_=[D,lastcolumn];
        Kd=null(D_);
        P=compute_permutation_constraint4(Kd);
        lambdas_=kernel_noise(P,1);
        lambda(1)=sqrt(abs(lambdas_(1)));
        lambda(2)=sqrt(abs(lambdas_(6)))*sign(lambdas_(2))*sign(lambdas_(1));
        lambda(3)=sqrt(abs(lambdas_(10)))*sign(lambdas_(3))*sign(lambdas_(1));
        lambda(4)=sqrt(abs(lambdas_(13)))*sign(lambdas_(4))*sign(lambdas_(1));
        lambda(5)=sqrt(abs(lambdas_(15)))*sign(lambdas_(5))*sign(lambdas_(1));
        betass_=lambda(1)*Kd(:,1)+lambda(2)*Kd(:,2)+lambda(3)*Kd(:,3)+lambda(4)*Kd(:,4)+lambda(5)*Kd(:,5);
        beta1=sqrt(abs(betass_(1)));
        beta2=sqrt(abs(betass_(5)))*sign(betass_(2));
        beta3=sqrt(abs(betass_(8)))*sign(betass_(3));
        beta4=sqrt(abs(betass_(10)))*sign(betass_(4));
        X4=beta1*Km1+beta2*Km2+beta3*Km3+beta4*Km4;
        [Cc,Xc]=compute_norm_sign_scaling_factor(X4,Cw,Alph,Xw);
        [R,T]=getrotT(Xw,Xc);
        err(4)=reprojection_error_usingRT(Xw,U,R,T,A);
        sol(4).Xc=Xc; sol(4).Cc=Cc; sol(4).R=R; sol(4).T=T; sol(4).error=err(4);
    else
        err(4)=inf;
    end

    [~,best_solution]=min(err);
    R=sol(best_solution).R;
    T=sol(best_solution).T;
    Xc=sol(best_solution).Xc;
end
function [R_opt, t_opt] = pnp_refine_gn(R_init, t_init, y_norm, P_world)
    MAX_ITER = 10;
    MIN_DELTA = 1e-6; % 收敛阈值
    

    if size(y_norm, 1) == 3
        pts_obs = y_norm(1:2, :) ./ y_norm(3, :);
    else
        pts_obs = y_norm(1:2, :);
    end
    
    N = size(P_world, 2);
    R = R_init;
    t = t_init;
    
    for iter = 1:MAX_ITER
        % 1. P_cam = R*P_w + t
        P_cam = R * P_world + t;
        X = P_cam(1, :);
        Y = P_cam(2, :);
        Z = P_cam(3, :);
        
        Z(abs(Z) < 1e-6) = 1e-6; 
        inv_Z = 1 ./ Z;
        
        % 2. 
        u_proj = X .* inv_Z;
        v_proj = Y .* inv_Z;
        
        % 3.  (Residual)
        res_u = u_proj - pts_obs(1, :);
        res_v = v_proj - pts_obs(2, :);
        residual = [res_u'; res_v'];
        

        if norm(residual) < 1e-6
            break; 
        end
        
        % 4.  (Jacobian) - 2N x 6

        u_invZ = u_proj .* inv_Z;
        v_invZ = v_proj .* inv_Z;
        

        % [1/Z, 0, -u/Z, -u*v, 1+u^2, -v]
        zeros_row = zeros(1, N);
        J_u_trans = [inv_Z; zeros_row; -u_invZ; -u_proj.*v_proj; 1+u_proj.^2; -v_proj];
        
  
        % [0, 1/Z, -v/Z, -1-v^2, u*v, u]
        J_v_trans = [zeros_row; inv_Z; -v_invZ; -1-v_proj.^2; u_proj.*v_proj; u_proj];
        

        % J = [J_u_trans'; J_v_trans']; 
        
        J = [J_u_trans, J_v_trans]'; 
        
        % 5. (Normal Equation: J'J * delta = -J'r)
        
        delta = - (J \ residual);
        
        if norm(delta) < MIN_DELTA
            break;
        end
        
        % 6. 
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
        
        %  T_new = dT * T_old
        % R_new = dR * R
        % t_new = dR * t + d_rho
        R = dR * R;
        t = dR * t + d_rho;
    end
    
    R_opt = R;
    t_opt = t;
end
