% Ablation2.m
% compare BAPnP, EPnP (Standard),  EPnP (with Greedy Base Selection)
clear; clc; close all;

addpath(genpath('EPnP')); 


npts = 20;                  
noise_levels = 0:0.5:5;     
num_trials = 1000;          


avg_rot_err   = zeros(3, length(noise_levels)); 
avg_trans_err = zeros(3, length(noise_levels));

fprintf(' (N=%d)...\n', npts);
fprintf('-------------------------------------------------------------------------------------------------------\n');
fprintf('| Noise | BAPnP Rot(°) | EPnP(Std) Rot(°) | EPnP+Greedy Rot(°) | BAPnP Tr(%%) | EPnP(Std) Tr(%%) | EPnP+Greedy Tr(%%) |\n');
fprintf('-------------------------------------------------------------------------------------------------------\n');


for i = 1:length(noise_levels)
    noise = noise_levels(i);
    

    stats = zeros(3, 2); 
    counts = zeros(3, 1);
    
    for k = 1:num_trials

        try
            [pts3d, ~, normalized_pts2d_noisy, ~, R_true, t_true] = ...
                generate_P6P_3D_to_2D_point_correspondences_noise(npts, noise);
        catch
             error('generate_P6P_3D_to_2D_point_correspondences_noise');
        end
        
        % -------------------------------------------------
        % A. Proposed Algorithm (BAPnP)
        % -------------------------------------------------
        try
            [R_est, t_est] = pnp_linear_only(normalized_pts2d_noisy, pts3d);
            [r_err, t_err] = calc_pose_error(R_est, t_est, R_true, t_true);
            if ~isnan(r_err) && ~isnan(t_err)
                stats(1,1) = stats(1,1) + r_err;
                stats(1,2) = stats(1,2) + t_err;
                counts(1) = counts(1) + 1;
            end
        catch
        end
        
        % -------------------------------------------------
        % B. EPnP Algorithm (Standard PCA Bases)
        % -------------------------------------------------
        try
            [R_est, t_est] = run_epnp(normalized_pts2d_noisy, pts3d);
            [r_err, t_err] = calc_pose_error(R_est, t_est, R_true, t_true);
            if ~isnan(r_err) && ~isnan(t_err)
                stats(2,1) = stats(2,1) + r_err;
                stats(2,2) = stats(2,2) + t_err;
                counts(2) = counts(2) + 1;
            end
        catch
        end

        % -------------------------------------------------
        % C. EPnP + Our Greedy Strategy (The New Curve)
        % -------------------------------------------------
        try

            cpts_greedy = select_greedy_bases(pts3d);
            

            [R_est, t_est] = run_epnp_with_cpts(normalized_pts2d_noisy, pts3d, cpts_greedy);
            
            [r_err, t_err] = calc_pose_error(R_est, t_est, R_true, t_true);
            if ~isnan(r_err) && ~isnan(t_err)
                stats(3,1) = stats(3,1) + r_err;
                stats(3,2) = stats(3,2) + t_err;
                counts(3) = counts(3) + 1;
            end
        catch
            % warning('EPnP+Greedy failed');
        end
    end
    

    for m = 1:3
        if counts(m) > 0
            avg_rot_err(m, i)   = stats(m,1) / counts(m);
            avg_trans_err(m, i) = stats(m,2) / counts(m);
        else
            avg_rot_err(m, i) = NaN; avg_trans_err(m, i) = NaN;
        end
    end
    
    fprintf('| %4.1f  |    %6.3f    |      %6.3f      |      %6.3f      |    %6.3f    |     %6.3f      |      %6.3f       |\n', ...
        noise, avg_rot_err(1,i), avg_rot_err(2,i), avg_rot_err(3,i), ...
        avg_trans_err(1,i), avg_trans_err(2,i), avg_trans_err(3,i));
end
fprintf('-------------------------------------------------------------------------------------------------------\n');


figure('Color', 'w', 'Position', [100, 400, 600, 400]);
hold on;
plot(noise_levels, avg_rot_err(1, :), '-ro', 'LineWidth', 2, 'MarkerFaceColor', 'r', 'DisplayName', 'BAPnP (Ours)');
plot(noise_levels, avg_rot_err(2, :), '-bs', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'DisplayName', 'EPnP (Std PCA)');
plot(noise_levels, avg_rot_err(3, :), '-g^', 'LineWidth', 2, 'MarkerFaceColor', 'g', 'DisplayName', 'EPnP + Our Selection');
xlabel('Noise Level (pixels)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Rotation Error (degrees)', 'FontSize', 12, 'FontWeight', 'bold');
title(['Rotation Error vs Noise (N=', num2str(npts), ')'], 'FontSize', 14);
grid on; legend('Location', 'NorthWest');
hold off;


figure('Color', 'w', 'Position', [750, 400, 600, 400]);
hold on;
plot(noise_levels, avg_trans_err(1, :), '-ro', 'LineWidth', 2, 'MarkerFaceColor', 'r', 'DisplayName', 'BAPnP (Ours)');
plot(noise_levels, avg_trans_err(2, :), '-bs', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'DisplayName', 'EPnP (Std PCA)');
plot(noise_levels, avg_trans_err(3, :), '-g^', 'LineWidth', 2, 'MarkerFaceColor', 'g', 'DisplayName', 'EPnP + Our Selection');
xlabel('Noise Level (pixels)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Translation Error (%)', 'FontSize', 12, 'FontWeight', 'bold');
title(['Translation Error vs Noise (N=', num2str(npts), ')'], 'FontSize', 14);
grid on; legend('Location', 'NorthWest');
hold off;

%% =========================================================================
%  Helper Functions
% =========================================================================

function [r_err_deg, t_err_percent] = calc_pose_error(R_est, t_est, R_true, t_true)
   
    R_diff = R_true' * R_est;
    val = (trace(R_diff) - 1) / 2;
    if val > 1, val = 1; end
    if val < -1, val = -1; end
    r_err_deg = rad2deg(acos(val));
    
    
    t_diff = norm(t_est - t_true);
    t_norm = norm(t_true);
    if t_norm < 1e-6, t_err_percent = 0; else, t_err_percent = (t_diff / t_norm) * 100; end
end

function cpts = select_greedy_bases(pts3d)

    N = size(pts3d, 2);
    used_indices = false(1, N);
    

    centroid = mean(pts3d, 2);
    

    dists = sum((pts3d - centroid).^2, 1);
    [~, idx1] = max(dists);
    Pb1 = pts3d(:, idx1);
    used_indices(idx1) = true;
    

    dists = sum((pts3d - Pb1).^2, 1);
    dists(used_indices) = -1; 
    [~, idx2] = max(dists);
    Pb2 = pts3d(:, idx2);
    used_indices(idx2) = true;
    

    u = Pb2 - Pb1; u = u / norm(u);
    max_dist = -1; idx3 = -1;
    for k = 1:N
        if used_indices(k), continue; end
        P = pts3d(:, k);
        v = P - Pb1;
        dist_sq = norm(cross(u, v))^2; 
        if dist_sq > max_dist, max_dist = dist_sq; idx3 = k; end
    end
    Pb3 = pts3d(:, idx3);
    used_indices(idx3) = true;
    

    v1_vec = Pb2 - Pb1; v2_vec = Pb3 - Pb1;
    n = cross(v1_vec, v2_vec); n = n / norm(n);
    max_dist = -1; idx4 = -1;
    for k = 1:N
        if used_indices(k), continue; end
        P = pts3d(:, k);
        dist = abs(dot(n, P - Pb1));
        if dist > max_dist, max_dist = dist; idx4 = k; end
    end
    Pb4 = pts3d(:, idx4);
    
    cpts = [Pb1, Pb2, Pb3, Pb4]'; % 返回 4x3 格式，符合 EPnP 习惯
end

function [R, t] = run_epnp_with_cpts(pts2d, pts3d, cpts)

    n = size(pts3d, 2);

    x3d_h = [pts3d', ones(n,1)];     % Nx4
    x2d_h = [pts2d', ones(n,1)];     % Nx3
    A_eye = eye(3);                  
    
    [R, t, ~, ~] = efficient_pnp_custom_cpts(x3d_h, x2d_h, A_eye, cpts);
end

% =========================================================================
% Modified EPnP Function (Supports Custom Control Points)
% Based on: cvlab-epfl/epnp/efficient_pnp.m
% =========================================================================
function [R,T,Xc,best_solution]=efficient_pnp_custom_cpts(x3d_h,x2d_h,A,Cw_custom)

    Xw=x3d_h(:,1:3);
    U=x2d_h(:,1:2);
    THRESHOLD_REPROJECTION_ERROR=20;

    % CHANGE 1: Use custom control points directly
    Cw = Cw_custom; 

    % CHANGE 2: Compute distances dsq based on custom Cw
    % define_distances_btw_control_points returns: [d12,d13,d14,d23,d24,d34]'
    c1=Cw(1,:); c2=Cw(2,:); c3=Cw(3,:); c4=Cw(4,:);
    dsq = [sum((c1-c2).^2); sum((c1-c3).^2); sum((c1-c4).^2); ...
           sum((c2-c3).^2); sum((c2-c4).^2); sum((c3-c4).^2)];

    % The rest is standard EPnP logic
    Alph=compute_alphas(Xw,Cw);
    M=compute_M_ver2(U,Alph,A);
    Km=kernel_noise(M,4); 
        
    % 1. dim(ker)=1
    dim_kerM=1;
    X1=Km(:,end);
    [Cc,Xc]=compute_norm_sign_scaling_factor(X1,Cw,Alph,Xw);
    [R,T]=getrotT(Xw,Xc); 
    err(1)=reprojection_error_usingRT(Xw,U,R,T,A);
    sol(1).Xc=Xc; sol(1).Cc=Cc; sol(1).R=R; sol(1).T=T; sol(1).error=err(1);

    % 2. dim(ker)=2
    Km1=Km(:,end-1); Km2=Km(:,end);
    D=compute_constraint_distance_2param_6eq_3unk(Km1,Km2);
    betas_=inv(D'*D)*D'*dsq; 
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
    Xc=sol(best_solution).Xc;
    R=sol(best_solution).R;
    T=sol(best_solution).T;
end




