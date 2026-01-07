function test_spectral_gap_comparison()
% TEST_SPECTRAL_GAP_COMPARISON
%
% Purpose:
%   Compare the spectral gap of BAPnP, EPnP and CPnP under
%   quasi-planar (near-coplanar) configurations.
%
% Spectral Gap Definition (Normalized):
%   - BAPnP (4 variables):  sigma_3  / sigma_1  (effective rank = 3)
%   - EPnP  (12 variables): sigma_11 / sigma_1  (effective rank = 11)
%   - CPnP  (11 variables): sigma_11 / sigma_1  (full-rank check)
%
% The spectral gap reflects the numerical stability of the
% underlying linear system.
    clc; clear; close all;

   %% 1. Parameters
    n_points = 20;
    n_trials = 100;
    % Degree of planarity: from general 3D to extremely planar
    gammas = logspace(-1, -10, 10); 
    
    gap_bapnp = zeros(length(gammas), 1);
    gap_epnp  = zeros(length(gammas), 1);
    gap_cpnp  = zeros(length(gammas), 1);
    
    fprintf('(BAPnP vs EPnP vs CPnP)...\n');
    fprintf('%-10s | %-12s | %-12s | %-12s\n', 'Gamma', 'Gap(BAPnP)', 'Gap(EPnP)', 'Gap(CPnP)');
    fprintf('-------------------------------------------------------------\n');

     %% 2. Main Loop
    for i = 1:length(gammas)
        gamma = gammas(i);
        g_b = [];
        g_e = [];
        g_c = [];
        
        for k = 1:n_trials
            %% A. Scene Generation

            % Generate 3D points (world frame)
            Pw_raw = (rand(3, n_points) - 0.5) * 10;
            Pw_raw(3, :) = Pw_raw(3, :) * gamma; % Flatten Z-axis
            
             % Transform to camera coordinates
            R_gt = random_rotation();
            T_gt = [0; 0; 20];
            Pc = R_gt * Pw_raw + T_gt;
            
             % Normalized image projection (fx = fy = 1)
            y = Pc(1:2, :) ./ Pc(3, :);
            
           %% B. Data Normalization (for fair comparison)
            cent = mean(Pw_raw, 2);
            P_centered = Pw_raw - cent;
            scale = sqrt(3) / mean(sqrt(sum(P_centered.^2, 1)));
            Pw_norm = P_centered * scale; 
            
            % BAPnP uses unit sphere bearing vectors
            y_sphere = [y; ones(1,n_points)];
            norms = sqrt(sum(y_sphere.^2, 1));
            y_sphere = y_sphere ./ norms;
            
            % EPnP/CPnP: Homogeneous / 2D coordinates
            y_homo = [y; ones(1,n_points)];
            
           %% C. Spectral Gap Computation
            
            % 1. BAPnP
            L = get_bapnp_matrix(y_sphere, Pw_norm); 
            s = svd(L);
            if length(s) >= 4, g_b = [g_b; s(3)/s(1)]; end
            
            % 2. EPnP
            M = get_epnp_matrix(y_homo, Pw_norm);
            s = svd(M);
            if length(s) >= 12, g_e = [g_e; s(11)/s(1)]; end
            
            % ---- CPnP ----
            Pesi = get_cpnp_matrix(y, Pw_norm); 
            s = svd(Pesi);
            
            if length(s) >= 11, g_c = [g_c; s(11)/s(1)]; end
        end
        
        gap_bapnp(i) = median(g_b);
        gap_epnp(i)  = median(g_e);
        gap_cpnp(i)  = median(g_c);
        
        fprintf('%.1e    | %.4e     | %.4e     | %.4e\n', gamma, gap_bapnp(i), gap_epnp(i), gap_cpnp(i));
    end

    %% 3. Visualization
    figure('Color', 'w', 'Position', [300, 300, 700, 500]);
    
    loglog(gammas, gap_bapnp, '-o', 'Color', [0.85, 0.33, 0.1], ...
        'LineWidth', 2, 'MarkerFaceColor', [0.85, 0.33, 0.1], 'DisplayName', 'BAPnP (Ours)');
    hold on;
    loglog(gammas, gap_epnp, '--s', 'Color', [0, 0.45, 0.74], ...
        'LineWidth', 2, 'MarkerFaceColor', [0, 0.45, 0.74], 'DisplayName', 'EPnP');
    loglog(gammas, gap_cpnp, '-.d', 'Color', [0.47, 0.67, 0.19], ...
        'LineWidth', 2, 'MarkerFaceColor', [0.47, 0.67, 0.19], 'DisplayName', 'CPnP');
    
    set(gca, 'XDir', 'reverse'); 
    grid on;
    xlabel('Degree of Coplanarity (\gamma)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Normalized Spectral Gap (\sigma_{min}/\sigma_{max})', 'FontSize', 12, 'FontWeight', 'bold');
    title('Linear Solver Stability vs. Planarity', 'FontSize', 14);
    legend('Location', 'southwest', 'FontSize', 11);
    ylim([1e-5, 1]); 

end

%% ========================================================================
%   CPnP Matrix Construction 
% ========================================================================
function pesi = get_cpnp_matrix(Psens_2D, s)

    
    N = size(s, 2);
    fx = 1; fy = 1; u0 = 0; v0 = 0;
    
   
    bar_s = sum(s, 2) / N; 
    
  
    Psens_2D = Psens_2D - [u0; v0];
    

    obs = Psens_2D(:);  
    
    pesi = zeros(2*N, 11); 
    
 
    for k = 1:N
     
        sk = s(:, k);
        uk = obs(2*k-1);
        vk = obs(2*k);
        
        diff_s = sk - bar_s;
        

        pesi(2*k-1, :) = [ ...
            -diff_s(1)*uk, -diff_s(2)*uk, -diff_s(3)*uk, ... 
            fx*sk(1),      fx*sk(2),      fx*sk(3),      ... 
            fx,            0,             0,             0, 0];
        

        pesi(2*k, :) = [ ...
            -diff_s(1)*vk, -diff_s(2)*vk, -diff_s(3)*vk, ...
            0,             0,             0,             ...
            0,             fy*sk(1),      fy*sk(2),      fy*sk(3), fy];
    end
end

%% ========================================================================
%  BAPnP Matrix Construction
% ========================================================================
function L = get_bapnp_matrix(y_norm, P_n)
    N = size(P_n, 2);
    base_idx = greedy_selection(P_n);
    perm = [base_idx, setdiff(1:N, base_idx)];
    P_sorted = P_n(:, perm);
    y_sorted = y_norm(:, perm);
    
    Pb = P_sorted(:, 1:4);
    Basis = Pb(:, 1:3) - Pb(:, 4);
    Target = P_sorted(:, 5:end) - Pb(:, 4);
    coeffs = pinv(Basis) * Target; % Robust inverse
    
    alphas = coeffs(1,:); betas = coeffs(2,:); gammas_c = coeffs(3,:);
    deltas = 1 - sum(coeffs, 1);
    
    L = zeros(3*(N-4), 4);
    y1=y_sorted(:,1); y2=y_sorted(:,2); y3=y_sorted(:,3); y4=y_sorted(:,4);
    y_others = y_sorted(:, 5:end);
    
    for j = 1:(N-4)
        yj = y_others(:, j);
        yx = [0 -yj(3) yj(2); yj(3) 0 -yj(1); -yj(2) yj(1) 0];
        row_idx = (j-1)*3 + 1 : j*3;
        L(row_idx, 1) = alphas(j)   * (yx * y1);
        L(row_idx, 2) = betas(j)    * (yx * y2);
        L(row_idx, 3) = gammas_c(j) * (yx * y3);
        L(row_idx, 4) = deltas(j)   * (yx * y4);
    end
end

function idx = greedy_selection(P)
    N = size(P, 2);
    idx = zeros(1, 4);
    d2 = sum(P.^2, 1); [~, idx(1)] = max(d2);
    d2 = sum((P - P(:,idx(1))).^2, 1); [~, idx(2)] = max(d2);
    v12 = P(:,idx(2)) - P(:,idx(1));
    v12 = v12 / (norm(v12) + eps);
    cp = cross(repmat(v12,1,N), P - P(:,idx(1)));
    [~, idx(3)] = max(sum(cp.^2, 1));
    v13 = P(:,idx(3)) - P(:,idx(1));
    n = cross(v12, v13); 
    if norm(n) < 1e-8, idx(4) = setdiff(1:N, idx(1:3)); idx(4)=idx(4); 
    else, n=n/norm(n); [~, idx(4)] = max(abs(n' * (P - P(:,idx(1))))); end
end

%% ========================================================================
%  EPnP Matrix Construction
% ========================================================================
function M = get_epnp_matrix(y_homo, P_c)
    N = size(P_c, 2);
    C = P_c * P_c'; [U, S, ~] = svd(C);
    sigma = sqrt(diag(S)); 
    Cw = [zeros(3,1), sigma(1)*U(:,1), sigma(2)*U(:,2), sigma(3)*U(:,3)];
    C_homo = [Cw; ones(1,4)];
    P_homo = [P_c; ones(1,N)];
    alphas_mat = pinv(C_homo) * P_homo;
    M = zeros(2*N, 12);
    uv = y_homo(1:2, :) ./ y_homo(3, :);
    for i = 1:N
        u = uv(1,i); v = uv(2,i); a = alphas_mat(:, i);
        for j = 1:4
            col = (j-1)*3 + 1;
            M(2*i-1, col) = a(j); M(2*i-1, col+2) = -a(j)*u;
            M(2*i, col+1) = a(j); M(2*i, col+2) = -a(j)*v;
        end
    end
end

function R = random_rotation()
    [Q, ~] = qr(randn(3)); R = Q;
end
