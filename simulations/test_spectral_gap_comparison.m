function test_spectral_gap_comparison.()
%
% Purpose:
%   Compare the spectral gap of BAPnP, EPnP, CPnP AND RPnP under
%   quasi-planar (near-coplanar) configurations.
%
% Spectral Gap Definitions:
%   - BAPnP (Rank 3):   sigma_3  / sigma_1
%   - EPnP  (Rank 11):  sigma_11 / sigma_1
%   - CPnP  (Rank 11):  sigma_11 / sigma_1
%   - RPnP  (Rank 5):   sigma_5  / sigma_1 
%     (RPnP refines a 6-variable linear system [c,s,tx,ty,tz,1]^T.
%      Effective rank is 5, null space dim is 1).

    clc; clear; close all;

    %% 1. Parameters
    n_points = 20;
    n_trials = 1000;
    % Degree of planarity: from general 3D to extremely planar
    gammas = logspace(-1, -12, 10); 
    
    gap_bapnp = zeros(length(gammas), 1);
    gap_epnp  = zeros(length(gammas), 1);
    gap_cpnp  = zeros(length(gammas), 1);
    gap_rpnp  = zeros(length(gammas), 1); 
    
    fprintf('Running Spectral Gap Analysis (BAPnP vs EPnP vs CPnP vs RPnP)...\n');
    fprintf('%-10s | %-10s | %-10s | %-10s | %-10s\n', 'Gamma', 'BAPnP', 'EPnP', 'CPnP', 'RPnP');
    fprintf('-----------------------------------------------------------------------------\n');

    %% 2. Main Loop
    for i = 1:length(gammas)
        gamma = gammas(i);
        g_b = [];
        g_e = [];
        g_c = [];
        g_r = []; % RPnP gaps
        
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
            
            %% B. Data Normalization
            cent = mean(Pw_raw, 2);
            P_centered = Pw_raw - cent;
            scale = sqrt(3) / mean(sqrt(sum(P_centered.^2, 1)));
            Pw_norm = P_centered * scale; 
            
            % For RPnP matrix proxy, we need consistency, so we use Pw_norm
            % and adjust the Ground Truth R/T conceptually, or just use
            % the geometric structure.
            % BAPnP uses unit sphere bearing vectors
            y_sphere = [y; ones(1,n_points)];
            norms = sqrt(sum(y_sphere.^2, 1));
            y_sphere = y_sphere ./ norms;
            
            % EPnP/CPnP: Homogeneous coordinates
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
            
            % 3. CPnP
            Pesi = get_cpnp_matrix(y, Pw_norm); 
            s = svd(Pesi);
            if length(s) >= 11, g_c = [g_c; s(11)/s(1)]; end
            
            % 4. RPnP (New)
            % RPnP linear matrix depends on the correct rotation hypothesis.
            % We use R_gt to construct the "Ideal" D matrix that RPnP would solve.
            D_rpnp = get_rpnp_matrix(y, Pw_norm, R_gt);
            s = svd(D_rpnp);
            % RPnP solves for 5 variables (c, s, tx, ty, tz) with homogenization
            % Rank should be 5.
            if length(s) >= 6, g_r = [g_r; s(5)/s(1)]; end
        end
        
        gap_bapnp(i) = median(g_b);
        gap_epnp(i)  = median(g_e);
        gap_cpnp(i)  = median(g_c);
        gap_rpnp(i)  = median(g_r);
        
        fprintf('%.1e    | %.4e   | %.4e   | %.4e   | %.4e\n', ...
            gamma, gap_bapnp(i), gap_epnp(i), gap_cpnp(i), gap_rpnp(i));
    end

    %% 3. Visualization
    figure('Color', 'w', 'Position', [300, 300, 800, 500]);
    
    loglog(gammas, gap_bapnp, '-o', 'Color', [0.85, 0.33, 0.1], ...
        'LineWidth', 2, 'MarkerFaceColor', [0.85, 0.33, 0.1], 'DisplayName', 'BAPnP');
    hold on;
    loglog(gammas, gap_epnp, '--s', 'Color', [0, 0.45, 0.74], ...
        'LineWidth', 2, 'MarkerFaceColor', [0, 0.45, 0.74], 'DisplayName', 'EPnP');
    loglog(gammas, gap_cpnp, '-.d', 'Color', [0.47, 0.67, 0.19], ...
        'LineWidth', 2, 'MarkerFaceColor', [0.47, 0.67, 0.19], 'DisplayName', 'CPnP');
    loglog(gammas, gap_rpnp, '-^', 'Color', [0.49, 0.18, 0.56], ...
        'LineWidth', 2, 'MarkerFaceColor', [0.49, 0.18, 0.56], 'DisplayName', 'RPnP');
    
    set(gca, 'XDir', 'reverse'); 
    xlabel('Degree of Coplanarity (\gamma)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Normalized Spectral Gap (\sigma_{min}/\sigma_{max})', 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'southwest', 'FontSize', 11);
    ylim([1e-6, 1.5]); 
end

%% ========================================================================
%  RPnP Matrix Construction (Proxy)
% ========================================================================
function D = get_rpnp_matrix(xx, XX, R_gt)
    % Inputs:
    %   xx: 2xN normalized image coordinates
    %   XX: 3xN 3D points
    %   R_gt: Ground Truth Rotation (World -> Camera)
    %
    % This function reconstructs the 'D' matrix used in the RPnP
    % linear refinement step, assuming the correct local frame is found.

    n = size(xx, 2);
    
    % 1. Edge Selection (Similar to RPnP.m)
    % Compute normalized image rays
    xxv = [xx; ones(1,n)];
    for i=1:n
        xxv(:,i) = xxv(:,i) / norm(xxv(:,i));
    end
    
    % Use the heuristic from RPnP: pick edge with min dot product (max angle)
    % We do a quick search similar to original code
    i1 = 1; i2 = 2;
    lmin = 1.0; 
    
    % Random sampling for edge selection (deterministic for this analysis?)
    % To be fair, we use a fixed number of samples or just find the min
    % brute-force for stability analysis. Let's do brute-force for 'ideal' gap.
    % (Or stick to RPnP's random sampling). Let's use RPnP's random method.
    samp_count = n; 
    rij = ceil(rand(samp_count,2)*n);
    
    for k = 1:samp_count
        i = rij(k,1); j = rij(k,2);
        if i == j, continue; end
        l = dot(xxv(:,i), xxv(:,j));
        if l < lmin
            lmin = l; i1 = i; i2 = j;
        end
    end
    
    % 2. Define Local Coordinate System (Edge Frame)
    p1 = XX(:, i1);
    p2 = XX(:, i2);
    p0 = (p1 + p2) / 2;
    x_axis = p2 - p0; 
    x_axis = x_axis / norm(x_axis);
    
    if abs([0 1 0]*x_axis) < abs([0 0 1]*x_axis)
        z_axis = cross(x_axis, [0; 1; 0]); z_axis = z_axis / norm(z_axis);
        y_axis = cross(z_axis, x_axis);    y_axis = y_axis / norm(y_axis);
    else
        y_axis = cross([0; 0; 1], x_axis); y_axis = y_axis / norm(y_axis);
        z_axis = cross(x_axis, y_axis);    z_axis = z_axis / norm(z_axis);
    end
    Ro = [x_axis, y_axis, z_axis]; % Rotation World -> Edge Frame
    
    % Transform World points to Local Edge Frame
    XX_local = Ro.' * (XX - repmat(p0, 1, n));
    
    % 3. Calculate "Rx" (Rotation Camera -> Edge Frame)
    % R_gt transforms World -> Camera.
    % We need the rotation that transforms Local -> Camera.
    % P_cam = R_gt * P_world + T
    % P_world = Ro * P_local + p0
    % P_cam = R_gt * (Ro * P_local + p0) + T
    % P_cam = (R_gt * Ro) * P_local + (R_gt * p0 + T)
    % So the Rotation R in RPnP's local problem is (R_gt * Ro).
    
    R_local = R_gt * Ro;
    
    % RPnP uses 'Rx' such that columns are x,y,z axes.
    % In RPnP code: Rx = [x y z]. 
    % And equations utilize r = Rx.'. 
    % Effectively Rx matches R_local.
    Rx = R_local; 
    r = Rx.'; % Transpose for the equation form
    
    % 4. Construct Matrix D
    % D * [c s tx ty tz 1]^T = 0
    D = zeros(2*n, 6);
    
    for j = 1:n
        ui = xx(1,j); 
        vi = xx(2,j);
        xi = XX_local(1,j); 
        yi = XX_local(2,j); 
        zi = XX_local(3,j);
        
        % Equations from RPnP.m (inside loop for i=1:m)
        % Note: RPnP variables are V1(1)=c, V1(2)=s.
        % The matrix columns correspond to: [c, s, t(1), t(2), t(3), const]
        
        % Row 1 (u-coordinate constraint)
        % -r(2)*yi + ui*(r(8)*yi+r(9)*zi) - r(3)*zi
        term_c = -r(2)*yi + ui*(r(8)*yi + r(9)*zi) - r(3)*zi; % Multiplies c
        term_s = -r(3)*yi + ui*(r(9)*yi - r(8)*zi) + r(2)*zi; % Multiplies s
        term_t1 = -1;
        term_t2 = 0;
        term_t3 = ui;
        term_const = ui*r(7)*xi - r(1)*xi;
        
        D(2*j-1, :) = [term_c, term_s, term_t1, term_t2, term_t3, term_const];
        
        % Row 2 (v-coordinate constraint)
        % -r(5)*yi + vi*(r(8)*yi+r(9)*zi) - r(6)*zi
        term_c = -r(5)*yi + vi*(r(8)*yi + r(9)*zi) - r(6)*zi;
        term_s = -r(6)*yi + vi*(r(9)*yi - r(8)*zi) + r(5)*zi;
        term_t1 = 0;
        term_t2 = -1;
        term_t3 = vi;
        term_const = vi*r(7)*xi - r(4)*xi;
        
        D(2*j, :) = [term_c, term_s, term_t1, term_t2, term_t3, term_const];
    end
end

%% ========================================================================
%  Helper Functions (Existing CPnP/BAPnP/EPnP matrices)
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

function L = get_bapnp_matrix(y_norm, P_n)
    N = size(P_n, 2);
    base_idx = greedy_selection(P_n);
    perm = [base_idx, setdiff(1:N, base_idx)];
    P_sorted = P_n(:, perm);
    y_sorted = y_norm(:, perm);
    Pb = P_sorted(:, 1:4);
    Basis = Pb(:, 1:3) - Pb(:, 4);
    Target = P_sorted(:, 5:end) - Pb(:, 4);
    coeffs = pinv(Basis) * Target;
    alphas = coeffs(1,:); betas = coeffs(2,:); gammas_c = coeffs(3,:);
    deltas = 1 - sum(coeffs, 1);
    L = zeros(3*(N-4), 4);
    y1=y_sorted(:,1); y2=y_sorted(:,2); y3=y_sorted(:,3); y4=y_sorted(:,4);
    y_others = y_sorted(:, 5:end);
    for j = 1:(N-4)
        yj = y_others(:, j);
        yx = [0 -yj(3) yj(2); yj(3) 0 -yj(1); -yj(2) yj(1) 0];
        row_idx = (j-1)*3 + 1 : j*3;
        L(row_idx, 1) = alphas(j)    * (yx * y1);
        L(row_idx, 2) = betas(j)     * (yx * y2);
        L(row_idx, 3) = gammas_c(j) * (yx * y3);
        L(row_idx, 4) = deltas(j)    * (yx * y4);
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
