function compare_condition_numbers()
    clc; clear; close all;
    
    n_points = 20;
    n_trials = 100;
    
    % Planarity factor: 1.0 (cube) -> 1e-7 (flat plane)
    gammas = logspace(-1, -7, 10); 
    
    mean_cond_bapnp = zeros(length(gammas), 1);
    mean_cond_epnp = zeros(length(gammas), 1);
    
    for k = 1:length(gammas)
        gamma = gammas(k);
        conds_b = [];
        conds_e = [];
        
        for t = 1:n_trials
            % 1. Generate 3D points (Quasi-planar)
            XX = (rand(n_points, 1) - 0.5) * 10;
            YY = (rand(n_points, 1) - 0.5) * 10;
            ZZ = (rand(n_points, 1) - 0.5) * 10 * gamma; % Compress Z
            P_w = [XX, YY, ZZ];
            
            % 2. Generate Camera & Projection (Standard setup)
            R = rodrigues(randn(3,1)); 
            T = [0; 0; 20];
            P_c = (R * P_w' + T)';
            uv = P_c(:, 1:2) ./ P_c(:, 3); % Normalized image plane
            
            % --- Method A: BAPnP Strategy (Greedy Real Points) ---
            % Simplified implementation of constructing Matrix L
            [base_indices] = select_greedy_basis(P_w); % You have this function
            Pb = P_w(base_indices, :);
            % Compute barycentric coords for all points
            % Construct L matrix (3*(N-4) x 4)
            L_bapnp = construct_L_matrix_BAPnP(P_w, uv, base_indices);
            conds_b = [conds_b; cond(L_bapnp)];
            
            % --- Method B: EPnP Strategy (PCA Virtual Points) ---
            % Compute PCA control points
            [~, ~, V] = svd(cov(P_w));
            mean_P = mean(P_w);
            % EPnP chooses 4 points along axes... 
            % Construct L matrix for EPnP (standard EPnP L matrix)
            % (Assuming you have a function to build EPnP's Mx12 matrix)
            L_epnp = construct_L_matrix_EPnP(P_w, uv); 
            conds_e = [conds_e; cond(L_epnp)];
        end
        
        mean_cond_bapnp(k) = mean(conds_b);
        mean_cond_epnp(k) = mean(conds_e);
    end
    
    % Plotting
    figure; 
    semilogx(gammas, log10(mean_cond_epnp), '-s', 'LineWidth', 2, 'DisplayName', 'EPnP (PCA)');
    hold on;
    semilogx(gammas, log10(mean_cond_bapnp), '-o', 'LineWidth', 2, 'DisplayName', 'BAPnP (Greedy)');
    set(gca, 'XDir', 'reverse'); % Right is flat
    xlabel('Degree of Coplanarity (\gamma)');
    ylabel('Log10 Condition Number of Matrix L');
    grid on;
    legend;
    title('Numerical Stability: Condition Number Analysis');
end

% Note: You need to fill in your specific construct_L functions 
% to make this run with your exact implementation.
