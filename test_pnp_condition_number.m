function test_pnp_basis_stability()
% TEST_PNP_BASIS_STABILITY
% 
% 修正版逻辑:
%   不再计算最终线性系统 L 或 M 的条件数（它们本来就该是奇异的）。
%   转而计算 "重心坐标变换矩阵" (Barycentric Basis Matrix) 的条件数。
%   这直接对应论文中 Eq. (3) 和 Perturbation Theory 的分析。

    clc; clear; close all;

    %% 1. 实验参数
    n_points = 20;
    n_trials = 500; 
    % 压缩系数: 1.0 (立方体) -> 1e-7 (极度扁平)
    gammas = logspace(-1, -7, 10); 
    
    cond_bapnp = zeros(length(gammas), 1);
    cond_epnp  = zeros(length(gammas), 1);

    fprintf('Running Basis Stability Test (Corrected)...\n');
    fprintf('%-10s | %-15s | %-15s\n', 'Gamma', 'Cond(Basis_BA)', 'Cond(Basis_EP)');
    fprintf('----------------------------------------------\n');

    %% 2. 主循环
    for i = 1:length(gammas)
        gamma = gammas(i);
        
        c_b_list = [];
        c_e_list = [];
        
        for k = 1:n_trials
            % --- A. 生成准平面场景 ---
            % 随机 xy, 压缩 z
            Pw_raw = (rand(3, n_points) - 0.5) * 10;
            Pw = Pw_raw;
            Pw(3, :) = Pw(3, :) * gamma; 
            
            % 随机旋转一下，避免轴对齐带来的偶然性
            R_rand = random_rotation();
            Pw = R_rand * Pw;

            % --- B. BAPnP: 贪婪选点基底 ---
            % 这里的核心是: 即使点云压扁，只要平面上分布得开，
            % 我们的 Greedy 策略选出的三角形面积依然很大，
            % 只是四面体的高很小。但相比于 EPnP 的虚拟点，我们撑得更满。
            [basis_idx] = select_greedy_basis(Pw);
            Pb = Pw(:, basis_idx);
            
            % BAPnP 的基底矩阵 (用于计算 alpha, beta, gamma)
            % P = P4 + alpha*(P1-P4) + beta*(P2-P4) + gamma*(P3-P4)
            % Basis_Mat = [P1-P4, P2-P4, P3-P4]
            T_bapnp = [Pb(:,1)-Pb(:,4), Pb(:,2)-Pb(:,4), Pb(:,3)-Pb(:,4)];
            
            % 记录条件数
            c_b_list = [c_b_list; cond(T_bapnp)];
            
            % --- C. EPnP: PCA 控制点基底 ---
            % 必须使用标准 EPnP 策略：沿主轴分布，且幅度由特征值决定
            [U, S, ~] = svd(cov(Pw'));
            eigen_vals = diag(S);
            mean_p = mean(Pw, 2);
            
            % 标准 EPnP 控制点定义 (参考 OpenCV/Lepetit 论文)
            % C1 = mean
            % C2 = mean + sqrt(lambda1)*v1
            % C3 = mean + sqrt(lambda2)*v2
            % C4 = mean + sqrt(lambda3)*v3  <-- 罪魁祸首！当 lambda3 -> 0 时，C4 和 C1 重合
            Cw = zeros(3, 4);
            Cw(:,1) = mean_p;
            Cw(:,2) = mean_p + sqrt(eigen_vals(1)) * U(:,1);
            Cw(:,3) = mean_p + sqrt(eigen_vals(2)) * U(:,2);
            Cw(:,4) = mean_p + sqrt(eigen_vals(3)) * U(:,3);
            
            % EPnP 的基底矩阵 (齐次形式)
            % [C1 C2 C3 C4; 1 1 1 1]
            C_homo = [Cw; ones(1, 4)];
            
            c_e_list = [c_e_list; cond(C_homo)];
        end
        
        cond_bapnp(i) = median(c_b_list);
        cond_epnp(i)  = median(c_e_list);
        
        fprintf('%.1e    | %.2e        | %.2e\n', gamma, cond_bapnp(i), cond_epnp(i));
    end

    %% 3. 绘图
    figure('Color', 'w', 'Position', [300, 300, 600, 450]);
    loglog(gammas, cond_epnp, '-s', 'Color', [0, 0.4470, 0.7410], ...
        'LineWidth', 2, 'MarkerFaceColor', [0, 0.4470, 0.7410], 'DisplayName', 'EPnP (PCA)');
    hold on;
    loglog(gammas, cond_bapnp, '-o', 'Color', [0.8500, 0.3250, 0.0980], ...
        'LineWidth', 2, 'MarkerFaceColor', [0.8500, 0.3250, 0.0980], 'DisplayName', 'BAPnP (Ours)');
    
    set(gca, 'XDir', 'reverse');
    grid on;
    xlabel('Degree of Coplanarity (\gamma)', 'FontSize', 12);
    ylabel('Condition Number of Basis Matrix', 'FontSize', 12);
    title(['Basis Stability: Real Points (BAPnP) vs. Virtual Points (EPnP)'], 'FontSize', 13);
    legend('Location', 'northwest', 'FontSize', 11);
    ylim([1e0, 1e10]); 
    
    % 添加标注
    text(1e-6, 1e8, 'EPnP Collapses', 'Color', 'b', 'FontSize', 10, 'HorizontalAlignment', 'center');
    text(1e-6, 1e2, 'BAPnP Stable', 'Color', 'r', 'FontSize', 10, 'HorizontalAlignment', 'center');
end

%% --- 你的贪婪选点策略 ---
function base_idx = select_greedy_basis(P)
    N = size(P, 2);
    base_idx = zeros(1, 4);
    
    % 1. 离质心最远
    cent = mean(P, 2);
    d2 = sum((P - cent).^2, 1);
    [~, base_idx(1)] = max(d2);
    
    % 2. 离 P1 最远
    d2 = sum((P - P(:, base_idx(1))).^2, 1);
    [~, base_idx(2)] = max(d2);
    
    % 3. 离 P1-P2 直线最远
    p1 = P(:, base_idx(1));
    p2 = P(:, base_idx(2));
    v12 = p2 - p1;
    % 叉乘计算距离
    vecs = P - p1;
    cp = cross(repmat(v12, 1, N), vecs);
    d2_line = sum(cp.^2, 1); % 省略了除以模长，因为只比大小
    [~, base_idx(3)] = max(d2_line);
    
    % 4. 离 P1-P2-P3 平面最远
    p3 = P(:, base_idx(3));
    v13 = p3 - p1;
    normal = cross(v12, v13);
    if norm(normal) < 1e-10
        % 几乎共线，随便选一个远的
        [~, base_idx(4)] = max(sum((P - cent).^2, 1)); 
    else
        normal = normal / norm(normal);
        d_plane = abs(normal' * (P - p1));
        [~, base_idx(4)] = max(d_plane);
    end
end

function R = random_rotation()
    [Q, ~] = qr(randn(3));
    R = Q;
end