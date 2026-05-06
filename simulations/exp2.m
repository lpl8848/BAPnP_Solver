clc; clear; close all;

%% 1. 实验设置
num_trials = 1000;           % 实验次数
fixed_noise = 2.0;          % 固定噪声等级 (推荐 2.0 px)

% 定义点数变化列表 (重点在小点数区域加密)
n_points_list = [6, 8, 10, 12, 15, 20, 30, 50, 80, 100, 200];
num_n_levels = length(n_points_list);

% 定义要画箱线图的具体点数 (代表性的：极少点、中等点、大量点)
boxplot_n_targets = [6, 10, 20, 100]; 

% 定义算法
algorithms = {
    'BAPnP',      @pnp_linear_only;
    'BAPnP-GN',   @BAPnP_new;
    'EPnP-GN',    @run_epnp_guass; 
    'OPnP',       @run_opnp;
    'RPnP',       @run_rpnp;
    'SRPnP-GN',   @run_srpnp;
    'MLPnP',      @run_mlpnp;
    'CPnP-GN',    @run_cpnp;
    'SQPnP',      @sqpnp;
    'oDLT-GN',    @run_odlt_gn;
};
num_algos = size(algorithms, 1);
algo_names = algorithms(:,1);

%% 2. 初始化存储容器
median_rot_err_per_level   = zeros(num_n_levels, num_algos);
median_trans_err_per_level = zeros(num_n_levels, num_algos);
median_repr_err_per_level  = zeros(num_n_levels, num_algos); 

raw_rot_data   = cell(num_n_levels, 1);
raw_trans_data = cell(num_n_levels, 1);
raw_repr_data  = cell(num_n_levels, 1); 
mean_time_per_level = zeros(num_n_levels, num_algos);

fprintf('=== 开始运行实验二: 点数敏感度 (Noise=%.1f px) ===\n', fixed_noise);

%% 3. 主循环
h_wait = []; % headless-compatible, use fprintf instead

for n_idx = 1:num_n_levels
    curr_n = n_points_list(n_idx);
    
    tmp_rot_err   = zeros(num_trials, num_algos);
    tmp_trans_err = zeros(num_trials, num_algos);
    tmp_repr_err  = zeros(num_trials, num_algos); 
    tmp_time      = zeros(num_trials, num_algos);
    
    for i = 1:num_trials
        % 3.1 生成数据
        [pts3d, ~, pts2d_norm, ~, R_gt, t_gt] = ...
            generate_P6P_3D_to_2D_point_correspondences_noise(curr_n, fixed_noise);
            
        % 预处理实际观测点(去齐次化)，保证是 2xN 格式
        if size(pts2d_norm, 1) == 3
            pts2d_obs = pts2d_norm(1:2, :) ./ pts2d_norm(3, :);
        else
            pts2d_obs = pts2d_norm(1:2, :);
        end
        
        % 3.2 遍历算法
        % 3.2 遍历算法
        for j = 1:num_algos
            algo_func = algorithms{j, 2};
            t_start = tic; 
            try
                [R_e, t_e] = algo_func(pts2d_norm, pts3d);
            catch
                R_e = eye(3); t_e = zeros(3,1);
            end
            tmp_time(i, j) = toc(t_start);
            
            % 计算旋转和平移误差
            [r_err, t_err] = calc_error(R_e, t_e, R_gt, t_gt);
            tmp_rot_err(i, j)   = r_err;
            tmp_trans_err(i, j) = t_err;
            
            % ========================================================
            % [终极修复版] 稳健计算重投影误差 (兼容所有崩溃、NaN、维度异常)
            % ========================================================
            best_repr_err = 1000; % 默认惩罚值：1000 (代表算法彻底失效)
            
            try
                % 获取返回的解个数
                if ndims(R_e) == 3
                    num_sols = size(R_e, 3);
                elseif iscell(R_e)
                    num_sols = length(R_e);
                elseif size(R_e, 2) > 3 && mod(size(R_e, 2), 3) == 0
                    num_sols = size(R_e, 2) / 3;
                else
                    num_sols = 1;
                end
                
                for k_sol = 1:num_sols
                    % 提取当前解
                    if ndims(R_e) == 3
                        R_curr = R_e(:, :, k_sol);
                        t_curr = t_e(:, min(k_sol, size(t_e, 2)));
                    elseif iscell(R_e)
                        R_curr = R_e{k_sol};
                        t_curr = t_e{k_sol};
                    elseif size(R_e, 2) > 3
                        R_curr = R_e(:, (k_sol-1)*3+1 : k_sol*3);
                        t_curr = t_e(:, min(k_sol, size(t_e, 2)));
                    else
                        R_curr = R_e;
                        t_curr = t_e(:, 1);
                    end
                    
                    % 1. 严格检查：如果元素个数不对，直接跳过当前解
                    if numel(R_curr) ~= 9 || numel(t_curr) ~= 3
                        continue; 
                    end
                    
                    % 强制重塑为标准尺寸，防万一
                    R_curr = reshape(R_curr, 3, 3);
                    t_curr = reshape(t_curr, 3, 1);
                    
                    % 2. 严格检查：如果包含 NaN 或 Inf，直接跳过
                    if any(isnan(R_curr(:))) || any(isinf(R_curr(:))) || ...
                       any(isnan(t_curr(:))) || any(isinf(t_curr(:)))
                        continue;
                    end
                    
                    % 3. 安全计算投影
                    P_c = R_curr * pts3d + repmat(t_curr, 1, size(pts3d, 2));
                    P_c(3, P_c(3,:) < 1e-6) = 1e-6;        % 避免除零
                    pts2d_proj = P_c(1:2, :) ./ P_c(3, :); 
                    
                    % 计算误差并更新最优解
                    curr_err = mean(sqrt(sum((pts2d_proj - pts2d_obs).^2, 1))); 
                    if curr_err < best_repr_err
                        best_repr_err = curr_err;
                    end
                end
            catch
                % 如果在投影过程中发生了任何不可预见的 MATLAB 错误，直接捕获
                % best_repr_err 将保持为 1000
            end
            
            tmp_repr_err(i, j) = best_repr_err;
            % ========================================================
        end
    end
    
    % 3.3 存储统计数据
    median_rot_err_per_level(n_idx, :)   = median(tmp_rot_err, 1);
    median_trans_err_per_level(n_idx, :) = median(tmp_trans_err, 1);
    median_repr_err_per_level(n_idx, :)  = median(tmp_repr_err, 1); 
    
    raw_rot_data{n_idx}   = tmp_rot_err;
    raw_trans_data{n_idx} = tmp_trans_err;
    raw_repr_data{n_idx}  = tmp_repr_err;
    
    mean_time_per_level(n_idx, :) = mean(tmp_time, 1) * 1000;
    
    if ~isempty(h_wait)
        waitbar(n_idx / num_n_levels, h_wait, ...
            sprintf('Points: %d (%d/%d)', curr_n, n_idx, num_n_levels));
    end
    fprintf('Points: %d (%d/%d) done\n', curr_n, n_idx, num_n_levels);
end
if ~isempty(h_wait), close(h_wait); end

%% 4. 绘图配置
line_colors = lines(num_algos);
line_colors(1,:) = [0.85, 0.33, 0.1]; % 突出第一个算法
idx_gauss = find(strcmp(algo_names, 'Proposed-Gauss'));
if ~isempty(idx_gauss), line_colors(idx_gauss, :) = [0, 0.45, 0.74]; end

line_styles = {'--', '-', '-.', ':', '-', '--', '-.', ':','-',':'};
markers     = {'o', 's', '^', 'd', 'v', '>', '<', 'p','*','h'};

target_indices = find(ismember(n_points_list, boxplot_n_targets));

%% 5. 绘图 A: 折线图 (趋势分析) 
figure('Name', 'Exp2_Line_Rot', 'Color', 'w', 'Position', [50, 400, 500, 400]);
hold on; grid on; box on;
for j = 1:num_algos
    lw = 1.5; if j==2, lw = 2.5; end
    plot(n_points_list, median_rot_err_per_level(:, j), ...
        'Color', line_colors(j,:), 'LineWidth', lw, ...
        'LineStyle', line_styles{j}, 'Marker', markers{j}, 'MarkerSize', 8);
end
xlabel('Number of Points (N)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Median Rotation Error (deg)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([min(n_points_list), max(n_points_list)]);
legend(algo_names, 'Location', 'northeast', 'Interpreter', 'none', 'FontSize', 10);

figure('Name', 'Exp2_Line_Trans', 'Color', 'w', 'Position', [550, 400, 500, 400]);
hold on; grid on; box on;
for j = 1:num_algos
    lw = 1.5; if j==2, lw = 2.5; end
    plot(n_points_list, median_trans_err_per_level(:, j), ...
        'Color', line_colors(j,:), 'LineWidth', lw, ...
        'LineStyle', line_styles{j}, 'Marker', markers{j}, 'MarkerSize', 8);
end
xlabel('Number of Points (N)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Median Translation Error (%)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([min(n_points_list), max(n_points_list)]);
legend(algo_names, 'Location', 'northeast', 'Interpreter', 'none', 'FontSize', 10);

figure('Name', 'Exp2_Line_Repr', 'Color', 'w', 'Position', [1050, 400, 500, 400]);
hold on; grid on; box on;
for j = 1:num_algos
    lw = 1.5; if j==2, lw = 2.5; end
    plot(n_points_list, median_repr_err_per_level(:, j), ...
        'Color', line_colors(j,:), 'LineWidth', lw, ...
        'LineStyle', line_styles{j}, 'Marker', markers{j}, 'MarkerSize', 8);
end
xlabel('Number of Points (N)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Median Norm. Reproj. Error', 'FontSize', 12, 'FontWeight', 'bold');
xlim([min(n_points_list), max(n_points_list)]);
legend(algo_names, 'Location', 'northeast', 'Interpreter', 'none', 'FontSize', 10);


%% 6. 绘图 B: 箱线图 (分布分析) 
if ~isempty(target_indices)
    for k = 1:length(target_indices)
        idx = target_indices(k);
        curr_n_val = n_points_list(idx);
        
        rot_data_now   = raw_rot_data{idx};
        trans_data_now = raw_trans_data{idx};
        repr_data_now  = raw_repr_data{idx}; 
        
        % 动态 Y 轴
        if curr_n_val <= 6
            lim_rot   = [0, 15]; 
            lim_trans = [0, 20];
        elseif curr_n_val == 10
            lim_rot   = [0, 3];
            lim_trans = [0, 1];
        elseif curr_n_val == 20
            lim_rot   = [0, 2];
            lim_trans = [0, 0.4];
        elseif curr_n_val >= 100
            lim_rot   = [0, 0.8];
            lim_trans = [0, 0.2];
        else
            lim_rot   = [0, 5];
            lim_trans = [0, 5];
        end
        
        % --- Boxplot: Rot ---
        fig_name_rot = sprintf('Exp2_Box_Rot_N%d', curr_n_val);
        figure('Name', fig_name_rot, 'Color', 'w', 'Position', [100 + k*30, 100, 400, 300]);
        boxplot(rot_data_now, 'Labels', algo_names, 'Symbol', 'r+'); 
        grid on; box on;
        ylabel(sprintf('Rot. Err. (deg) at N=%d', curr_n_val), 'FontSize', 11, 'FontWeight', 'bold');
        xtickangle(45);
        ylim(lim_rot);
        
        % --- Boxplot: Trans ---
        fig_name_trans = sprintf('Exp2_Box_Trans_N%d', curr_n_val);
        figure('Name', fig_name_trans, 'Color', 'w', 'Position', [550 + k*30, 100, 400, 300]);
        boxplot(trans_data_now, 'Labels', algo_names, 'Symbol', 'r+'); 
        grid on; box on;
        ylabel(sprintf('Trans. Err. (%%) at N=%d', curr_n_val), 'FontSize', 11, 'FontWeight', 'bold');
        xtickangle(45);
        ylim(lim_trans);
        
        % --- Boxplot: Reproj ---
        fig_name_repr = sprintf('Exp2_Box_Repr_N%d', curr_n_val);
        figure('Name', fig_name_repr, 'Color', 'w', 'Position', [1000 + k*30, 100, 400, 300]);
        boxplot(repr_data_now, 'Labels', algo_names, 'Symbol', 'r+'); 
        grid on; box on;
        ylabel(sprintf('Norm. Reproj. at N=%d', curr_n_val), 'FontSize', 11, 'FontWeight', 'bold');
        xtickangle(45);
        % 对于重投影误差，使用默认比例缩放以显示数据的真实波动
    end
end

fprintf('\n实验二绘图完成。\n');