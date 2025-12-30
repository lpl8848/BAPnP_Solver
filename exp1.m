clc; clear; close all;

%% 1. 实验设置
num_trials = 500;       % 实验次数
N = 20;                 % 固定点数

% 定义噪声等级
noise_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]; 
num_noise_levels = length(noise_levels);

% 定义要画箱线图的具体噪声点
boxplot_noise_targets = [1.0, 3.0, 5.0]; 

% 定义算法
algorithms = {
    'BAPnP',   @pnp_linear_only;
    'BAPnP-GN',   @BAPnP;
    'EPnP-GN', @run_epnp_guass; 
    'OPnP',       @run_opnp;
    'RPnP',       @run_rpnp;
    'SRPnP-GN',        @run_srpnp;
    'MLPnP',      @run_mlpnp;
    'CPnP-GN',      @run_cpnp;
};
num_algos = size(algorithms, 1);
algo_names = algorithms(:,1);

%% 2. 初始化存储容器
median_rot_err_per_level   = zeros(num_noise_levels, num_algos);
median_trans_err_per_level = zeros(num_noise_levels, num_algos);
raw_rot_data   = cell(num_noise_levels, 1);
raw_trans_data = cell(num_noise_levels, 1);
mean_time_per_level = zeros(num_noise_levels, num_algos);

fprintf('=== 开始运行实验一: 噪声敏感度 (N=%d) ===\n', N);

%% 3. 主循环
h_wait = waitbar(0, 'Running Simulation...');

for n_idx = 1:num_noise_levels
    curr_noise = noise_levels(n_idx);
    
    tmp_rot_err   = zeros(num_trials, num_algos);
    tmp_trans_err = zeros(num_trials, num_algos);
    tmp_time      = zeros(num_trials, num_algos);
    
    for i = 1:num_trials
        [pts3d, ~, pts2d_norm, ~, R_gt, t_gt] = ...
            generate_P6P_3D_to_2D_point_correspondences_noise(N, curr_noise);
        
        for j = 1:num_algos
            algo_func = algorithms{j, 2};
            t_start = tic; 
            try
                [R_e, t_e] = algo_func(pts2d_norm, pts3d);
            catch
                R_e = eye(3); t_e = zeros(3,1);
            end
            tmp_time(i, j) = toc(t_start);
            
            [r_err, t_err] = calc_error(R_e, t_e, R_gt, t_gt);
            tmp_rot_err(i, j)   = r_err;
            tmp_trans_err(i, j) = t_err;
        end
    end
    
    median_rot_err_per_level(n_idx, :)   = median(tmp_rot_err, 1);
    median_trans_err_per_level(n_idx, :) = median(tmp_trans_err, 1);
    
    raw_rot_data{n_idx}   = tmp_rot_err;
    raw_trans_data{n_idx} = tmp_trans_err;
    
    mean_time_per_level(n_idx, :) = mean(tmp_time, 1) * 1000;
    
    waitbar(n_idx / num_noise_levels, h_wait, ...
        sprintf('Noise: %.1f px (%d/%d)', curr_noise, n_idx, num_noise_levels));
end
close(h_wait);

%% 4. 绘图配置
line_colors = lines(num_algos);
% 特殊处理 Proposed 算法颜色 (假设它是第一个)
line_colors(1,:) = [0.85, 0.33, 0.1]; 

% 检查是否有 'Proposed-Gauss' 并赋予特殊颜色
idx_gauss = find(strcmp(algo_names, 'Proposed-Gauss'));
if ~isempty(idx_gauss), line_colors(idx_gauss, :) = [0, 0.45, 0.74]; end

line_styles = {'--', '-', '-.', ':', '-', '--', '-.', ':'};
markers     = {'o', 's', '^', 'd', 'v', '>', '<', 'p'};

% 获取需要画箱线图的索引
target_indices = find(ismember(noise_levels, boxplot_noise_targets));

%% 5. 绘图 A: 折线图 (趋势分析) - 独立窗口

% ========================================================
% Figure A1: 旋转误差 (Rotation Error) 折线图
% ========================================================
figure('Name', 'LinePlot_Rotation', 'Color', 'w', 'Position', [100, 400, 600, 500]);
hold on; grid on; box on;

for j = 1:num_algos
    lw = 1.5; if j==2, lw = 2.5; end
    plot(noise_levels, median_rot_err_per_level(:, j), ...
        'Color', line_colors(j,:), 'LineWidth', lw, ...
        'LineStyle', line_styles{j}, 'Marker', markers{j}, 'MarkerSize', 8); % MarkerSize 调大便于观察
end

xlabel('Gaussian Noise \sigma (pixels)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Median Rotation Error (deg)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0, max(noise_levels)]);

% -----------------------------------------------------------
% [用户自定义] Y轴范围设置 (旋转折线图)
% -----------------------------------------------------------
% ylim([0, 5]);  % <--- 如果需要固定Y轴范围，请取消注释并修改数值
% -----------------------------------------------------------

% 图注设置
legend(algo_names, 'Location', 'northwest', 'Interpreter', 'none', 'FontSize', 10);


% ========================================================
% Figure A2: 平移误差 (Translation Error) 折线图
% ========================================================
figure('Name', 'LinePlot_Translation', 'Color', 'w', 'Position', [750, 400, 600, 500]);
hold on; grid on; box on;

for j = 1:num_algos
    lw = 1.5; if j==2, lw = 2.5; end
    plot(noise_levels, median_trans_err_per_level(:, j), ...
        'Color', line_colors(j,:), 'LineWidth', lw, ...
        'LineStyle', line_styles{j}, 'Marker', markers{j}, 'MarkerSize', 8);
end

xlabel('Gaussian Noise \sigma (pixels)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Median Translation Error (%)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0, max(noise_levels)]);

% -----------------------------------------------------------
% [用户自定义] Y轴范围设置 (平移折线图)
% -----------------------------------------------------------
% ylim([0, 10]); % <--- 如果需要固定Y轴范围，请取消注释并修改数值
% -----------------------------------------------------------

legend(algo_names, 'Location', 'northwest', 'Interpreter', 'none', 'FontSize', 10);


%% 6. 绘图 B: 箱线图 (分布分析) - 动态设置Y轴范围

if ~isempty(target_indices)
    for k = 1:length(target_indices)
        idx = target_indices(k);
        curr_noise_val = noise_levels(idx);
        
        % 准备数据
        rot_data_now   = raw_rot_data{idx};
        trans_data_now = raw_trans_data{idx};
        
        % ========================================================
        % [关键修改]：在此处根据噪声等级定义 Y 轴范围
        % ========================================================
        % 使用 abs(x-y)<1e-6 是为了防止浮点数比较误差
        if abs(curr_noise_val - 1.0) < 1e-6
            % 噪声 = 1.0 时的设置
            lim_rot   = [0, 2]; 
            lim_trans = [0, 3];
            
        elseif abs(curr_noise_val - 3.0) < 1e-6
            % 噪声 = 3.0 时的设置
            lim_rot   = [0, 3];
            lim_trans = [0, 4];
            
        elseif abs(curr_noise_val - 5.0) < 1e-6
            % 噪声 = 5.0 时的设置
            lim_rot   = [0, 4];
            lim_trans = [0, 4];
            
        else
            % 其他情况的默认设置 (比如以后加了噪声 2.0)
            lim_rot   = [0, 5];
            lim_trans = [0, 5];
        end
        % ========================================================
        
        
        % --- Figure B(k)-Rot: 旋转误差箱线图 ---
        fig_name_rot = sprintf('Boxplot_Rot_Noise_%.1f', curr_noise_val);
        figure('Name', fig_name_rot, 'Color', 'w', 'Position', [100 + k*30, 100, 500, 400]);
        
        boxplot(rot_data_now, 'Labels', algo_names, 'Symbol', 'r+'); 
        grid on; box on;
        
        ylabel(sprintf('Rot. Err. (deg) at \\sigma=%.1f', curr_noise_val), ...
               'FontSize', 11, 'FontWeight', 'bold');
        xtickangle(45);
        
        % 应用动态设置的 Y 轴范围
        ylim(lim_rot); 
        
        
        % --- Figure B(k)-Trans: 平移误差箱线图 ---
        fig_name_trans = sprintf('Boxplot_Trans_Noise_%.1f', curr_noise_val);
        figure('Name', fig_name_trans, 'Color', 'w', 'Position', [700 + k*30, 100, 500, 400]);
        
        boxplot(trans_data_now, 'Labels', algo_names, 'Symbol', 'r+'); 
        grid on; box on;
        
        ylabel(sprintf('Trans. Err. (%%) at \\sigma=%.1f', curr_noise_val), ...
               'FontSize', 11, 'FontWeight', 'bold');
        xtickangle(45);
        
        % 应用动态设置的 Y 轴范围
        ylim(lim_trans); 
    end
end

fprintf('\n绘图完成。\n');
fprintf('折线图已生成。\n');
fprintf('箱线图已生成 (根据噪声等级 %.1f, %.1f, %.1f 自动调整了Y轴范围)。\n', ...
    boxplot_noise_targets);