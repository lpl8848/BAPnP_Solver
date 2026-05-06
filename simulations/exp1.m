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
median_rot_err_per_level   = zeros(num_noise_levels, num_algos);
median_trans_err_per_level = zeros(num_noise_levels, num_algos);
median_repr_err_per_level  = zeros(num_noise_levels, num_algos); % [新增] 重投影误差中位数

raw_rot_data   = cell(num_noise_levels, 1);
raw_trans_data = cell(num_noise_levels, 1);
raw_repr_data  = cell(num_noise_levels, 1); % [新增] 原始重投影误差数据(用于箱线图)
mean_time_per_level = zeros(num_noise_levels, num_algos);

fprintf('=== 开始运行实验一: 噪声敏感度 (N=%d) ===\n', N);

%% 3. 主循环
h_wait = []; % headless-compatible, use fprintf instead

for n_idx = 1:num_noise_levels
    curr_noise = noise_levels(n_idx);
    
    tmp_rot_err   = zeros(num_trials, num_algos);
    tmp_trans_err = zeros(num_trials, num_algos);
    tmp_repr_err  = zeros(num_trials, num_algos); % [新增]
    tmp_time      = zeros(num_trials, num_algos);
    
    for i = 1:num_trials
        [pts3d, ~, pts2d_norm, ~, R_gt, t_gt] = ...
            generate_P6P_3D_to_2D_point_correspondences_noise(N, curr_noise);
        
        % [新增] 预处理实际观测点(去齐次化)
        if size(pts2d_norm, 1) == 3
            pts2d_obs = pts2d_norm(1:2, :) ./ pts2d_norm(3, :);
        else
            pts2d_obs = pts2d_norm(1:2, :);
        end
        
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
            
            % [新增] 计算归一化平面上的重投影误差
            P_c = R_e * pts3d + t_e;                    % 转换到相机坐标系
            P_c(3, P_c(3,:) < 1e-6) = 1e-6;             % 避免除零或点在相机后方
            pts2d_proj = P_c(1:2, :) ./ P_c(3, :);      % 透视投影
            
            % 计算当前 trial 均方根/平均重投影误差
            curr_repr_err = mean(sqrt(sum((pts2d_proj - pts2d_obs).^2, 1))); 
            tmp_repr_err(i, j) = curr_repr_err;
        end
    end
    
    median_rot_err_per_level(n_idx, :)   = median(tmp_rot_err, 1);
    median_trans_err_per_level(n_idx, :) = median(tmp_trans_err, 1);
    median_repr_err_per_level(n_idx, :)  = median(tmp_repr_err, 1); % [新增]
    
    raw_rot_data{n_idx}   = tmp_rot_err;
    raw_trans_data{n_idx} = tmp_trans_err;
    raw_repr_data{n_idx}  = tmp_repr_err; % [新增]
    
    mean_time_per_level(n_idx, :) = mean(tmp_time, 1) * 1000;
    
    if ~isempty(h_wait)
        waitbar(n_idx / num_noise_levels, h_wait, ...
            sprintf('Noise: %.1f px (%d/%d)', curr_noise, n_idx, num_noise_levels));
    end
    fprintf('Noise: %.1f px (%d/%d) done\n', curr_noise, n_idx, num_noise_levels);
end
if ~isempty(h_wait), close(h_wait); end

%% 4. 绘图配置
line_colors = lines(num_algos);
line_colors(1,:) = [0.85, 0.33, 0.1]; % 突出第一个算法

idx_gauss = find(strcmp(algo_names, 'Proposed-Gauss'));
if ~isempty(idx_gauss), line_colors(idx_gauss, :) = [0, 0.45, 0.74]; end

line_styles = {'--', '-', '-.', ':', '-', '--', '-.', ':','-',':'};
markers     = {'o', 's', '^', 'd', 'v', '>', '<', 'p','*','h'};

target_indices = find(ismember(noise_levels, boxplot_noise_targets));

%% 5. 绘图 A: 折线图 (趋势分析)

% --- A1: 旋转误差 (Rotation Error) ---
figure('Name', 'LinePlot_Rotation', 'Color', 'w', 'Position', [50, 400, 500, 400]);
hold on; grid on; box on;
for j = 1:num_algos
    lw = 1.5; if j==2, lw = 2.5; end
    plot(noise_levels, median_rot_err_per_level(:, j), ...
        'Color', line_colors(j,:), 'LineWidth', lw, ...
        'LineStyle', line_styles{j}, 'Marker', markers{j}, 'MarkerSize', 8); 
end
xlabel('Gaussian Noise \sigma (pixels)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Median Rotation Error (deg)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0, max(noise_levels)]);
legend(algo_names, 'Location', 'northwest', 'Interpreter', 'none', 'FontSize', 10);

% --- A2: 平移误差 (Translation Error) ---
figure('Name', 'LinePlot_Translation', 'Color', 'w', 'Position', [550, 400, 500, 400]);
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
legend(algo_names, 'Location', 'northwest', 'Interpreter', 'none', 'FontSize', 10);

% --- A3: 重投影误差 (Reprojection Error) [新增] ---
figure('Name', 'LinePlot_Reprojection', 'Color', 'w', 'Position', [1050, 400, 500, 400]);
hold on; grid on; box on;
for j = 1:num_algos
    lw = 1.5; if j==2, lw = 2.5; end
    plot(noise_levels, median_repr_err_per_level(:, j), ...
        'Color', line_colors(j,:), 'LineWidth', lw, ...
        'LineStyle', line_styles{j}, 'Marker', markers{j}, 'MarkerSize', 8);
end
xlabel('Gaussian Noise \sigma (pixels)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Median Norm. Reprojection Error', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0, max(noise_levels)]);
legend(algo_names, 'Location', 'northwest', 'Interpreter', 'none', 'FontSize', 10);


%% 6. 绘图 B: 箱线图 (分布分析)

if ~isempty(target_indices)

    % --- 图 B1：旋转误差箱线图 ---
    figure('Name','Boxplot_Rotation','Color','w','Position',[100,50,900,300]);
    for k = 1:length(target_indices)
        idx = target_indices(k);
        curr_noise_val = noise_levels(idx);
        rot_data_now = raw_rot_data{idx};
        
        if abs(curr_noise_val - 1.0) < 1e-6
            lim_rot = [0 2];
        elseif abs(curr_noise_val - 3.0) < 1e-6
            lim_rot = [0 3];
        elseif abs(curr_noise_val - 5.0) < 1e-6
            lim_rot = [0 4];
        else
            lim_rot = [0 5];
        end

        subplot(1,length(target_indices),k)
        boxplot(rot_data_now,'Labels',algo_names,'Symbol','r+');
        grid on; box on;
        title(sprintf('\\sigma = %.1f px',curr_noise_val))
        ylabel('Rotation Error (deg)')
        xtickangle(45)
        ylim(lim_rot)
    end

    % --- 图 B2：重投影误差箱线图 [新增] ---
    % 将重投影误差也做成箱线图，可以直接回应审稿人2关于优化指标分布的问题
    figure('Name','Boxplot_Reprojection','Color','w','Position',[150,50,900,300]);
    for k = 1:length(target_indices)
        idx = target_indices(k);
        curr_noise_val = noise_levels(idx);
        repr_data_now = raw_repr_data{idx};
        
        subplot(1,length(target_indices),k)
        boxplot(repr_data_now,'Labels',algo_names,'Symbol','r+');
        grid on; box on;
        title(sprintf('\\sigma = %.1f px',curr_noise_val))
        ylabel('Norm. Reproj Error')
        xtickangle(45)
        % 你可以根据实际数据跑出来的范围，解开下方的限制以保持美观
        % ylim([0, curr_noise_val * 0.005]); 
    end
end

fprintf('\n绘图完成。\n');
fprintf('已生成 折线图 (旋转、平移、重投影)。\n');
fprintf('已生成 箱线图 (旋转、重投影)，根据噪声等级 %.1f, %.1f, %.1f 绘制。\n', ...
    boxplot_noise_targets);