clc; clear; close all;

num_trials = 500;         
fixed_noise = 2.0;       


n_points_list = [6, 8, 10, 12, 15, 20, 30, 50, 80, 100];
num_n_levels = length(n_points_list);


boxplot_n_targets = [6,10, 20, 100]; 


algorithms = {
    'BAPnP',   @pnp_linear_only;
    'BAPnP-GN',   @BAPnP;
    'EPnP-GN', @run_epnp_guass; 
    'OPnP',       @run_opnp;
    'RPnP',       @run_rpnp;
    'SRPnP-GN',        @run_srpnp;
    'MLPnP',      @run_mlpnp;
    'CPnP-GN',      @run_cpnp;
    'SQPnP',      @sqpnp;
};
num_algos = size(algorithms, 1);
algo_names = algorithms(:,1);


median_rot_err_per_level   = zeros(num_n_levels, num_algos);
median_trans_err_per_level = zeros(num_n_levels, num_algos);
raw_rot_data   = cell(num_n_levels, 1);
raw_trans_data = cell(num_n_levels, 1);
mean_time_per_level = zeros(num_n_levels, num_algos);

fprintf(' (Noise=%.1f px) ===\n', fixed_noise);


h_wait = waitbar(0, 'Running Simulation...');

for n_idx = 1:num_n_levels
    curr_n = n_points_list(n_idx);
    
    tmp_rot_err   = zeros(num_trials, num_algos);
    tmp_trans_err = zeros(num_trials, num_algos);
    tmp_time      = zeros(num_trials, num_algos);
    
    for i = 1:num_trials
       
        [pts3d, ~, pts2d_norm, ~, R_gt, t_gt] = ...
            generate_P6P_3D_to_2D_point_correspondences_noise(curr_n, fixed_noise);
        
       
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
    
    waitbar(n_idx / num_n_levels, h_wait, ...
        sprintf('Points: %d (%d/%d)', curr_n, n_idx, num_n_levels));
end
close(h_wait);


line_colors = lines(num_algos);
line_colors(1,:) = [0.85, 0.33, 0.1]; % Proposed 
idx_gauss = find(strcmp(algo_names, 'Proposed-Gauss'));
if ~isempty(idx_gauss), line_colors(idx_gauss, :) = [0, 0.45, 0.74]; end

line_styles = {'--', '-', '-.', ':', '-', '--', '-.', ':','-'};
markers     = {'o', 's', '^', 'd', 'v', '>', '<', 'p','*'};

boxplot_n_targets_new = [6, 10, 20, 100]; 
target_indices = find(ismember(n_points_list, boxplot_n_targets_new));


figure('Name', 'Exp2_Line_Rot', 'Color', 'w', 'Position', [100, 400, 600, 500]);
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



figure('Name', 'Exp2_Line_Trans', 'Color', 'w', 'Position', [750, 400, 600, 500]);
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



if ~isempty(target_indices)
    for k = 1:length(target_indices)
        idx = target_indices(k);
        curr_n_val = n_points_list(idx);
        
      
        rot_data_now   = raw_rot_data{idx};
        trans_data_now = raw_trans_data{idx};
        
      
        if curr_n_val <= 6
            % N=6
            lim_rot   = [0, 15]; 
            lim_trans = [0, 20];
            
        elseif curr_n_val == 10
            % N=10 
            lim_rot   = [0, 3];
            lim_trans = [0, 1];
            
        elseif curr_n_val == 20
            % N=20 
            lim_rot   = [0, 2];
            lim_trans = [0, 0.4];
            
        elseif curr_n_val >= 100
            % N=100 
            lim_rot   = [0, 0.8];
            lim_trans = [0, 0.2];
            
        else
           
            lim_rot   = [0, 5];
            lim_trans = [0, 5];
        end
        % ========================================================

        
        
        fig_name_rot = sprintf('Exp2_Box_Rot_N%d', curr_n_val);
        figure('Name', fig_name_rot, 'Color', 'w', 'Position', [100 + k*30, 100, 500, 400]);
        
        boxplot(rot_data_now, 'Labels', algo_names, 'Symbol', 'r+'); 
        grid on; box on;
        
        ylabel(sprintf('Rot. Err. (deg) at N=%d', curr_n_val), ...
               'FontSize', 11, 'FontWeight', 'bold');
        xtickangle(45);
        
      
        ylim(lim_rot);
        
        
      
        fig_name_trans = sprintf('Exp2_Box_Trans_N%d', curr_n_val);
        figure('Name', fig_name_trans, 'Color', 'w', 'Position', [700 + k*30, 100, 500, 400]);
        
        boxplot(trans_data_now, 'Labels', algo_names, 'Symbol', 'r+'); 
        grid on; box on;
        
        ylabel(sprintf('Trans. Err. (%%) at N=%d', curr_n_val), ...
               'FontSize', 11, 'FontWeight', 'bold');
        xtickangle(45);
        
       
        ylim(lim_trans);
        
    end
end


