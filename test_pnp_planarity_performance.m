
% ============================================================
% Paper-ready experiment (FINAL VERSION)
% PnP robustness under near-coplanar and coplanar configurations
%
% Key point:
%   Coplanar PnP is theoretically UNIQUE (homography-induced),
%   therefore pose errors ARE WELL-DEFINED and ARE REPORTED.
% ============================================================

clc; clear; close all;

%% ------------------ Algorithm list ------------------
algorithms = {
    'BAPnP-GN',   @BAPnP;
    'EPnP-GN', @run_epnp_guass; 
    'OPnP',       @run_opnp;
    'RPnP',       @run_rpnp;
    'SRPnP-GN',        @run_srpnp;
    'MLPnP',      @run_mlpnp;
    'CPnP-GN',      @run_cpnp;
};
n_algs = size(algorithms,1);

%% ------------------ Parameters ------------------
n_points   = 20;
n_trials   = 1000;

focal       = 800;
pixel_noise = 1.0;

% from non-planar -> strictly planar
z_spread_levels = [1e-1, 1e-2, 1e-3,1e-5, 1e-7, 1e-10, 1e-12];

% success criterion (numerical consistency)
rot_thresh   = 2;    % degrees
trans_thresh = 2;   % percent

%% ------------------ Result storage ------------------
n_levels = numel(z_spread_levels);

MedianRot   = zeros(n_levels, n_algs);
MedianTrans = zeros(n_levels, n_algs);
SuccessRate = zeros(n_levels, n_algs);
Planarity   = zeros(n_levels,1);   % sigma3 / sigma2

fprintf('Running FINAL planarity experiment...\n');

%% ====================================================
for i = 1:n_levels
    z_spread = z_spread_levels(i);

    rot_err   = nan(n_trials, n_algs);
    trans_err = nan(n_trials, n_algs);
    success   = false(n_trials, n_algs);
    rho_vals  = zeros(n_trials,1);

    for k = 1:n_trials
        %% ---------- Generate 3D points ----------
        Pw = 2*(rand(3,n_points) - 0.5);

        if z_spread == 0
            Pw(3,:) = 0;                 % strictly coplanar
        else
            Pw(3,:) = Pw(3,:) * z_spread;
        end

        % planarity metric (geometry-driven)
        P0 = Pw - mean(Pw,2);
        s  = svd(P0');
        rho_vals(k) = s(3) / s(2);

        %% ---------- Camera pose ----------
        R_gt = random_rotation();
        t_gt = [0;0;4] + 0.5*(rand(3,1)-0.5);

        %% ---------- Projection ----------
        Pc = R_gt * Pw + t_gt;
        y  = Pc(1:2,:) ./ Pc(3,:);
        y  = y + (pixel_noise / focal) * randn(2,n_points);
        y  = [y; ones(1,n_points)];

        %% ---------- Run solvers ----------
        for a = 1:n_algs
            try
                [R_est, t_est] = algorithms{a,2}(y, Pw);
                [re, te] = pose_error(R_gt, t_gt, R_est, t_est);

                rot_err(k,a)   = re;
                trans_err(k,a) = te;
                success(k,a)   = (re < rot_thresh) && (te < trans_thresh);
            catch
                % leave NaN for failed trials
            end
        end
    end

    %% ---------- Aggregate statistics ----------
    Planarity(i) = median(rho_vals);

    for a = 1:n_algs
        valid = ~isnan(rot_err(:,a));

        % IMPORTANT:
        % Coplanar case IS INCLUDED.
        % Pose error is meaningful because coplanar PnP is unique.
        MedianRot(i,a)   = median(rot_err(valid,a));
        MedianTrans(i,a) = median(trans_err(valid,a));
        SuccessRate(i,a) = mean(success(valid,a)) * 100;
    end

    fprintf('Level %d/%d done (z = %.1e)\n', i, n_levels, z_spread);
end

%% ------------------ Output tables ------------------
%% ------------------ Final Paper Plot (3 Separate Figures) ------------------

% === 1. 公共绘图设置 (统一风格) ===
% 论文插图推荐尺寸：单栏一般宽 400-500px，高 300-400px
fig_pos = [200, 200, 500, 400]; 

line_styles = {'-', '--', '-.', ':', '-', '--', '-.', ':'};
colors = lines(n_algs); 
colors(1,:) = [0.85, 0.33, 0.1]; % Proposed 橙红色
markers     = {'o', 's', '^', 'd', 'v', '>', '<', 'p'};
line_width = 1.5;
marker_size = 8;     % 独立图可以把点画大一点，更清楚
font_name = 'Times New Roman';
font_size = 14;      % 独立图字号可以大一点

% 构造 X 轴标签 (等间距非连续)
x_labels = arrayfun(@(x) sprintf('10^{%d}', round(log10(x))), z_spread_levels, 'UniformOutput', false);
x_idx = 1:n_levels; 

% =============================================================
%   Figure 1: Rotation Error (Clipped)
% =============================================================
figure(1); clf; 
set(gcf, 'Color', 'w', 'Position', fig_pos);
hold on; grid on; box on;

for a = 1:n_algs
    plot(x_idx, MedianRot(:, a), line_styles{a}, ...
        'Color', colors(a,:), 'LineWidth', line_width,  'Marker', markers{a},'MarkerSize', marker_size,'MarkerFaceColor', colors(a,:), ...
        'DisplayName', algorithms{a,1});
end

% 坐标轴设置
set(gca, 'XTick', x_idx, 'XTickLabel', x_labels);
xlabel('Degree of Coplanarity', 'FontName', font_name);
ylabel('Rotation Error (deg)', 'FontName', font_name);

% 【核心】截断 Y 轴
% 建议：设为 Proposed 算法最大误差的 1.5 到 2 倍，让 EPnP 穿出去
ylim([0, 5]); 

% 字体与图例
set(gca, 'FontName', font_name, 'FontSize', font_size);
legend('Location', 'northwest', 'Interpreter', 'none','FontSize',8); % 单独图例
% title('Rotation Accuracy'); % 论文正式图通常不需要 title，标题写在 LaTeX caption 里

% 保存
exportgraphics(gcf, 'Fig_Rot_Error.pdf', 'ContentType', 'vector');


% =============================================================
%   Figure 2: Translation Error (Clipped)
% =============================================================
figure(2); clf;
set(gcf, 'Color', 'w', 'Position', fig_pos + [50, -50, 0, 0]); % 稍微错开一点位置
hold on; grid on; box on;

for a = 1:n_algs
    plot(x_idx, MedianTrans(:, a), line_styles{a}, ...
        'Color', colors(a,:), 'LineWidth', line_width, 'Marker', markers{a},'MarkerSize', marker_size,'MarkerFaceColor', colors(a,:), ...
        'DisplayName', algorithms{a,1});
end

set(gca, 'XTick', x_idx, 'XTickLabel', x_labels);
xlabel('Degree of Coplanarity', 'FontName', font_name);
ylabel('Translation Error (%)', 'FontName', font_name);

% 【核心】截断 Y 轴
ylim([0, 10]); 

set(gca, 'FontName', font_name, 'FontSize', font_size);
legend('Location', 'northwest', 'Interpreter', 'none','FontSize',8);
% title('Translation Accuracy');

exportgraphics(gcf, 'Fig_Trans_Error.pdf', 'ContentType', 'vector');


% =============================================================
%   Figure 3: Success Rate (0-100)
% =============================================================
figure(3); clf;
set(gcf, 'Color', 'w', 'Position', fig_pos + [100, -100, 0, 0]);
hold on; grid on; box on;

for a = 1:n_algs
    plot(x_idx, SuccessRate(:, a), line_styles{a}, ...
        'Color', colors(a,:), 'LineWidth', line_width,  'Marker', markers{a},'MarkerSize', marker_size,'MarkerFaceColor', colors(a,:), ...
        'DisplayName', algorithms{a,1});
end

set(gca, 'XTick', x_idx, 'XTickLabel', x_labels);
xlabel('Degree of Coplanarity', 'FontName', font_name);
ylabel('Success Rate (%)', 'FontName', font_name);

% 成功率固定范围
ylim([-2, 102]); 

set(gca, 'FontName', font_name, 'FontSize', font_size);
% 成功率图的图例通常放左下角（因为曲线通常在上方）
legend('Location', 'northwest', 'Interpreter', 'none','FontSize',8);
% title('Success Rate');

exportgraphics(gcf, 'Fig_Success_Rate.pdf', 'ContentType', 'vector');

fprintf('Three figures generated and saved as PDF.\n');

%% ====================================================
function R = random_rotation()
    q = randn(4,1);
    q = q / norm(q);
    w=q(1); x=q(2); y=q(3); z=q(4);
    R = [ ...
        1-2*y^2-2*z^2, 2*x*y-2*z*w,   2*x*z+2*y*w;
        2*x*y+2*z*w,   1-2*x^2-2*z^2, 2*y*z-2*x*w;
        2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x^2-2*y^2 ];
end

function [re, te] = pose_error(R_gt, t_gt, R_est, t_est)
    R_err = R_gt' * R_est;
    v = (trace(R_err) - 1) / 2;
    v = max(min(v,1), -1);
    re = rad2deg(acos(v));
    te = norm(t_gt - t_est) / norm(t_gt) * 100;
end
