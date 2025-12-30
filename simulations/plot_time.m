clc; clear; close all;

% --- 1. Experimental Data (10,000 Trials) ---
N = [6, 10, 20, 50, 100, 200, 500, 1000];

% Time in milliseconds (ms)
t_bapnp   = [0.0115, 0.0083, 0.0108, 0.0181, 0.0304, 0.0556, 0.1298, 0.2734];
t_epnp    = [0.0557, 0.0564, 0.0548, 0.0609, 0.0748, 0.1013, 0.2212, 0.3993];
t_epnp_lm = [0.2216, 0.2126, 0.2352, 0.3196, 0.4557, 0.7246, 2.1990, 4.6120];
t_cpnp    = [0.0449, 0.0472, 0.0450, 0.0595, 0.0847, 0.1632, 0.3911, 0.6636];
t_sqpnp   = [0.0520, 0.0494, 0.0529, 0.0599, 0.0720, 0.0914, 0.1277, 0.1860];

% --- 2. Plotting Configuration ---
figure('Color', 'w', 'Position', [100, 100, 600, 450]); % 标准论文插图尺寸

% 使用双对数坐标 (Log-Log Plot)
loglog(N, t_bapnp,   '-ro', 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'BAPnP (Ours)'); hold on;
loglog(N, t_epnp_lm, '--ms', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'EPnP+LM (OpenCV)');
loglog(N, t_sqpnp,   '-.^g', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'SQPnP (OpenCV)');
loglog(N, t_cpnp,    '--dk', 'LineWidth', 1.2, 'MarkerSize', 6, 'DisplayName', 'CPnP');
loglog(N, t_epnp,    ':xb',  'LineWidth', 1.2, 'MarkerSize', 7, 'DisplayName', 'EPnP (Linear)');

% --- 3. Aesthetics & Annotations ---
grid on;
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.1); % 网格淡一点
set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'FontName', 'Times New Roman'); % IEEE 字体

xlabel('Number of Points ($N$)', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Execution Time (ms)', 'Interpreter', 'latex', 'FontSize', 14);
title('Computational Efficiency vs. Number of Points', 'FontWeight', 'bold');

% 设置 X 轴刻度，避免科学计数法显示不直观
xticks(N);
xticklabels(string(N));
xlim([5, 1100]);

% 图例位置
legend('Location', 'northwest', 'FontSize', 11, 'Interpreter', 'latex');

% --- 4. 添加 "25x Faster" 的标注 (Highlight) ---
% 在 N=10 的位置标注
% 计算箭头起点和终点
x_arrow = 10;
y_start = t_epnp_lm(2); % 从 EPnP+LM 附近指下来
y_end   = t_bapnp(2);   % 指向 BAPnP

% 使用 text 添加标注
text(12, 0.012, '\leftarrow \textbf{25$\times$ Faster}', ...
    'Interpreter', 'latex', 'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');

% (可选) 绘制一条垂直虚线连接两个点，展示差距
plot([10, 10], [t_bapnp(2), t_epnp_lm(2)], '--k', 'LineWidth', 0.5, 'HandleVisibility', 'off');

% --- 5. Save ---
% 建议保存为 .eps 或 .pdf 矢量图
exportgraphics(gcf, 'exp3_time_benchmark.pdf', 'ContentType', 'vector');
exportgraphics(gcf, 'exp3_time_benchmark.png', 'Resolution', 300);

fprintf('Plot saved as exp3_time_benchmark.pdf and .png\n');
