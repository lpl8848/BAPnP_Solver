clc; clear; close all;

% --- 1. Experimental Data (10,000 Trials) ---
N = [6, 10, 20, 50, 100, 200, 400, 500, 1000];

% Time in milliseconds (ms)
t_bapnp   = [ 0.0039, 0.0044, 0.0059, 0.0103, 0.0166, 0.0299, 0.0556, 0.0678,0.1317];
t_epnp    = [0.0381,  0.0381, 0.0411, 0.0467, 0.0509, 0.0657, 0.0956, 0.1092,0.1876];
t_cpnp    = [0.0234, 0.0259, 0.0337, 0.0553, 0.0834, 0.1853, 0.3478 , 0.4508,0.7595];
t_sqpnp   = [0.0266, 0.0265, 0.0308, 0.0399, 0.0425, 0.0489, 0.0574, 0.0619,0.0740];

% --- 2. Plotting Configuration ---
figure('Color', 'w', 'Position', [100, 100, 600, 450]); 


loglog(N, t_bapnp,   '-ro', 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'BAPnP'); hold on;
loglog(N, t_sqpnp,   '-.^g', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'SQPnP');
loglog(N, t_cpnp,    '--dk', 'LineWidth', 1.2, 'MarkerSize', 6, 'DisplayName', 'CPnP');
loglog(N, t_epnp,    ':xb',  'LineWidth', 1.2, 'MarkerSize', 7, 'DisplayName', 'EPnP');

% --- 3. Aesthetics & Annotations ---
grid on;
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.1); 
set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'FontName', 'Times New Roman'); 

xlabel('Number of Points ($N$)', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Execution Time (ms)', 'Interpreter', 'latex', 'FontSize', 14);
title('Computational Efficiency vs. Number of Points', 'FontWeight', 'bold');


xticks(N);
xticklabels(string(N));
xlim([5, 1100]);


legend('Location', 'northwest', 'FontSize', 11, 'Interpreter', 'latex');


x_arrow = 10;
y_start = t_epnp(2);
y_end   = t_bapnp(2); 


plot([10, 10], [t_bapnp(2), t_epnp(2)], '--k', 'LineWidth', 0.5, 'HandleVisibility', 'off');

% --- 5. Save ---

exportgraphics(gcf, 'exp3_time_benchmark.pdf', 'ContentType', 'vector');
exportgraphics(gcf, 'exp3_time_benchmark.png', 'Resolution', 300);

fprintf('Plot saved as exp3_time_benchmark.pdf and .png\n');
