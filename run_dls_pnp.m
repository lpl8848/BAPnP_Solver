function [R, t] = run_dls_pnp(y, P)
% DLS-PnP 接口封装
% 输入:
%   y: 3xN 归一化图像点 (齐次坐标)
%   P: 3xN 世界坐标点
% 输出:
%   R: 3x3 旋转矩阵
%   t: 3x1 平移向量

    % 1. 转换为非齐次 2D 坐标 (2xN)
    z = y(1:2, :) ./ y(3, :);
    
    % 2. 调用原始 dls_pnp
    % 注意：dls_pnp 可能返回多个解
    [C_est, t_est, cost, ~] = dls_pnp(P, z);
    
    % 3. 选择最优解
    if isempty(cost)
        R = eye(3); t = zeros(3,1); % 默认失败返回
        return;
    end
    
    [~, best_idx] = min(cost);
    R = C_est(:, :, best_idx);
    t = t_est(:, best_idx);
    
    % 确保 t 是列向量
    if size(t, 2) > 1
        t = t';
    end
end