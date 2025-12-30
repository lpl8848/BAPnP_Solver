function [R, t] = run_ceppnp(y, P)
% CEPPnP 接口封装
% 输入:
%   y: 3xN 归一化图像点 (齐次坐标)
%   P: 3xN 世界坐标点
% 输出:
%   R: 3x3 旋转矩阵
%   t: 3x1 平移向量

    % 1. 转换为非齐次 2D 坐标 (2xN)
    impts = y(1:2, :) ./ y(3, :);
    
    % 2. 准备协方差矩阵 Cu
    % CEPPnP 是 "Covariance-Enabled" PnP，必须提供 2x2xN 的协方差矩阵。
    n_points = size(impts, 2);
    Cu = repmat(eye(2), 1, 1, n_points);
    
    % 3. 调用原始 CEPPnP
    % 注意：CEPPnP 会调用 FNSani，现在维度匹配了
    [R, t, ~] = CEPPnP(P, impts, Cu);
    
    % 确保 t 是列向量
    if size(t, 2) > 1
        t = t';
    end
end