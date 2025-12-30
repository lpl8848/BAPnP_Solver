function [R, t] = run_epnp_guass(y, P)
% EPnP (Gauss-Newton) 接口封装
% 输入:
%   y: 3xN 归一化图像点 (齐次坐标)
%   P: 3xN 世界坐标点
% 输出:
%   R: 3x3 旋转矩阵
%   t: 3x1 平移向量

    n = size(P, 2);
    
    % 1. 格式转换
    % EPnP 需要 Nx3 的世界坐标 (Xw)
    Xw_h = [P', ones(n, 1)]; % Nx4, 齐次
    
    % EPnP 需要 Nx2 的图像坐标 (U), 这里也传入齐次 Nx3 方便处理
    % 注意：efficient_pnp 内部期望的是 Nx3 的齐次坐标形式用于 U
    x2d_h = y'; % Nx3
    
    % 2. 内参矩阵 (归一化坐标下为单位阵)
    A = eye(3);
    
    % 3. 调用 EPnP (带高斯牛顿优化)
    [R, t, ~, ~] = efficient_pnp_gauss(Xw_h, x2d_h, A);
   %[R, t, ~, ~] = efficient_pnp(Xw_h, x2d_h, A);
    % 确保 t 是列向量
    if size(t, 2) > 1
        t = t';
    end
end
