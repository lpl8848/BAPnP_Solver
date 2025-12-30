function [R, t] = run_cpnp(y, P)
% CPnP 接口封装 (Wrapper for CPnP)
% 参考 run_opnp 的风格进行封装，以便统一调用
%
% 输入:
%   y: 3xN 归一化图像点 (齐次坐标, Normalized Image Coordinates)
%      通常由 inv(K) * [u; v; 1] 得到
%   P: 3xN 世界坐标点 (World Coordinates)
%
% 输出:
%   R: 3x3 旋转矩阵 (Refined by Gauss-Newton)
%   t: 3x1 平移向量 (Refined by Gauss-Newton)

    % 1. 转换为非齐次 2D 坐标 (2xN)
    % 将归一化平面的齐次坐标转换为 2D 坐标
    u = y(1:2, :) ./ y(3, :);

    % 2. 设置虚拟内参
    % 因为输入 y 已经是归一化坐标，所以对应的是标准相机模型：
    % fx = 1, fy = 1, u0 = 0, v0 = 0
    fx = 1;
    fy = 1;
    u0 = 0;
    v0 = 0;

    % 3. 调用 CPnP
    % CPnP 函数签名: [R,t,R_GN,t_GN] = CPnP(s, Psens_2D, fx, fy, u0, v0)
    % s 对应 P (3D点), Psens_2D 对应 u (2D点)
    [~, ~, R_GN, t_GN] = CPnP(P, u, fx, fy, u0, v0);

    % 4. 返回结果
    % 使用 Gauss-Newton 优化后的结果作为最终输出
    R = R_GN;
    t = t_GN;
    
    % 确保 t 是列向量 (以防万一)
    if size(t, 2) > 1
        t = t';
    end
end
