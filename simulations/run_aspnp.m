function [R, t] = run_aspnp(y, P)
% ASPnP 接口封装
% 输入:
%   y: 3xN 归一化图像点 (齐次坐标)
%   P: 3xN 世界坐标点
% 输出:
%   R: 3x3 旋转矩阵
%   t: 3x1 平移向量

    % 1. 转换为非齐次 2D 坐标 (2xN)
    % ASPnP 内部其实也可以处理齐次，但为了保险，我们传入它期望的格式
    u = y(1:2, :) ./ y(3, :);
    
    % 2. 调用原始 ASPnP
    % ASPnP(U0, u0, K)
    % 第三个参数 K如果不传，函数内部默认 u0 是归一化坐标
    [R, t, ~] = ASPnP(P, u);
    
    % 确保 t 是列向量
    if size(t, 2) > 1
        t = t';
    end
end
