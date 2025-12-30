function [R, t] = run_reppnp(y, P)
% REPPnP 接口封装
% 输入:
%   y: 3xN 归一化图像点 (齐次坐标)
%   P: 3xN 世界坐标点
% 输出:
%   R: 3x3 旋转矩阵
%   t: 3x1 平移向量

    % 1. 转换为非齐次 2D 坐标 (2xN)
    impts = y(1:2, :) ./ y(3, :);
    
    % 2. 调用原始 REPPnP
    % [R,T,mask_inliers, robustiters, err] = REPPnP(Pts,impts,varargin)
    [R, t, ~, ~, ~] = REPPnP(P, impts);
    
    % 确保 t 是列向量
    if size(t, 2) > 1
        t = t';
    end
end