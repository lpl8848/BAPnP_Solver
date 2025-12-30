function [R, t] = run_lhm(y, P)
% RUN_LHM LHM (Lu-Hager-Mjolsness) 算法接口封装
%
% 输入:
%   y: 3xN 归一化图像点 (齐次坐标, y(3,:)通常为1)
%   P: 3xN 世界坐标点
%
% 输出:
%   R: 3x3 旋转矩阵
%   t: 3x1 平移向量

    % 1. 数据预处理
    % LHM 需要 2xN 的非齐次归一化坐标 (u, v)
    % 将 3xN 齐次坐标转为 2xN
    xx = y(1:2, :) ./ y(3, :);
    
    % 世界坐标保持 3xN
    XX = P;
    
    % 2. 调用 LHM
    % LHM.m 内部会调用 objpose.m
    % 注意：请确保整个 'lhm' 文件夹都在 MATLAB 路径中
    [R, t] = LHM(XX, xx);
    
    % 3. 格式确保
    % 确保 t 是 3x1 列向量 (以防有些工具箱返回行向量)
    if size(t, 2) > 1
        t = t';
    end
    
    % 确保 R 是合法的旋转矩阵 (行列式为 1)
    if det(R) < 0
        [U, ~, V] = svd(R);
        R = U * diag([1 1 -1]) * V';
    end
end