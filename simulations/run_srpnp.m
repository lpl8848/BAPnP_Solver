function [R, t] = run_srpnp(y, P)
% SRPnP 算法接口
% 输入:
%   y: 3xN 归一化图像点 (齐次坐标)
%   P: 3xN 世界坐标点
% 输出:
%   R: 3x3 旋转矩阵
%   t: 3x1 平移向量

    % 1. 数据预处理
    % SRPnP 需要 2xN 的非齐次归一化坐标 (u, v)
    xx = y(1:2, :) ./ y(3, :);
    
    % 世界坐标保持 3xN
    XX = P;
    
    % 2. 调用 SRPnP1 (选择最优解版本)
    % 确保 SRPnP1.m 及其依赖函数 (getp3p.m, getpoly7.m 等) 都在路径中
    [R, t] = SRPnP1(XX, xx);
    
    % 3. 格式确保 (防止输出行向量)
    if size(t, 2) > 1
        t = t';
    end
end
