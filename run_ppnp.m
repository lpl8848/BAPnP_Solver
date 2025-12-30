function [R, t] = run_ppnp(y, P)
% PPnP 接口封装
% 输入:
%   y: 3xN 归一化图像点 (齐次坐标)
%   P: 3xN 世界坐标点
% 输出:
%   R: 3x3 旋转矩阵
%   t: 3x1 平移向量

    % 1. 准备数据，PPnP 需要 nx3 的输入
    % P_img: nx3 (u, v, 1)
    % S_world: nx3 (X, Y, Z)
    P_img = y'; 
    S_world = P';
    
    % 设定收敛容差
    tol = 1e-6;
    
    % 2. 调用原始 ppnp
    [R, t] = ppnp(P_img, S_world, tol);
    
    % 确保 t 是列向量
    if size(t, 2) > 1
        t = t';
    end
end