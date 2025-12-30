function [R, t] = run_mlpnp(y, P)
%
% 引用: Urban et al., "MLPnP - A Real-Time Maximum Likelihood Solution to the Perspective-n-Point Problem", 2016.
%
% 输入:
%   y: 3xN 归一化图像点 (齐次坐标, y(3,:)通常为1)
%   P: 3xN 世界坐标点
%
% 输出:
%   R: 3x3 旋转矩阵
%   t: 3x1 平移向量

    % 1. 数据预处理
    % MLPnP 需要单位化的视线向量 (Unit Bearing Vectors)
    bearings = y;
    norms = sqrt(sum(bearings.^2, 1));
    bearings = bearings ./ norms;
    
    XX = P;
    
    % 2. 调用 MLPnP
    % 修正点：MLPnP 返回的是一个 3x4 的变换矩阵 T，而不是 [R, t]
    try
        % 尝试直接调用 MLPnP (无协方差模式)
        T = MLPnP(XX, bearings); 
        
        % 3. 拆解结果
        R = T(1:3, 1:3);
        t = T(1:3, 4);
        
    catch
        % 如果直接调用失败，尝试调用包装器 MLPNP_without_COV
        if exist('MLPNP_without_COV', 'file')
            [R, t] = MLPNP_without_COV(XX, bearings); % 这个函数内部已经做了拆解
        else
            error('MLPnP calling failed. Please check the MLPnP toolbox path.');
        end
    end
    
    % 4. 格式标准化
    % 确保 t 是 3x1 列向量
    if size(t, 2) > 1
        t = t';
    end
    
    % 确保 R 是合法的旋转矩阵
    if det(R) < 0
        [U, ~, V] = svd(R);
        R = U * diag([1 1 -1]) * V';
    end
end
