function [r_err, t_err, is_valid] = calc_error(R_e, t_e, R_gt, t_gt)
% CALC_ERROR 计算估计位姿与真实位姿之间的误差
% 若解数量不一致或无解，不报错，返回 Inf，并标记 is_valid=false
%
% 输出:
%   r_err   : 最小旋转误差 (deg)
%   t_err   : 对应平移误差
%   is_valid: 是否存在有效解

    % ========= 默认输出（失败情形） =========
    r_err   = Inf;
    t_err   = Inf;
    is_valid = false;

    % ========= 基本合法性检查 =========
    if isempty(R_e) || isempty(t_e)
        return;
    end

    % ========= 解数量一致性处理 =========
    num_R = size(R_e, 3);

    % t_e 可能是 3xN 或 Nx3
    if size(t_e, 2) == num_R
        % OK: 3xN
    elseif size(t_e, 1) == num_R
        t_e = t_e';
    else
        % 解数量不一致：直接判为无效
        return;
    end

    num_sol = num_R;

    r_errs = zeros(num_sol, 1);
    t_errs = zeros(num_sol, 1);

    % ========= 逐解计算误差 =========
    for i = 1:num_sol
        R_est = R_e(:, :, i);
        t_est = t_e(:, i);

        % --- 1. Rotation error ---
        R_diff = R_est * R_gt';
        tr = trace(R_diff);

        % 数值保护
        tr = max(min(tr, 3), -1);

        theta = acos((tr - 1) / 2);
        r_errs(i) = rad2deg(theta);

        % --- 2. Translation error ---
        t_errs(i) = norm(t_est - t_gt);
    end

    % ========= 选择 best match =========
    [r_err, best_idx] = min(r_errs);
    t_err = t_errs(best_idx);
    is_valid = true;
end
