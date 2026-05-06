function [R, t] = run_odlt_gn(y_norm, P_world)
% oDLT + Gauss-Newton refinement (fair comparison with BAPnP-GN)
% Uses oDLT's own optimize_pose_gn for refinement
    N = size(P_world, 2);
    K = [800, 0, 320; 0, 800, 240; 0, 0, 1];

    if size(y_norm, 1) == 2
        y_h = [y_norm; ones(1, N)];
    else
        y_h = y_norm ./ y_norm(3, :);
    end

    U_pix = (K * y_h)';
    X_world = P_world';

    % Linear oDLT
    [R_init, t_init] = pnp_odlt(X_world, U_pix, K);

    % GN refinement (oDLT's own implementation from CPnP paper)
    [R, t] = optimize_pose_gn(X_world, U_pix, K, R_init, t_init);
end
