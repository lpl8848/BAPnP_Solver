function [R, t] = run_odlt_planar_gn(y_norm, P_world)
% oDLT + GN wrapper for planar experiment (K=I)
    N = size(P_world, 2);
    if size(y_norm, 1) == 2
        y_h = [y_norm; ones(1, N)];
    else
        y_h = y_norm ./ y_norm(3, :);
    end
    U_pix = y_h';
    X_world = P_world';
    K = eye(3);
    [R_init, t_init] = pnp_odlt(X_world, U_pix, K);
    [R, t] = optimize_pose_gn(X_world, U_pix, K, R_init, t_init);
end
