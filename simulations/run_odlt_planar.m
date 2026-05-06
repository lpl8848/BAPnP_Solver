function [R, t] = run_odlt_planar(y_norm, P_world)
% oDLT wrapper for planar experiment - uses K=I since coords already normalized
    N = size(P_world, 2);
    if size(y_norm, 1) == 2
        y_h = [y_norm; ones(1, N)];
    else
        y_h = y_norm ./ y_norm(3, :);
    end
    U_pix = y_h';  % K=I for planar experiment
    X_world = P_world';
    K = eye(3);
    [R, t] = pnp_odlt(X_world, U_pix, K);
end
