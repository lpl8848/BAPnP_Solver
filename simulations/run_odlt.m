function [R, t] = run_odlt(y_norm, P_world)
% RUN_ODLT Wrapper for oDLT solver (Henry & Christian, 2024)
% Note: pnp_odlt.m contains function pnp_odlt_vfast_normalized
% MATLAB R2020b requires calling by filename, not function name

    N = size(P_world, 2);
    K = [800, 0, 320; 0, 800, 240; 0, 0, 1];

    if size(y_norm, 1) == 2
        y_h = [y_norm; ones(1, N)];
    else
        y_h = y_norm ./ y_norm(3, :);
    end

    U_pix = (K * y_h)';     % Nx3 pixel coords
    X_world = P_world';     % Nx3 world coords

    [R, t] = pnp_odlt(X_world, U_pix, K);
end
