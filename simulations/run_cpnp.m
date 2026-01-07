function [R, t] = run_cpnp(y, P)
% Wrapper for CPnP

    u = y(1:2, :) ./ y(3, :);

    % fx = 1, fy = 1, u0 = 0, v0 = 0
    fx = 1;
    fy = 1;
    u0 = 0;
    v0 = 0;

    % 3.  CPnP
    % CPnP  [R,t,R_GN,t_GN] = CPnP(s, Psens_2D, fx, fy, u0, v0)
    [~, ~, R_GN, t_GN] = CPnP(P, u, fx, fy, u0, v0);

    % 4. 
    %  Gauss-Newton 
    R = R_GN;
    t = t_GN;
    
    if size(t, 2) > 1
        t = t';
    end
end

