function [R, t] = run_epnp(y, P)
% EPnP 

    xx = y(1:2, :) ./ y(3, :);
    
    % 2.  EPnP
    % [R,t] = EPnP(XX,xx)
    [R, t] = EPnP(P, xx);
    
    
    if size(t, 2) > 1
        t = t';
    end
end

