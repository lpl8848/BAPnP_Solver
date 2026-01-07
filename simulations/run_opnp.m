function [R, t] = run_opnp(y, P)

    
    u = y(1:2, :) ./ y(3, :);
    
    
    % [R0 t0 error0 flag] = OPnP(U,u,label_polish)
    [R, t, ~, ~] = OPnP(P, u);
    
   
    if size(t, 2) > 1
        t = t';
    end
end

