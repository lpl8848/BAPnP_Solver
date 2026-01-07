function [R, t] = run_rpnp(y, P)


   
    xx = y(1:2, :) ./ y(3, :);
    

    [R, t] = RPnP(P, xx);
    
   
    if size(t, 2) > 1
        t = t';
    end
end

