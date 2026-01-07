function [R, t] = run_srpnp(y, P)



    xx = y(1:2, :) ./ y(3, :);
    
   
    XX = P;
    

    [R, t] = SRPnP1(XX, xx);
    

    if size(t, 2) > 1
        t = t';
    end
end

