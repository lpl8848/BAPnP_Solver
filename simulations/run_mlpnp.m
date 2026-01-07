function [R, t] = run_mlpnp(y, P)
%
%  Urban et al., "MLPnP - A Real-Time Maximum Likelihood Solution to the Perspective-n-Point Problem", 2016.



    bearings = y;
    norms = sqrt(sum(bearings.^2, 1));
    bearings = bearings ./ norms;
    
    XX = P;
    

    try
       
        T = MLPnP(XX, bearings); 
        
        
        R = T(1:3, 1:3);
        t = T(1:3, 4);
        
    catch
      
        if exist('MLPNP_without_COV', 'file')
            [R, t] = MLPNP_without_COV(XX, bearings); 
        else
            error('MLPnP calling failed. Please check the MLPnP toolbox path.');
        end
    end
    
 
    if size(t, 2) > 1
        t = t';
    end
    
   
    if det(R) < 0
        [U, ~, V] = svd(R);
        R = U * diag([1 1 -1]) * V';
    end
end

