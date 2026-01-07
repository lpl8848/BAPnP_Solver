function [R, t] = run_epnp_guass(y, P)
% EPnP (Gauss-Newton) 


    n = size(P, 2);
    

    Xw_h = [P', ones(n, 1)]; 
    

    x2d_h = y'; % Nx3
    

    A = eye(3);
    

    [R, t, ~, ~] = efficient_pnp_gauss(Xw_h, x2d_h, A);
   %[R, t, ~, ~] = efficient_pnp(Xw_h, x2d_h, A);

    if size(t, 2) > 1
        t = t';
    end
end

