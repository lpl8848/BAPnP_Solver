function [pts3d, pts2d_noisy, normalized_pts2d_noisy, K, R, t] = generate_P6P_3D_to_2D_point_correspondences_noise(npts,noise_std)

    if nargin < 1
        noise_std = 0;   % default no noise
    end
    
    
    % projective depth
    min_depth = 5;
    max_depth = 10;
    d = min_depth + (max_depth - min_depth)*rand(1, npts);

    % ====== Camera parameters ======
    w = 640;
    h = 480;
    f = 800;
    K = [f, 0, w/2
         0, f, h/2
         0, 0,  1];
    

    pts2d_true = [w*rand(1, npts)
                  h*rand(1, npts)
                  ones(1, npts)];


    [U,~,V] = svd(rand(3));
    R = U * diag([1,1,det(U*V')]) * V';
    t = min_depth/2 * rand(3,1);


    normalized_pts2d_true = K \ pts2d_true;
    pts3d = R' * (normalized_pts2d_true * diag(d) - t * ones(1, npts));
    

    noise = noise_std * randn(2, npts);
    pts2d_noisy = pts2d_true;
    pts2d_noisy(1:2, :) = pts2d_noisy(1:2, :) + noise;


    normalized_pts2d_noisy = K \ pts2d_noisy;

end

