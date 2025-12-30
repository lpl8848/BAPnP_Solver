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
    
    % ====== 1. 生成纯净的 2D 像素点 (True 2D) ======
    % 先生成不带噪声的点，用于构建真实的 3D 结构
    pts2d_true = [w*rand(1, npts)
                  h*rand(1, npts)
                  ones(1, npts)];

    % ====== 2. 生成随机位姿 (True Pose) ======
    [U,~,V] = svd(rand(3));
    R = U * diag([1,1,det(U*V')]) * V';
    t = min_depth/2 * rand(3,1);

    % ====== 3. 反投影生成真实的 3D 点 (True 3D) ======
    % 使用纯净的 2D 点和内参进行反投影
    % 这样 pts3d 和 pts2d_true 才是完美对应的
    normalized_pts2d_true = K \ pts2d_true;
    pts3d = R' * (normalized_pts2d_true * diag(d) - t * ones(1, npts));
    
    % ====== 4. 给 2D 观测加噪声 (Measurement Noise) ======
    % 这才是算法看到的“观测数据”
    noise = noise_std * randn(2, npts);
    pts2d_noisy = pts2d_true;
    pts2d_noisy(1:2, :) = pts2d_noisy(1:2, :) + noise;

    % ====== 5. 归一化带噪坐标 (Input to Algo) ======
    % 算法拿到的是带噪声的像素点对应的归一化坐标
    normalized_pts2d_noisy = K \ pts2d_noisy;

end
