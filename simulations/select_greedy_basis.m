function [base_idx]=select_greedy_basis(P_world)    
N = size(P_world, 2);
    
    % 2. 3D 数据归一化
    cent_3d = mean(P_world, 2);
    P_centered = P_world - cent_3d;
    sq_dists = sum(P_centered.^2, 1);
    rms_dist = sqrt(sum(sq_dists) / N);
    if rms_dist < 1e-6, rms_dist = 1; end
    scale_3d = 1.732050807568877 / rms_dist; 
    P_n = P_centered * scale_3d;
    
    % 3. 极速基底选择 (RPnP Style - 保持不变)
    base_idx = zeros(1, 4);
    [~, base_idx(1)] = max(sq_dists);
    p1 = P_n(:, base_idx(1));
    
    d2 = sum((P_n - p1).^2, 1);
    [~, base_idx(2)] = max(d2);
    p2 = P_n(:, base_idx(2));
    
    v12 = p2 - p1;
    v12_sq = sum(v12.^2); if v12_sq < 1e-8, v12_sq = 1; end
    
    vecs = P_n - p1;
    cp_x = v12(2)*vecs(3,:) - v12(3)*vecs(2,:);
    cp_y = v12(3)*vecs(1,:) - v12(1)*vecs(3,:);
    cp_z = v12(1)*vecs(2,:) - v12(2)*vecs(1,:);
    d2_line = cp_x.^2 + cp_y.^2 + cp_z.^2;
    [~, base_idx(3)] = max(d2_line);
    p3 = P_n(:, base_idx(3));
    
    
    
    
    v13 = p3 - p1;
    nx = v12(2)*v13(3) - v12(3)*v13(2);
    ny = v12(3)*v13(1) - v12(1)*v13(3);
    nz = v12(1)*v13(2) - v12(2)*v13(1);
    d2_plane = (nx*vecs(1,:) + ny*vecs(2,:) + nz*vecs(3,:)).^2;
    [~, base_idx(4)] = max(d2_plane);
    
end