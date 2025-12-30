function [R_est, t_est] = run_aqpnp(p_c, pts3D)
% RUN_AQPNP 复现 AQPnP 算法求解相机位姿
%
% 输入:
%   pts2D: 2xN 图像坐标 (像素)
%   pts3D: 3xN 世界坐标点
%   K: 3x3 相机内参矩阵
%
% 输出:
%   R_est: 3x3 旋转矩阵
%   t_est: 3x1 平移向量
%
% 参考文献: "AQPnP: an accurate and quaternion-based solution for the PnP problem", 2024

    N = size(pts3D, 2);
    if N < 3
        error('Need at least 3 points.');
    end

    
    % 球面归一化 (Spherical Normalization) - 对应论文 Eq(6)
    norms = sqrt(sum(p_c.^2, 1));
    v = p_c ./ norms; % 3xN，每个列向量 v_i 模长为 1

    % --- 2. 构建矩阵 M (2N x 13) ---
    % 优化变量 alpha = [a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2, t1, t2, t3]'
    % M 的每一行对应 v_i 的零空间向量 m_i 或 n_i 对 alpha 的约束
    
    % 为了速度，我们预分配 M
    M = zeros(2*N, 13);
    
    for i = 1:N
        Xi = pts3D(1, i);
        Yi = pts3D(2, i);
        Zi = pts3D(3, i);
        vi = v(:, i);
        
        % 计算 vi 的零空间基向量 m, n (对应论文 Eq 7)
        % 方法: 构造 Householder 变换或简单的正交补
        [U_null, ~, ~] = svd(vi); 
        % svd(vi) -> U(:,1) 是 vi 方向, U(:,2) 和 U(:,3) 是零空间
        mi = U_null(:, 2);
        ni = U_null(:, 3);
        
        % 填充 M 矩阵的行
        % 基于论文附录 A 的推导: m^T * (R*P + t) = 0
        % 旋转矩阵 R(q) 的参数化 (论文 Eq 3) 代入展开
        
        % Row for m
        row_idx_m = 2*i - 1;
        M(row_idx_m, :) = build_row_coeffs(mi, Xi, Yi, Zi);
        
        % Row for n
        row_idx_n = 2*i;
        M(row_idx_n, :) = build_row_coeffs(ni, Xi, Yi, Zi);
    end

    % --- 3. 构造约束矩阵 A ---
    % 约束: a^2 + b^2 + c^2 + d^2 = 1
    % alpha 向量中索引: 1->a^2, 5->b^2, 8->c^2, 10->d^2
    A = zeros(1, 13);
    A(1) = 1; A(5) = 1; A(8) = 1; A(10) = 1;

    % --- 4. 求解等式约束二次规划 (Equality Constrained QP) ---
    % 目标: min alpha' * (M' * M) * alpha
    % 约束: A * alpha = 1
    
    % 构造 KKT 系统求解
    % [ 2*M'M   A' ] [ alpha ]   [ 0 ]
    % [ A       0  ] [ lambda] = [ 1 ]
    
    MM = M' * M; % 13x13
    
    % 为了数值稳定性，可以加上微小的正则项
    reg = 1e-8 * eye(13);
    
    KKT_LHS = [2*MM + reg, A'; A, 0];
    KKT_RHS = [zeros(13, 1); 1];
    
    solution = KKT_LHS \ KKT_RHS;
    alpha = solution(1:13);
    
    % --- 5. 恢复旋转和平移 ---
    % alpha 前 10 项对应四元数二次项:
    % [a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]
    
    % 构造矩阵 Q_mat = q * q'
    %     a   b   c   d
    % a [ a^2 ab  ac  ad ]
    % b [ ba  b^2 bc  bd ]
    % c [ ca  cb  c^2 cd ]
    % d [ da  db  dc  d^2]
    
    vec_q = alpha(1:10);
    Q_mat = [vec_q(1), vec_q(2), vec_q(3), vec_q(4);  % a ...
             vec_q(2), vec_q(5), vec_q(6), vec_q(7);  % b ...
             vec_q(3), vec_q(6), vec_q(8), vec_q(9);  % c ...
             vec_q(4), vec_q(7), vec_q(9), vec_q(10)]; % d ...
             
    % 通过特征分解恢复四元数 q = [a, b, c, d]
    [V, D] = eig(Q_mat);
    [~, max_idx] = max(diag(D));
    q_est = V(:, max_idx);
    
    % 强制 a >= 0 (论文 Eq 33/Eq 3 下方注释)
    if q_est(1) < 0
        q_est = -q_est;
    end
    q_est = q_est / norm(q_est); % 归一化
    
    % 将四元数转换为旋转矩阵
    R_est = quat2rot_matrix(q_est);
    
    % 恢复平移 t
    % alpha 后 3 项是 t1, t2, t3
    t_est = alpha(11:13);
    
    % 可选：进一步优化 (如 Gauss-Newton)，但 AQPnP 本身作为直接求解器通常不需要
end

function row = build_row_coeffs(m, X, Y, Z)
    % 辅助函数：根据 m^T * (R*P + t) 展开系数
    % m = [mx, my, mz]'
    % P = [X, Y, Z]'
    % R 是由 q 构成的旋转矩阵 (论文 Eq 3)
    % 
    % 返回 1x13 的行向量，对应 alpha 的系数
    % alpha: [a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2, t1, t2, t3]
    
    mx = m(1); my = m(2); mz = m(3);
    
    % R * P 展开项系数推导:
    % R = [ a^2+b^2-c^2-d^2,  2bc-2ad,          2bd+2ac
    %       2bc+2ad,          a^2-b^2+c^2-d^2,  2cd-2ab
    %       2bd-2ac,          2cd+2ab,          a^2-b^2-c^2+d^2 ]
    % 注意：论文中的 Eq 3 矩阵写法可能与标准不同（行列定义），这里采用标准 Hamilton 四元数旋转公式，
    % 并与论文附录 A 的项进行核对。
    % 
    % 展开 m' * R * P:
    % = mx * (R11*X + R12*Y + R13*Z) + ...
    
    row = zeros(1, 13);
    
    % --- t 的系数 (最后 3 项) ---
    % m' * t = mx*t1 + my*t2 + mz*t3
    row(11) = mx;
    row(12) = my;
    row(13) = mz;
    
    % --- 四元数二次项系数 ---
    % 1. a^2: (mx*X + my*Y + mz*Z)
    row(1) = mx*X + my*Y + mz*Z;
    
    % 2. ab: 2*my*X - 2*mx*Y ? 
    % R12项包含 -2ad ? R21包含 2ad. 
    % 让我们按标准公式组合:
    % Term X: mx(a^2+b^2-c^2-d^2) + my(2bc+2ad) + mz(2bd-2ac)
    % Term Y: mx(2bc-2ad) + my(a^2-b^2+c^2-d^2) + mz(2cd+2ab)
    % Term Z: mx(2bd+2ac) + my(2cd-2ab) + mz(a^2-b^2-c^2+d^2)
    
    % 收集各项:
    
    % a^2: (+X)mx + (+Y)my + (+Z)mz
    row(1) = mx*X + my*Y + mz*Z;
    
    % ab: (+2Z)my + (-2Z)mx ? No.
    % Look at ab terms: From Term Y (mz*2ab) -> 2*Z*mz. From Term Z (-my*2ab) -> -2*Y*my.
    % No ab in Term X.
    % Wait, checking standard rotation R_quat(q):
    % R(1,2) = 2(ab - cd) or 2(bc - ad)? 
    % Standard: R = [1-2y2-2z2, 2xy-2zw, ...]. 
    % If q=[w, x, y, z] = [a, b, c, d].
    % R11 = a^2+b^2-c^2-d^2 (Consistent)
    % R12 = 2(bc - ad) (Standard) -> AQPnP paper Eq 3 says 2bc - 2ad. (Consistent)
    % R21 = 2(bc + ad) (Standard) -> AQPnP paper Eq 3 says 2bc + 2ad. (Consistent)
    % R31 = 2(bd - ac) (Standard) -> AQPnP paper Eq 3 says 2bd - 2ac. (Consistent)
    % R13 = 2(bd + ac) (Standard) -> AQPnP paper Eq 3 says 2bd + 2ac. (Consistent)
    % R23 = 2(cd - ab) (Standard) -> AQPnP paper Eq 3 says 2cd - 2ab. (Consistent)
    % R32 = 2(cd + ab) (Standard) -> AQPnP paper Eq 3 says 2cd + 2ab. (Consistent)
    
    % Re-evaluating coefficients for row vector:
    
    % ab terms:
    % From R23*Y*my -> (-2ab)*Y*my = -2*Y*my
    % From R32*Y*mz -> (+2ab)*Y*mz  Wait, R32 is *Y*? No R32 multiplies Y in P? No.
    % R * P = [R11 X + R12 Y + R13 Z; ...]
    % Term 1 (row 1 of result): mx * (...) -> mx*(R11 X + R12 Y + R13 Z)
    % Term 2 (row 2 of result): my * (...) -> my*(R21 X + R22 Y + R23 Z)
    % Term 3 (row 3 of result): mz * (...) -> mz*(R31 X + R32 Y + R33 Z)
    
    % Coefficient of ab:
    % In Term 1: 0
    % In Term 2: my * Z * (-2) = -2*Z*my
    % In Term 3: mz * Y * (+2) = +2*Y*mz
    row(2) = 2*Y*mz - 2*Z*my;
    
    % ac terms:
    % In Term 1: mx * Z * (+2) = 2*Z*mx
    % In Term 2: 0
    % In Term 3: mz * X * (-2) = -2*X*mz
    row(3) = 2*Z*mx - 2*X*mz;
    
    % ad terms:
    % In Term 1: mx * Y * (-2) = -2*Y*mx
    % In Term 2: my * X * (+2) = 2*X*my
    % In Term 3: 0
    row(4) = 2*X*my - 2*Y*mx;
    
    % b^2 terms:
    % Term 1: mx * X * (+1) = mx*X
    % Term 2: my * Y * (-1) = -my*Y
    % Term 3: mz * Z * (-1) = -mz*Z
    row(5) = mx*X - my*Y - mz*Z;
    
    % bc terms:
    % Term 1: mx * Y * (+2) = 2*Y*mx
    % Term 2: my * X * (+2) = 2*X*my
    % Term 3: 0
    row(6) = 2*Y*mx + 2*X*my;
    
    % bd terms:
    % Term 1: mx * Z * (+2) = 2*Z*mx
    % Term 2: 0
    % Term 3: mz * X * (+2) = 2*X*mz
    row(7) = 2*Z*mx + 2*X*mz;
    
    % c^2 terms:
    % Term 1: mx * X * (-1) = -mx*X
    % Term 2: my * Y * (+1) = my*Y
    % Term 3: mz * Z * (-1) = -mz*Z
    row(8) = -mx*X + my*Y - mz*Z;
    
    % cd terms:
    % Term 1: 0
    % Term 2: my * Z * (+2) = 2*Z*my
    % Term 3: mz * Y * (+2) = 2*Y*mz
    row(9) = 2*Z*my + 2*Y*mz;
    
    % d^2 terms:
    % Term 1: mx * X * (-1) = -mx*X
    % Term 2: my * Y * (-1) = -my*Y
    % Term 3: mz * Z * (+1) = mz*Z
    row(10) = -mx*X - my*Y + mz*Z;
    
end

function R = quat2rot_matrix(q)
    % q = [a, b, c, d]
    a = q(1); b = q(2); c = q(3); d = q(4);
    
    R = [a^2+b^2-c^2-d^2, 2*b*c-2*a*d,     2*b*d+2*a*c;
         2*b*c+2*a*d,     a^2-b^2+c^2-d^2, 2*c*d-2*a*b;
         2*b*d-2*a*c,     2*c*d+2*a*b,     a^2-b^2-c^2+d^2];
end
