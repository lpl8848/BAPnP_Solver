function [R, t] = sqpnp(y, P)
% SQPNP 求解 Perspective-n-Point (PnP) 问题
% 基于 G. Terzakis 和 M. Lourakis 提出的 SQPnP 算法
% 
% 输入:
%   y: 3xN 归一化图像点 (齐次坐标 [x; y; 1])
%   P: 3xN 世界坐标点
% 输出:
%   R: 3x3 旋转矩阵
%   t: 3x1 平移向量

    %% 1. 输入验证与数据转换
    if size(y, 1) ~= 3 || size(P, 1) ~= 3
        error('输入 y 和 P 必须是 3xN 矩阵。');
    end
    n = size(y, 2);
    if n ~= size(P, 2) || n < 3
        error('点对数量不匹配或少于 3 个点。');
    end

    % 转换为 Nx3 和 Nx2 (匹配原C++作者的内部结构)
    points_3d = P';
    projections_2d = y(1:2, :)';
    weights = ones(n, 1); % 默认权重为 1
    
    %% 2. 初始化参数与状态变量 (相当于C++中的类成员)
    parameters.rank_tolerance = 1e-7;
    parameters.sqp_squared_tolerance = 1e-10;
    parameters.sqp_det_threshold = 1.001;
    parameters.sqp_max_iteration = 15;
    parameters.nearest_rotation_method = 'FOAM';
    parameters.orthogonality_squared_error_threshold = 1e-8;
    parameters.equal_vectors_squared_diff = 1e-10;
    parameters.equal_squared_errors_diff = 1e-6;
    
    Omega = zeros(9, 9);
    P_mat = zeros(3, 9);
    solutions = repmat(struct('r', zeros(9,1), 'r_hat', zeros(9,1), 't', zeros(3,1), 'num_iterations', 0, 'sq_error', 0), 18, 1);
    num_solutions = 0;
    
    %% 3. 构造系统 (原C++中的 Constructor 逻辑)
    sum_wx = 0.0; sum_wy = 0.0; sum_wx2_plus_wy2 = 0.0; sum_w = 0.0;
    sum_wX = 0.0; sum_wY = 0.0; sum_wZ = 0.0;
    QA = zeros(3, 9);
    
    for i = 1:n
        w = weights(i);
        if w == 0.0, continue; end
        
        wx = w * projections_2d(i, 1);
        wy = w * projections_2d(i, 2);
        wsq_norm_m = w * (projections_2d(i, 1)^2 + projections_2d(i, 2)^2);
        
        sum_wx = sum_wx + wx;
        sum_wy = sum_wy + wy;
        sum_wx2_plus_wy2 = sum_wx2_plus_wy2 + wsq_norm_m;
        sum_w = sum_w + w;
        
        X = points_3d(i, 1); Y = points_3d(i, 2); Z = points_3d(i, 3);
        wX = w * X; wY = w * Y; wZ = w * Z;
        
        sum_wX = sum_wX + wX; sum_wY = sum_wY + wY; sum_wZ = sum_wZ + wZ;
        X2 = X*X; XY = X*Y; XZ = X*Z; Y2 = Y*Y; YZ = Y*Z; Z2 = Z*Z;
        
        % Block (1:3, 1:3) 
        Omega(1,1) = Omega(1,1) + w*X2; Omega(1,2) = Omega(1,2) + w*XY; Omega(1,3) = Omega(1,3) + w*XZ;
        Omega(2,2) = Omega(2,2) + w*Y2; Omega(2,3) = Omega(2,3) + w*YZ; Omega(3,3) = Omega(3,3) + w*Z2;
        
        % Block (1:3, 7:9) 
        Omega(1,7) = Omega(1,7) - wx*X2; Omega(1,8) = Omega(1,8) - wx*XY; Omega(1,9) = Omega(1,9) - wx*XZ;
        Omega(2,8) = Omega(2,8) - wx*Y2; Omega(2,9) = Omega(2,9) - wx*YZ; Omega(3,9) = Omega(3,9) - wx*Z2;
        
        % Block (4:6, 7:9) 
        Omega(4,7) = Omega(4,7) - wy*X2; Omega(4,8) = Omega(4,8) - wy*XY; Omega(4,9) = Omega(4,9) - wy*XZ;
        Omega(5,8) = Omega(5,8) - wy*Y2; Omega(5,9) = Omega(5,9) - wy*YZ; Omega(6,9) = Omega(6,9) - wy*Z2;
        
        % Block (7:9, 7:9) 
        Omega(7,7) = Omega(7,7) + wsq_norm_m*X2; Omega(7,8) = Omega(7,8) + wsq_norm_m*XY; Omega(7,9) = Omega(7,9) + wsq_norm_m*XZ;
        Omega(8,8) = Omega(8,8) + wsq_norm_m*Y2; Omega(8,9) = Omega(8,9) + wsq_norm_m*YZ; Omega(9,9) = Omega(9,9) + wsq_norm_m*Z2;
        
        % Accumulating QA
        QA(1,1) = QA(1,1) + wX; QA(1,2) = QA(1,2) + wY; QA(1,3) = QA(1,3) + wZ;   
        QA(1,7) = QA(1,7) - wx*X; QA(1,8) = QA(1,8) - wx*Y; QA(1,9) = QA(1,9) - wx*Z;
        QA(2,7) = QA(2,7) - wy*X; QA(2,8) = QA(2,8) - wy*Y; QA(2,9) = QA(2,9) - wy*Z;
        QA(3,7) = QA(3,7) + wsq_norm_m*X; QA(3,8) = QA(3,8) + wsq_norm_m*Y; QA(3,9) = QA(3,9) + wsq_norm_m*Z;
    end
    
    % Complete QA
    QA(2,4) = QA(1,1); QA(2,5) = QA(1,2); QA(2,6) = QA(1,3);
    QA(3,1) = QA(1,7); QA(3,2) = QA(1,8); QA(3,3) = QA(1,9);
    QA(3,4) = QA(2,7); QA(3,5) = QA(2,8); QA(3,6) = QA(2,9);
    
    % Fill-in off-diagonal
    Omega(2,7) = Omega(1,8); Omega(3,7) = Omega(1,9); Omega(3,8) = Omega(2,9);
    Omega(5,7) = Omega(4,8); Omega(6,7) = Omega(4,9); Omega(6,8) = Omega(5,9);
    Omega(8,7) = Omega(7,8); Omega(9,7) = Omega(7,9); Omega(9,8) = Omega(8,9);
    
    % Fill-in upper triangle of block (4:6, 4:6)
    Omega(4,4) = Omega(1,1); Omega(4,5) = Omega(1,2); Omega(4,6) = Omega(1,3);
    Omega(5,5) = Omega(2,2); Omega(5,6) = Omega(2,3); Omega(6,6) = Omega(3,3);
    
    % Fill lower triangle
    Omega(2,1) = Omega(1,2);
    Omega(3,1) = Omega(1,3); Omega(3,2) = Omega(2,3);
    Omega(4,1) = Omega(1,4); Omega(4,2) = Omega(2,4); Omega(4,3) = Omega(3,4);
    Omega(5,1) = Omega(1,5); Omega(5,2) = Omega(2,5); Omega(5,3) = Omega(3,5); Omega(5,4) = Omega(4,5);
    Omega(6,1) = Omega(1,6); Omega(6,2) = Omega(2,6); Omega(6,3) = Omega(3,6); Omega(6,4) = Omega(4,6); Omega(6,5) = Omega(5,6);
    Omega(7,1) = Omega(1,7); Omega(7,2) = Omega(2,7); Omega(7,3) = Omega(3,7); Omega(7,4) = Omega(4,7); Omega(7,5) = Omega(5,7); Omega(7,6) = Omega(6,7);
    Omega(8,1) = Omega(1,8); Omega(8,2) = Omega(2,8); Omega(8,3) = Omega(3,8); Omega(8,4) = Omega(4,8); Omega(8,5) = Omega(5,8); Omega(8,6) = Omega(6,8);
    Omega(9,1) = Omega(1,9); Omega(9,2) = Omega(2,9); Omega(9,3) = Omega(3,9); Omega(9,4) = Omega(4,9); Omega(9,5) = Omega(5,9); Omega(9,6) = Omega(6,9);
    
    Q = [sum_w, 0.0, -sum_wx;
         0.0, sum_w, -sum_wy;
         -sum_wx, -sum_wy, sum_wx2_plus_wy2];
    
    Qinv = InvertSymmetric3x3(Q);
    P_mat = -Qinv * QA;
    Omega = Omega + QA' * P_mat;
    
    % Decompose Omega (Using SVD)
    [U, S_mat, ~] = svd(Omega);
    s = diag(S_mat);
    
    num_null_vectors = 0;
    for i = 9:-1:1
        if s(i) < parameters.rank_tolerance
            num_null_vectors = num_null_vectors + 1;
        else
            break;
        end
    end
    
    if num_null_vectors > 6
        error('SQPnP: Invalid null space dimension (>6). Data might be collinear/coplanar in a degenerate way.');
    end
    
    inv_sum_w = 1.0 / sum_w;
    point_mean = [sum_wX * inv_sum_w; sum_wY * inv_sum_w; sum_wZ * inv_sum_w];
    
    %% 4. PnP 求解核心 (原C++中的 Solve() 逻辑)
    min_sq_error = realmax;
    num_eigen_points = max(1, num_null_vectors);
    SQRT3 = sqrt(3.0);
    
    for i = (10 - num_eigen_points) : 9
        e = SQRT3 * U(:, i);
        orthogonality_sq_error = OrthogonalityError(e);
        
        sol = struct('r', zeros(9,1), 'r_hat', zeros(9,1), 't', zeros(3,1), 'num_iterations', 0, 'sq_error', 0);
        
        if orthogonality_sq_error < parameters.orthogonality_squared_error_threshold
            sol.r_hat = Determinant9x1(e) * e;
            sol.t = P_mat * sol.r_hat;
            sol.num_iterations = 0;
            min_sq_error = HandleSolution(sol, min_sq_error);
        else
            sol.r = NearestRotationMatrix(e);
            sol = RunSQP(sol.r);
            sol.t = P_mat * sol.r_hat;
            min_sq_error = HandleSolution(sol, min_sq_error);
            
            sol2.r = NearestRotationMatrix(-e);
            sol2 = RunSQP(sol2.r);
            sol2.t = P_mat * sol2.r_hat;
            min_sq_error = HandleSolution(sol2, min_sq_error);
        end
    end
    
    c = 1;
    index = 10 - num_eigen_points - c;
    while index >= 1 && min_sq_error > 3 * s(index)
        e = U(:, index);
        
        sol.r = NearestRotationMatrix(e);
        sol = RunSQP(sol.r);
        sol.t = P_mat * sol.r_hat;
        min_sq_error = HandleSolution(sol, min_sq_error);
        
        sol2.r = NearestRotationMatrix(-e);
        sol2 = RunSQP(sol2.r);
        sol2.t = P_mat * sol2.r_hat;
        min_sq_error = HandleSolution(sol2, min_sq_error);
        
        c = c + 1;
        index = 10 - num_eigen_points - c;
    end
    
    %% 5. 返回结果：选取最佳解 (Best Error)
    if num_solutions > 0
        best_sol = solutions(1);
        % 原作者用的是 9x1 row-major 平铺，此处 reshape 并转置回 3x3
        R = reshape(best_sol.r_hat, 3, 3)'; 
        t = best_sol.t;
    else
        R = [];
        t = [];
    end

    % =====================================================================
    % ======================== 嵌套辅助函数 (Nested) ========================
    % 它们可以直接读写外部变量 (Omega, P_mat, point_mean, solutions...)
    % =====================================================================
    
    function min_err = HandleSolution(solution, min_err)
        cheirok = TestPositiveDepth(solution) || TestPositiveMajorityDepths(solution);
        if cheirok
            solution.sq_error = (Omega * solution.r_hat)' * solution.r_hat;
            if abs(min_err - solution.sq_error) > parameters.equal_squared_errors_diff
                if min_err > solution.sq_error
                    min_err = solution.sq_error;
                    solutions(1) = solution;
                    num_solutions = 1;
                end
            else
                found = false;
                for k = 1:num_solutions
                    if sum((solutions(k).r_hat - solution.r_hat).^2) < parameters.equal_vectors_squared_diff
                        if solutions(k).sq_error > solution.sq_error
                            solutions(k) = solution;
                        end
                        found = true;
                        break;
                    end
                end
                if ~found
                    num_solutions = num_solutions + 1;
                    solutions(num_solutions) = solution;
                end
                if min_err > solution.sq_error
                    min_err = solution.sq_error;
                end
            end
        end
    end
    
    function solution = RunSQP(r0)
        r = r0;
        delta_squared_norm = realmax;
        step = 0;
        
        while delta_squared_norm > parameters.sqp_squared_tolerance && step < parameters.sqp_max_iteration
            delta = SolveSQPSystem(r);
            r = r + delta;
            delta_squared_norm = sum(delta.^2);
            step = step + 1;
        end
        
        solution.num_iterations = step;
        solution.r = r;
        
        det_r = Determinant9x1(solution.r);
        if det_r < 0
            solution.r = -r;
            det_r = -det_r;
        end
        
        if det_r > parameters.sqp_det_threshold
            solution.r_hat = NearestRotationMatrix(solution.r);
        else
            solution.r_hat = solution.r;
        end
    end

    function delta = SolveSQPSystem(r)
        sqnorm_r1 = r(1)^2 + r(2)^2 + r(3)^2;
        sqnorm_r2 = r(4)^2 + r(5)^2 + r(6)^2;
        sqnorm_r3 = r(7)^2 + r(8)^2 + r(9)^2;
        
        dot_r1r2 = r(1)*r(4) + r(2)*r(5) + r(3)*r(6);
        dot_r1r3 = r(1)*r(7) + r(2)*r(8) + r(3)*r(9);
        dot_r2r3 = r(4)*r(7) + r(5)*r(8) + r(6)*r(9);
        
        [H, N, JH] = RowAndNullSpace(r, 0.1);
        
        g = zeros(6,1);
        g(1) = 1 - sqnorm_r1; g(2) = 1 - sqnorm_r2; g(3) = 1 - sqnorm_r3;
        g(4) = -dot_r1r2;     g(5) = -dot_r2r3;     g(6) = -dot_r1r3;
        
        x = zeros(6,1);
        x(1) = g(1) / JH(1,1);
        x(2) = g(2) / JH(2,2);
        x(3) = g(3) / JH(3,3);
        x(4) = (g(4) - JH(4,1)*x(1) - JH(4,2)*x(2)) / JH(4,4);
        x(5) = (g(5) - JH(5,2)*x(2) - JH(5,3)*x(3) - JH(5,4)*x(4)) / JH(5,5);
        x(6) = (g(6) - JH(6,1)*x(1) - JH(6,3)*x(3) - JH(6,4)*x(4) - JH(6,5)*x(5)) / JH(6,6);
        
        delta = H * x;
        
        NtOmega = N' * Omega;
        W = NtOmega * N;
        rhs = -(NtOmega * (delta + r));
        
        y = W \ rhs;
        delta = delta + N * y;
    end

    function [H, N, K] = RowAndNullSpace(r, norm_threshold)
        H = zeros(9, 6); K = zeros(6, 6); N = zeros(9, 3);
        
        norm_r1 = sqrt(r(1)^2 + r(2)^2 + r(3)^2);
        inv_norm_r1 = 0; if norm_r1 > 1e-5, inv_norm_r1 = 1.0 / norm_r1; end
        H(1,1) = r(1)*inv_norm_r1; H(2,1) = r(2)*inv_norm_r1; H(3,1) = r(3)*inv_norm_r1;
        K(1,1) = 2*norm_r1;
        
        norm_r2 = sqrt(r(4)^2 + r(5)^2 + r(6)^2);
        inv_norm_r2 = 1.0 / norm_r2;
        H(4,2) = r(4)*inv_norm_r2; H(5,2) = r(5)*inv_norm_r2; H(6,2) = r(6)*inv_norm_r2;
        K(2,1) = 0; K(2,2) = 2*norm_r2;
        
        norm_r3 = sqrt(r(7)^2 + r(8)^2 + r(9)^2);
        inv_norm_r3 = 1.0 / norm_r3;
        H(7,3) = r(7)*inv_norm_r3; H(8,3) = r(8)*inv_norm_r3; H(9,3) = r(9)*inv_norm_r3;
        K(3,1) = 0; K(3,2) = 0; K(3,3) = 2*norm_r3;
        
        dot_j4q1 = r(4)*H(1,1) + r(5)*H(2,1) + r(6)*H(3,1);
        dot_j4q2 = r(1)*H(4,2) + r(2)*H(5,2) + r(3)*H(6,2);
        H(1,4) = r(4) - dot_j4q1*H(1,1); H(2,4) = r(5) - dot_j4q1*H(2,1); H(3,4) = r(6) - dot_j4q1*H(3,1);
        H(4,4) = r(1) - dot_j4q2*H(4,2); H(5,4) = r(2) - dot_j4q2*H(5,2); H(6,4) = r(3) - dot_j4q2*H(6,2);
        inv_norm_j4 = 1.0 / sqrt(H(1,4)^2 + H(2,4)^2 + H(3,4)^2 + H(4,4)^2 + H(5,4)^2 + H(6,4)^2);
        H(1:6,4) = H(1:6,4) * inv_norm_j4;
        
        K(4,1) = r(4)*H(1,1) + r(5)*H(2,1) + r(6)*H(3,1); K(4,2) = r(1)*H(4,2) + r(2)*H(5,2) + r(3)*H(6,2);
        K(4,3) = 0; K(4,4) = r(4)*H(1,4) + r(5)*H(2,4) + r(6)*H(3,4) + r(1)*H(4,4) + r(2)*H(5,4) + r(3)*H(6,4);
        
        dot_j5q2 = r(7)*H(4,2) + r(8)*H(5,2) + r(9)*H(6,2);
        dot_j5q3 = r(4)*H(7,3) + r(5)*H(8,3) + r(6)*H(9,3);
        dot_j5q4 = r(7)*H(4,4) + r(8)*H(5,4) + r(9)*H(6,4);
        
        H(1,5) = -dot_j5q4*H(1,4); H(2,5) = -dot_j5q4*H(2,4); H(3,5) = -dot_j5q4*H(3,4);
        H(4,5) = r(7) - dot_j5q2*H(4,2) - dot_j5q4*H(4,4); H(5,5) = r(8) - dot_j5q2*H(5,2) - dot_j5q4*H(5,4); H(6,5) = r(9) - dot_j5q2*H(6,2) - dot_j5q4*H(6,4);
        H(7,5) = r(4) - dot_j5q3*H(7,3); H(8,5) = r(5) - dot_j5q3*H(8,3); H(9,5) = r(6) - dot_j5q3*H(9,3);
        H(:,5) = H(:,5) / norm(H(:,5));
        
        K(5,1) = 0; K(5,2) = r(7)*H(4,2) + r(8)*H(5,2) + r(9)*H(6,2); K(5,3) = r(4)*H(7,3) + r(5)*H(8,3) + r(6)*H(9,3);
        K(5,4) = r(7)*H(4,4) + r(8)*H(5,4) + r(9)*H(6,4);
        K(5,5) = r(7)*H(4,5) + r(8)*H(5,5) + r(9)*H(6,5) + r(4)*H(7,5) + r(5)*H(8,5) + r(6)*H(9,5);
        
        dot_j6q1 = r(7)*H(1,1) + r(8)*H(2,1) + r(9)*H(3,1);
        dot_j6q3 = r(1)*H(7,3) + r(2)*H(8,3) + r(3)*H(9,3);
        dot_j6q4 = r(7)*H(1,4) + r(8)*H(2,4) + r(9)*H(3,4);
        dot_j6q5 = r(1)*H(7,5) + r(2)*H(8,5) + r(3)*H(9,5) + r(7)*H(1,5) + r(8)*H(2,5) + r(9)*H(3,5);
        
        H(1,6) = r(7) - dot_j6q1*H(1,1) - dot_j6q4*H(1,4) - dot_j6q5*H(1,5);
        H(2,6) = r(8) - dot_j6q1*H(2,1) - dot_j6q4*H(2,4) - dot_j6q5*H(2,5);
        H(3,6) = r(9) - dot_j6q1*H(3,1) - dot_j6q4*H(3,4) - dot_j6q5*H(3,5);
        H(4,6) = -dot_j6q5*H(4,5) - dot_j6q4*H(4,4);
        H(5,6) = -dot_j6q5*H(5,5) - dot_j6q4*H(5,4);
        H(6,6) = -dot_j6q5*H(6,5) - dot_j6q4*H(6,4);
        H(7,6) = r(1) - dot_j6q3*H(7,3) - dot_j6q5*H(7,5);
        H(8,6) = r(2) - dot_j6q3*H(8,3) - dot_j6q5*H(8,5);
        H(9,6) = r(3) - dot_j6q3*H(9,3) - dot_j6q5*H(9,5);
        H(:,6) = H(:,6) / norm(H(:,6));
        
        K(6,1) = r(7)*H(1,1) + r(8)*H(2,1) + r(9)*H(3,1); K(6,2) = 0; K(6,3) = r(1)*H(7,3) + r(2)*H(8,3) + r(3)*H(9,3);
        K(6,4) = r(7)*H(1,4) + r(8)*H(2,4) + r(9)*H(3,4); 
        K(6,5) = r(7)*H(1,5) + r(8)*H(2,5) + r(9)*H(3,5) + r(1)*H(7,5) + r(2)*H(8,5) + r(3)*H(9,5);
        K(6,6) = r(7)*H(1,6) + r(8)*H(2,6) + r(9)*H(3,6) + r(1)*H(7,6) + r(2)*H(8,6) + r(3)*H(9,6);
        
        Pn = eye(9) - H * H';
        
        index1 = -1; index2 = -1; index3 = -1;
        max_norm1 = -realmax; min_dot12 = realmax; min_dot1323 = realmax;
        
        col_norms = zeros(9,1);
        for k = 1:9
            col_norms(k) = norm(Pn(:,k));
            if col_norms(k) >= norm_threshold
                if max_norm1 < col_norms(k)
                    max_norm1 = col_norms(k);
                    index1 = k;
                end
            end
        end
        
        v1 = Pn(:, index1);
        N(:,1) = v1 * (1.0 / max_norm1);
        col_norms(index1) = -1.0;
        
        for k = 1:9
            if col_norms(k) >= norm_threshold
                cos_v1_x_col = abs(dot(Pn(:,k), v1) / col_norms(k));
                if cos_v1_x_col <= min_dot12
                    index2 = k;
                    min_dot12 = cos_v1_x_col;
                end
            end
        end
        
        v2 = Pn(:, index2);
        N(:,2) = v2 - dot(v2, N(:,1)) * N(:,1);
        N(:,2) = N(:,2) / norm(N(:,2));
        col_norms(index2) = -1.0;
        
        for k = 1:9
            if col_norms(k) >= norm_threshold
                inv_norm_c = 1.0 / col_norms(k);
                cos_v1_x_col = abs(dot(Pn(:,k), v1) * inv_norm_c);
                cos_v2_x_col = abs(dot(Pn(:,k), v2) * inv_norm_c);
                if cos_v1_x_col + cos_v2_x_col <= min_dot1323
                    index3 = k;
                    min_dot1323 = cos_v1_x_col + cos_v2_x_col;
                end
            end
        end
        
        v3 = Pn(:, index3);
        N(:,3) = v3 - dot(v3, N(:,2)) * N(:,2) - dot(v3, N(:,1)) * N(:,1);
        N(:,3) = N(:,3) / norm(N(:,3));
    end

    function r = NearestRotationMatrix(e)
        if strcmp(parameters.nearest_rotation_method, 'FOAM')
            r = NearestRotationMatrix_FOAM(e);
        else
            r = NearestRotationMatrix_SVD(e);
        end
    end

    function res = TestPositiveDepth(solution)
        r = solution.r_hat; t_vec = solution.t; M = point_mean;
        res = (r(7)*M(1) + r(8)*M(2) + r(9)*M(3) + t_vec(3) > 0);
    end

    function res = TestPositiveMajorityDepths(solution)
        r = solution.r_hat; t_vec = solution.t;
        npos = 0; nneg = 0;
        for k = 1:n
            if weights(k) == 0.0, continue; end
            M = points_3d(k, :);
            if r(7)*M(1) + r(8)*M(2) + r(9)*M(3) + t_vec(3) > 0
                npos = npos + 1;
            else
                nneg = nneg + 1;
            end
        end
        res = npos >= nneg;
    end

    function Qinv = InvertSymmetric3x3(Q_sym)
        a = Q_sym(1,1); b = Q_sym(2,1); d = Q_sym(2,2);
        c = Q_sym(3,1); e = Q_sym(3,2); f = Q_sym(3,3);
        
        t2 = e*e; t4 = a*d; t7 = b*b; t9 = b*c; t12 = c*c;
        detQ = -t4*f+a*t2+t7*f-2.0*t9*e+t12*d;
        
        if abs(detQ) < 1e-10
            Qinv = pinv(Q_sym);
            return;
        end
        
        t15 = 1.0/detQ; t20 = (-b*f+c*e)*t15; t24 = (b*e-c*d)*t15; t30 = (a*e-t9)*t15;
        Qinv = zeros(3,3);
        Qinv(1,1) = (-d*f+t2)*t15;
        Qinv(1,2) = -t20; Qinv(2,1) = -t20;
        Qinv(1,3) = -t24; Qinv(3,1) = -t24;
        Qinv(2,2) = -(a*f-t12)*t15;
        Qinv(2,3) = t30;  Qinv(3,2) = t30;
        Qinv(3,3) = -(t4-t7)*t15;
    end

    function e_err = OrthogonalityError(a)
        sq_norm_a1 = a(1)^2 + a(2)^2 + a(3)^2;
        sq_norm_a2 = a(4)^2 + a(5)^2 + a(6)^2;
        sq_norm_a3 = a(7)^2 + a(8)^2 + a(9)^2;
        dot_a1a2 = a(1)*a(4) + a(2)*a(5) + a(3)*a(6);
        dot_a1a3 = a(1)*a(7) + a(2)*a(8) + a(3)*a(9);
        dot_a2a3 = a(4)*a(7) + a(5)*a(8) + a(6)*a(9);
        
        e_err = (sq_norm_a1 - 1)^2 + (sq_norm_a2 - 1)^2 + (sq_norm_a3 - 1)^2 + ...
                2*(dot_a1a2^2 + dot_a1a3^2 + dot_a2a3^2);
    end

    function d = Determinant9x1(r)
        d = (r(1)*r(5)*r(9) + r(2)*r(6)*r(7) + r(3)*r(4)*r(8)) - ...
            (r(7)*r(5)*r(3) + r(8)*r(6)*r(1) + r(9)*r(4)*r(2));
    end

    function r = NearestRotationMatrix_SVD(e)
        E = reshape(e, 3, 3)'; 
        [Usvd, ~, Vsvd] = svd(E);
        detUV = det(Usvd) * det(Vsvd);
        Rot = Usvd * diag([1, 1, detUV]) * Vsvd';
        r = reshape(Rot', 9, 1);
    end

    function r = NearestRotationMatrix_FOAM(e)
        B = e;
        detB = (B(1)*B(5)*B(9) - B(1)*B(6)*B(8) - B(2)*B(4)*B(9)) + ...
               (B(3)*B(4)*B(8) + B(2)*B(7)*B(6) - B(3)*B(7)*B(5));
        if abs(detB) < 1E-04
            r = NearestRotationMatrix_SVD(e);
            return;
        end
        
        adjB = zeros(9,1);
        adjB(1)=B(5)*B(9) - B(6)*B(8); adjB(2)=B(3)*B(8) - B(2)*B(9); adjB(3)=B(2)*B(6) - B(3)*B(5);
        adjB(4)=B(6)*B(7) - B(4)*B(9); adjB(5)=B(1)*B(9) - B(3)*B(7); adjB(6)=B(3)*B(4) - B(1)*B(6);
        adjB(7)=B(4)*B(8) - B(5)*B(7); adjB(8)=B(2)*B(7) - B(1)*B(8); adjB(9)=B(1)*B(5) - B(2)*B(4);
        
        Bsq = sum(B.^2);
        adjBsq = sum(adjB.^2);
        
        l = 0.5*(Bsq + 3.0);
        if detB < 0.0, l = -l; end
        
        lprev = 0.0;
        for k = 1:15
            if abs(l-lprev) <= 1E-12*abs(lprev), break; end
            tmp = (l*l - Bsq);
            p = (tmp*tmp - 8.0*l*detB - 4.0*adjBsq);
            pp = 8.0*(0.5*tmp*l - detB);
            lprev = l;
            l = l - p/pp;
        end
        
        BBt = zeros(9,1);
        BBt(1)=B(1)^2+B(2)^2+B(3)^2; BBt(2)=B(1)*B(4)+B(2)*B(5)+B(3)*B(6); BBt(3)=B(1)*B(7)+B(2)*B(8)+B(3)*B(9);
        BBt(4)=BBt(2);               BBt(5)=B(4)^2+B(5)^2+B(6)^2;             BBt(6)=B(4)*B(7)+B(5)*B(8)+B(6)*B(9);
        BBt(7)=BBt(3);               BBt(8)=BBt(6);                           BBt(9)=B(7)^2+B(8)^2+B(9)^2;
        
        tmp = zeros(9,1);
        tmp(1)=BBt(1)*B(1)+BBt(2)*B(4)+BBt(3)*B(7); tmp(2)=BBt(1)*B(2)+BBt(2)*B(5)+BBt(3)*B(8); tmp(3)=BBt(1)*B(3)+BBt(2)*B(6)+BBt(3)*B(9);
        tmp(4)=BBt(4)*B(1)+BBt(5)*B(4)+BBt(6)*B(7); tmp(5)=BBt(4)*B(2)+BBt(5)*B(5)+BBt(6)*B(8); tmp(6)=BBt(4)*B(3)+BBt(5)*B(6)+BBt(6)*B(9);
        tmp(7)=BBt(7)*B(1)+BBt(8)*B(4)+BBt(9)*B(7); tmp(8)=BBt(7)*B(2)+BBt(8)*B(5)+BBt(9)*B(8); tmp(9)=BBt(7)*B(3)+BBt(8)*B(6)+BBt(9)*B(9);
        
        a = l*l + Bsq;
        denom = 1.0 / (l*(l*l-Bsq) - 2.0*detB);
        
        r = zeros(9,1);
        r(1)=(a*B(1) + 2.0*(l*adjB(1) - tmp(1)))*denom; r(2)=(a*B(2) + 2.0*(l*adjB(4) - tmp(2)))*denom; r(3)=(a*B(3) + 2.0*(l*adjB(7) - tmp(3)))*denom;
        r(4)=(a*B(4) + 2.0*(l*adjB(2) - tmp(4)))*denom; r(5)=(a*B(5) + 2.0*(l*adjB(5) - tmp(5)))*denom; r(6)=(a*B(6) + 2.0*(l*adjB(8) - tmp(6)))*denom;
        r(7)=(a*B(7) + 2.0*(l*adjB(3) - tmp(7)))*denom; r(8)=(a*B(8) + 2.0*(l*adjB(6) - tmp(8)))*denom; r(9)=(a*B(9) + 2.0*(l*adjB(9) - tmp(9)))*denom;
    end
end
