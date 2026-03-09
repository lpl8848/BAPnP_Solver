#include "bapnp.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Cholesky> 

using namespace Eigen;
using namespace std;

namespace {
    const double SQRT3 = 1.732050807568877;
}

void BAPnP::solve(const MatrixXd& y_norm, const MatrixXd& P_world, Matrix3d& R, Vector3d& t) {
    Matrix3d R_init;
    Vector3d t_init;
    
    // Step 1: Linear Initialization
    linear_solver(y_norm, P_world, R_init, t_init);

    // Step 2: GN Refinement
    refine_gn(R_init, t_init, y_norm, P_world, R, t);
}

void BAPnP::linear_solver(const MatrixXd& y_norm, const MatrixXd& P_world, Matrix3d& R_out, Vector3d& t_out) {
    const int N = P_world.cols();
    if (N < 4) {
        R_out.setIdentity();
        t_out.setZero();
        return;
    }

    // --- 1. 3D 数据归一化 ---
    Vector3d cent_3d = P_world.rowwise().sum() / (double)N;
    
    Matrix<double, 3, Dynamic> P_n(3, N);
    double sq_sum = 0.0;
    
    for(int i=0; i<N; ++i) {
        P_n.col(i) = P_world.col(i) - cent_3d;
        sq_sum += P_n.col(i).squaredNorm();
    }
    
    double rms_dist = sqrt(sq_sum / N);
    if (rms_dist < 1e-6) rms_dist = 1.0;
    
    double scale_3d = SQRT3 / rms_dist;
    P_n *= scale_3d; 

    // --- 2. 极速基底选择 ---
    int base_idx[4] = {0, 0, 0, 0};

    double max_dist = -1.0;
    for(int i=0; i<N; ++i) {
        double d = P_n.col(i).squaredNorm();
        if(d > max_dist) { max_dist = d; base_idx[0] = i; }
    }
    Vector3d p1 = P_n.col(base_idx[0]);

    max_dist = -1.0;
    for(int i=0; i<N; ++i) {
        double d = (P_n.col(i) - p1).squaredNorm();
        if(d > max_dist) { max_dist = d; base_idx[1] = i; }
    }
    Vector3d p2 = P_n.col(base_idx[1]);

    max_dist = -1.0;
    Vector3d v12 = p2 - p1;
    for(int i=0; i<N; ++i) {
        double d = v12.cross(P_n.col(i) - p1).squaredNorm();
        if(d > max_dist) { max_dist = d; base_idx[2] = i; }
    }
    Vector3d p3 = P_n.col(base_idx[2]);

    max_dist = -1.0;
    Vector3d v13 = p3 - p1;
    Vector3d n_plane = v12.cross(v13); 
    for(int i=0; i<N; ++i) {
        double val = n_plane.dot(P_n.col(i) - p1);
        double d = val * val;
        if(d > max_dist) { max_dist = d; base_idx[3] = i; }
    }

    vector<int> perm;
    perm.reserve(N);
    bool is_base[N]; 
    std::fill(is_base, is_base + N, false);
    
    for(int i=0; i<4; ++i) {
        perm.push_back(base_idx[i]);
        is_base[base_idx[i]] = true;
    }
    for(int i=0; i<N; ++i) {
        if(!is_base[i]) perm.push_back(i);
    }

    Matrix3d R0; 
    {
        Vector3d P1_vec = P_n.col(perm[0]);
        Vector3d P2_vec = P_n.col(perm[1]);
        Vector3d P3_vec = P_n.col(perm[2]);
        Vector3d C0 = (P1_vec + P2_vec + P3_vec) / 3.0;

        Vector3d r1 = (P1_vec - C0).normalized();
        Vector3d r3 = r1.cross(P2_vec - C0).normalized();
        Vector3d r2 = r3.cross(r1);
        R0.row(0) = r1; R0.row(1) = r2; R0.row(2) = r3;
    }

    // --- 3. 线性求解深度 ---
    Matrix<double, 3, 4> P_base_perm;
    for(int i=0; i<4; ++i) P_base_perm.col(i) = P_n.col(perm[i]);
    
    Vector3d C0_perm = P_base_perm.leftCols(3).rowwise().mean(); 
    Matrix<double, 3, 4> W_prime = R0 * (P_base_perm.colwise() - C0_perm);

    Matrix3d B;
    B.col(0) = W_prime.col(0) - W_prime.col(3);
    B.col(1) = W_prime.col(1) - W_prime.col(3);
    B.col(2) = W_prime.col(2) - W_prime.col(3);

    Matrix3d B_inv = B.inverse(); 
    Vector3d W_base4 = W_prime.col(3); 

    // --- 4. 优化：构建与求解 L^T * L (4x4) ---
    Matrix4d M = Matrix4d::Zero();
    
    const int n_others = N - 4;
    double u_base[4], v_base[4];
    for(int i=0; i<4; ++i) {
        u_base[i] = y_norm(0, perm[i]);
        v_base[i] = y_norm(1, perm[i]);
    }

    Vector3d W_curr;
    Vector3d coeffs_abc; 
    double dx[4], dy[4], dxy[4];

    for(int k=0; k < n_others; ++k) {
        int original_idx = perm[k + 4];
        
        W_curr = R0 * (P_n.col(original_idx) - C0_perm);
        coeffs_abc = B_inv * (W_curr - W_base4);
        
        double weights[4] = {
            coeffs_abc(0), coeffs_abc(1), coeffs_abc(2), 
            1.0 - coeffs_abc(0) - coeffs_abc(1) - coeffs_abc(2)
        };

        double uk = y_norm(0, original_idx);
        double vk = y_norm(1, original_idx);

        // 提取公共偏差量，省去 L_block 矩阵的构建
        for (int j = 0; j < 4; ++j) {
            dx[j]  = weights[j] * (v_base[j] - vk);
            dy[j]  = weights[j] * (uk - u_base[j]);
            dxy[j] = weights[j] * (uk * v_base[j] - vk * u_base[j]);
        }

        // 仅展开上三角，省去了一半以上的乘法计算
        for (int r = 0; r < 4; ++r) {
            for (int c = r; c < 4; ++c) {
                M(r, c) += dx[r]*dx[c] + dy[r]*dy[c] + dxy[r]*dxy[c];
            }
        }
    }
    // 补齐下三角
    M.triangularView<StrictlyLower>() = M.transpose();

    SelfAdjointEigenSolver<Matrix4d> eig(M);
    Vector4d rho = eig.eigenvectors().col(0); 
    
    if (rho.sum() < 0) rho = -rho; 
    if (std::abs(rho(0)) < 1e-8) rho(0) = 1e-8;

    // --- 恢复深度 ---
    Matrix<double, 3, Dynamic> P_cam_metric(3, N);
    VectorXd Z_all(N);

    for(int i=0; i<4; ++i) Z_all(perm[i]) = rho(i);

    for(int i=4; i<N; ++i) {
        int idx = perm[i];
        W_curr = R0 * (P_n.col(idx) - C0_perm);
        coeffs_abc = B_inv * (W_curr - W_base4);
        Z_all(idx) = coeffs_abc(0)*rho(0) + coeffs_abc(1)*rho(1) + 
                     coeffs_abc(2)*rho(2) + (1.0 - coeffs_abc.sum())*rho(3);
    }
    
    for(int i=0; i<N; ++i) {
         double z = Z_all(i);
         P_cam_metric(0, i) = y_norm(0, i) * z;
         P_cam_metric(1, i) = y_norm(1, i) * z;
         P_cam_metric(2, i) = z;
    }

    // --- 5. 恢复尺度并对齐 ---
    Vector3d cent_cam = P_cam_metric.rowwise().sum() / (double)N;
    double s_sq_sum = 0;
    for(int i=0; i<N; ++i) s_sq_sum += (P_cam_metric.col(i) - cent_cam).squaredNorm();
    double s_cam = sqrt(s_sq_sum / N);
    
    if(s_cam < 1e-6) s_cam = 1.0;
    double true_scale = SQRT3 / s_cam;
    
    P_cam_metric *= true_scale;

    Matrix3d R_est; Vector3d t_est_norm;
    compute_procrustes(P_n, P_cam_metric, R_est, t_est_norm);

    // --- 6. 1-Step Refinement ---
    for(int i=0; i<N; ++i) {
        Vector3d P_c = R_est * P_n.col(i) + t_est_norm;
        double z_ref = P_c(2);
        if(z_ref < 1e-6) z_ref = 1e-6; 

        P_cam_metric(0, i) = y_norm(0, i) * z_ref;
        P_cam_metric(1, i) = y_norm(1, i) * z_ref;
        P_cam_metric(2, i) = z_ref;
    }
    
    compute_procrustes(P_n, P_cam_metric, R_out, t_out);
    
    t_out = t_out / scale_3d - R_out * cent_3d;
}

void BAPnP::refine_gn(const Matrix3d& R_init, const Vector3d& t_init,
                      const MatrixXd& y_norm, const MatrixXd& P_world,
                      Matrix3d& R, Vector3d& t) 
{
    const int MAX_ITER = 1; 
    const double MIN_DELTA = 1e-6;
    const int N = P_world.cols();

    R = R_init;
    t = t_init;

    Vector3d P_c;
    Matrix<double, 6, 6> H;
    Matrix<double, 6, 1> b;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        H.setZero();
        b.setZero();
        
        double total_err = 0;

        for(int i=0; i<N; ++i) {
            P_c.noalias() = R * P_world.col(i) + t;

            double X = P_c(0);
            double Y = P_c(1);
            double Z = P_c(2);

            if (Z < 1e-6) Z = 1e-6;
            double inv_Z = 1.0 / Z;
            double inv_Z2 = inv_Z * inv_Z;

            double u = X * inv_Z;
            double v = Y * inv_Z;
            double u2 = u * u;
            double v2 = v * v;
            double uv = u * v;

            double u_obs = y_norm(0, i);
            double v_obs = y_norm(1, i);
            if (y_norm.rows() == 3) {
                double w = y_norm(2, i);
                u_obs /= w; v_obs /= w;
            }

            double r0 = u - u_obs;
            double r1 = v - v_obs;
            total_err += r0 * r0 + r1 * r1;

            // 1. 解析累加 b = -J^T * r
            b(0) -= inv_Z * r0;
            b(1) -= inv_Z * r1;
            b(2) += (u * r0 + v * r1) * inv_Z;
            b(3) += uv * r0 + (1.0 + v2) * r1;
            b(4) -= (1.0 + u2) * r0 + uv * r1;
            b(5) += v * r0 - u * r1;

            // 2. 解析累加 H = J^T * J (仅上三角)
            H(0,0) += inv_Z2;
            H(0,2) -= u * inv_Z2;
            H(0,3) -= uv * inv_Z;
            H(0,4) += (1.0 + u2) * inv_Z;
            H(0,5) -= v * inv_Z;

            H(1,1) += inv_Z2;
            H(1,2) -= v * inv_Z2;
            H(1,3) -= (1.0 + v2) * inv_Z;
            H(1,4) += uv * inv_Z;
            H(1,5) += u * inv_Z;

            H(2,2) += (u2 + v2) * inv_Z2;
            H(2,3) += v * (u2 + v2 + 1.0) * inv_Z;
            H(2,4) -= u * (u2 + v2 + 1.0) * inv_Z;
            // H(2,5) 严格等于 0，跳过

            H(3,3) += u2*v2 + (1.0+v2)*(1.0+v2);
            H(3,4) -= uv * (2.0 + u2 + v2);
            H(3,5) -= u;

            H(4,4) += (1.0+u2)*(1.0+u2) + u2*v2;
            H(4,5) -= v;

            H(5,5) += u2 + v2; 
        }

        if (total_err / N < 1e-9) break;

        // 使用 LDLT 求解对称正定方程 
        Vector6d delta = H.selfadjointView<Upper>().ldlt().solve(b);

        if (delta.norm() < MIN_DELTA) break;

        Vector3d d_t = delta.head(3);
        Vector3d d_w = delta.tail(3);

        double angle = d_w.norm();
        Matrix3d dR;
        if (angle < 1e-10) {
            dR.setIdentity();
        } else {
            dR = AngleAxisd(angle, d_w / angle).toRotationMatrix();
        }

        R = dR * R;
        t = dR * t + d_t;
    }
}

void BAPnP::compute_procrustes(const MatrixXd& P_src, const MatrixXd& P_dst, Matrix3d& R, Vector3d& t) {
    int N = P_src.cols();
    
    // c_src 理论上必须为 0，因为 P_src (即 P_n) 已经去均值过了。省去 O(N) 的多余遍历。
    Vector3d c_dst = P_dst.rowwise().sum() / N;

    Matrix3d H = Matrix3d::Zero();
    for(int i=0; i<N; ++i) {
        // 直接省掉原先这里的 6 次减法操作，数学上完全等价。
        H.noalias() += P_src.col(i) * P_dst.col(i).transpose(); 
    }

    JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);
    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();

    R = V * U.transpose();
    
    if (R.determinant() < 0) {
        Matrix3d S = Matrix3d::Identity();
        S(2, 2) = -1;
        R = V * S * U.transpose();
    }
    
    // 因为 c_src == 0，所以 t = c_dst - R * c_src 直接简化为：
    t = c_dst; 
}
