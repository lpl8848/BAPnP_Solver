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
    P_n *= scale_3d; // In-place 缩放


    int base_idx[4] = {0, 0, 0, 0};

    // P1: 
    double max_dist = -1.0;
    for(int i=0; i<N; ++i) {
        double d = P_n.col(i).squaredNorm();
        if(d > max_dist) { max_dist = d; base_idx[0] = i; }
    }
    Vector3d p1 = P_n.col(base_idx[0]);

    // P2: 
    max_dist = -1.0;
    for(int i=0; i<N; ++i) {
        double d = (P_n.col(i) - p1).squaredNorm();
        if(d > max_dist) { max_dist = d; base_idx[1] = i; }
    }
    Vector3d p2 = P_n.col(base_idx[1]);

    // P3: 
    max_dist = -1.0;
    Vector3d v12 = p2 - p1;
    for(int i=0; i<N; ++i) {
        double d = v12.cross(P_n.col(i) - p1).squaredNorm();
        if(d > max_dist) { max_dist = d; base_idx[2] = i; }
    }
    Vector3d p3 = P_n.col(base_idx[2]);

    // P4: 
    max_dist = -1.0;
    Vector3d v13 = p3 - p1;
    Vector3d n_plane = v12.cross(v13); // 法向量
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

    // --- 3. ---

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

    // --- 4. L^T * L (4x4) ---
    Matrix4d M = Matrix4d::Zero();
    
    const int n_others = N - 4;

    double u_base[4], v_base[4];
    for(int i=0; i<4; ++i) {
        u_base[i] = y_norm(0, perm[i]);
        v_base[i] = y_norm(1, perm[i]);
    }


    Vector3d W_curr;
    Vector3d coeffs_abc; // alpha, beta, gamma
    Matrix<double, 3, 4> L_block; 

    for(int k=0; k < n_others; ++k) {
        int original_idx = perm[k + 4];
        

        W_curr = R0 * (P_n.col(original_idx) - C0_perm);
        coeffs_abc = B_inv * (W_curr - W_base4);
        
        double alpha = coeffs_abc(0);
        double beta  = coeffs_abc(1);
        double gamma = coeffs_abc(2);
        double delta = 1.0 - alpha - beta - gamma;


        double uk = y_norm(0, original_idx);
        double vk = y_norm(1, original_idx);


        
        // j=0 (alpha)
        L_block(0,0) = alpha * (v_base[0] - vk);
        L_block(1,0) = alpha * (uk - u_base[0]);
        L_block(2,0) = alpha * (uk * v_base[0] - vk * u_base[0]);

        // j=1 (beta)
        L_block(0,1) = beta * (v_base[1] - vk);
        L_block(1,1) = beta * (uk - u_base[1]);
        L_block(2,1) = beta * (uk * v_base[1] - vk * u_base[1]);

        // j=2 (gamma)
        L_block(0,2) = gamma * (v_base[2] - vk);
        L_block(1,2) = gamma * (uk - u_base[2]);
        L_block(2,2) = gamma * (uk * v_base[2] - vk * u_base[2]);

        // j=3 (delta)
        L_block(0,3) = delta * (v_base[3] - vk);
        L_block(1,3) = delta * (uk - u_base[3]);
        L_block(2,3) = delta * (uk * v_base[3] - vk * u_base[3]);


        M.noalias() += L_block.transpose() * L_block;
    }


    SelfAdjointEigenSolver<Matrix4d> eig(M);
    Vector4d rho = eig.eigenvectors().col(0); 
    
    if (rho.sum() < 0) rho = -rho; 

    if (std::abs(rho(0)) < 1e-8) rho(0) = 1e-8;


    Matrix<double, 3, Dynamic> P_cam_metric(3, N);
    

    VectorXd Z_all(N);

    for(int i=0; i<4; ++i) Z_all(perm[i]) = rho(i);


    for(int i=0; i<N; ++i) {
        int idx = perm[i];
        double z = 0;
        if (i < 4) {
            z = rho(i);
        } else {
          
            W_curr = R0 * (P_n.col(idx) - C0_perm);
            coeffs_abc = B_inv * (W_curr - W_base4);
            z = coeffs_abc(0)*rho(0) + coeffs_abc(1)*rho(1) + 
                coeffs_abc(2)*rho(2) + (1.0 - coeffs_abc.sum())*rho(3);
            Z_all(idx) = z;
        }
        
        double u = y_norm(0, idx);
        double v = y_norm(1, idx);
        
        P_cam_metric(0, i) = u * z; 
        P_cam_metric(1, i) = v * z; 
        P_cam_metric(2, i) = z;     

    }
    
    for(int i=0; i<N; ++i) {
         double z = Z_all(i);
         P_cam_metric(0, i) = y_norm(0, i) * z;
         P_cam_metric(1, i) = y_norm(1, i) * z;
         P_cam_metric(2, i) = z;
    }


    Vector3d cent_cam = P_cam_metric.rowwise().sum() / (double)N;
    double s_sq_sum = 0;
    for(int i=0; i<N; ++i) s_sq_sum += (P_cam_metric.col(i) - cent_cam).squaredNorm();
    double s_cam = sqrt(s_sq_sum / N);
    
    if(s_cam < 1e-6) s_cam = 1.0;
    double true_scale = SQRT3 / s_cam;
    
    P_cam_metric *= true_scale;

    Matrix3d R_est; Vector3d t_est_norm;

    compute_procrustes(P_n, P_cam_metric, R_est, t_est_norm);

    // --- 6. 1-Step Refinement  ---

    for(int i=0; i<N; ++i) {
        Vector3d P_w = P_n.col(i);
        Vector3d P_c = R_est * P_w + t_est_norm;
        double z_ref = P_c(2);
        if(z_ref < 1e-6) z_ref = 1e-6; 

        P_cam_metric(0, i) = y_norm(0, i) * z_ref;
        P_cam_metric(1, i) = y_norm(1, i) * z_ref;
        P_cam_metric(2, i) = z_ref;
    }
    
    compute_procrustes(P_n, P_cam_metric, R_out, t_out);
    

    // t_final = t_out / scale - R * C_world
    t_out = t_out / scale_3d - R_out * cent_3d;
}

void BAPnP::refine_gn(const Matrix3d& R_init, const Vector3d& t_init,
                      const MatrixXd& y_norm, const MatrixXd& P_world,
                      Matrix3d& R, Vector3d& t) 
{
    const int MAX_ITER = 5; 
    const double MIN_DELTA = 1e-6;
    const int N = P_world.cols();

    R = R_init;
    t = t_init;


    Matrix<double, 2, 6> J_i;
    Vector2d res_i;
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

            double u_proj = X * inv_Z;
            double v_proj = Y * inv_Z;


            double u_obs = y_norm(0, i);
            double v_obs = y_norm(1, i);
            if (y_norm.rows() == 3) {
                 double w = y_norm(2, i);
                 u_obs /= w; v_obs /= w;
            }

            res_i << u_proj - u_obs, v_proj - v_obs;
            total_err += res_i.squaredNorm();


            // du/dt
            J_i(0, 0) = inv_Z;
            J_i(0, 1) = 0;
            J_i(0, 2) = -X * inv_Z2;
            
            // du/dw
            J_i(0, 3) = -X * Y * inv_Z2;     // -u*v
            J_i(0, 4) = 1.0 + X * X * inv_Z2; // 1+u^2
            J_i(0, 5) = -Y * inv_Z;          // -v

            // dv/dt
            J_i(1, 0) = 0;
            J_i(1, 1) = inv_Z;
            J_i(1, 2) = -Y * inv_Z2;

            // dv/dw
            J_i(1, 3) = -1.0 - Y * Y * inv_Z2; // -1-v^2
            J_i(1, 4) = X * Y * inv_Z2;        // u*v
            J_i(1, 5) = X * inv_Z;             // u

            // 累加 H 和 b
            // H += J^T * J
            // b -= J^T * res
            H.noalias() += J_i.transpose() * J_i;
            b.noalias() -= J_i.transpose() * res_i;
        }

        if (total_err / N < 1e-9) break;


        Vector6d delta = H.ldlt().solve(b);

        if (delta.norm() < MIN_DELTA) break;

        // Update
        Vector3d d_t = delta.head(3);
        Vector3d d_w = delta.tail(3);

        // Rodrigues 
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
    

    Vector3d c_src = P_src.rowwise().sum() / N;
    Vector3d c_dst = P_dst.rowwise().sum() / N;


    Matrix3d H = Matrix3d::Zero();
    for(int i=0; i<N; ++i) {
        H.noalias() += (P_src.col(i) - c_src) * (P_dst.col(i) - c_dst).transpose();
    }

    // SVD of 3x3 Matrix
    JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);
    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();

    R = V * U.transpose();
    
    if (R.determinant() < 0) {
        Matrix3d S = Matrix3d::Identity();
        S(2, 2) = -1;
        R = V * S * U.transpose();
    }
    
    t = c_dst - R * c_src;
}
