#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <random>

#include <Eigen/Dense>

#include "bapnp.h"
#include "cpnp.h"
#include "epnp.h"      
#include "sqpnp.h"     

using namespace std;
using namespace Eigen;

// --- 核心计时函数 ---
template <typename Func>
double measure_median_time(Func func) {
    int iterations = 100; 
    std::vector<double> times(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto t1 = chrono::high_resolution_clock::now();
        func(); 
        auto t2 = chrono::high_resolution_clock::now();
        times[i] = chrono::duration<double, milli>(t2 - t1).count();
    }
    std::sort(times.begin(), times.end());
    return times[iterations / 2];
}

int main() {
    ofstream outFile("runtime_benchmark_result.txt");
    if (!outFile.is_open()) return -1;
    outFile << "N BAPnP EPnP(Raw+GN) CPnP SQPnP(Raw)" << endl; 

    vector<int> N_list = {6, 10, 20, 50, 100, 200, 400, 500, 1000};
    srand(2025); 

    const double f_sim = 800.0;
    const double cx_sim = 320.0;
    const double cy_sim = 240.0;

    for (int n : N_list) {
        double t_bapnp = 0, t_epnp = 0, t_cpnp = 0, t_sqpnp = 0;
        int trials = 1000; 

        std::default_random_engine generator(2025 + n);
        std::normal_distribution<double> noise_dist(0.0, 2.0);

        for (int k = 0; k < trials; ++k) {
            MatrixXd P_world = MatrixXd::Random(3, n) * 10.0; 
            P_world.row(2).array() += 20.0; 
            
            Matrix3d R_gt = Quaterniond::UnitRandom().toRotationMatrix();
            Vector3d t_gt = Vector3d::Random();
            MatrixXd P_cam = (R_gt * P_world).colwise() + t_gt;
            
            MatrixXd y_norm(3, n);
            vector<Vector3d> cpnp_P3D(n);
            vector<Vector2d> cpnp_P2D_norm(n); 
            vector<double> epnp_u(n), epnp_v(n);

            for(int i=0; i<n; ++i) {
                double X = P_cam(0, i), Y = P_cam(1, i), Z = P_cam(2, i);
                double u_noisy = f_sim * (X / Z) + cx_sim + noise_dist(generator);
                double v_noisy = f_sim * (Y / Z) + cy_sim + noise_dist(generator);

                double u_n = (u_noisy - cx_sim) / f_sim;
                double v_n = (v_noisy - cy_sim) / f_sim;

                y_norm(0, i) = u_n; y_norm(1, i) = v_n; y_norm(2, i) = 1.0;

                cpnp_P3D[i] = P_world.col(i);
                cpnp_P2D_norm[i] = Vector2d(u_n, v_n);
                
                epnp_u[i] = u_noisy;
                epnp_v[i] = v_noisy;
            }

            // 1. BAPnP
            Matrix3d R_est_bapnp; Vector3d t_est_bapnp;
            t_bapnp += measure_median_time([&]() {
                BAPnP::solve(y_norm, P_world, R_est_bapnp, t_est_bapnp);
            });

            // 2. EPnP (Raw C++ 包含 GN)
            epnp PnP;
            PnP.set_internal_parameters(cx_sim, cy_sim, f_sim, f_sim);
            PnP.set_maximum_number_of_correspondences(n);
            t_epnp += measure_median_time([&]() {
                PnP.reset_correspondences();
                for(int i=0; i<n; i++) PnP.add_correspondence(P_world(0,i), P_world(1,i), P_world(2,i), epnp_u[i], epnp_v[i]);
                double R_est[3][3], t_est[3];
                PnP.compute_pose(R_est, t_est);
            });

            // 3. CPnP
            std::vector<double> params = {1.0, 1.0, 0.0, 0.0};
            Eigen::Vector4d q_out, q_gn; Eigen::Vector3d t_out, t_gn;
            t_cpnp += measure_median_time([&]() {
                pnpsolver::CPnP(cpnp_P2D_norm, cpnp_P3D, params, q_out, t_out, q_gn, t_gn);
            });

            // 4. SQPnP (Raw C++)
            t_sqpnp += measure_median_time([&]() {
                sqpnp::PnPSolver solver(cpnp_P3D, cpnp_P2D_norm);
                if (solver.IsValid()) solver.Solve();
            });
        }

        double avg_bapnp = t_bapnp / trials;
        double avg_epnp  = t_epnp / trials;
        double avg_cpnp  = t_cpnp / trials;
        double avg_sqpnp = t_sqpnp / trials;

        cout << left << "N=" << setw(5) << n 
             << " | BAPnP: " << fixed << setprecision(4) << avg_bapnp 
             << " | EPnP(Raw): " << avg_epnp 
             << " | CPnP: " << avg_cpnp 
             << " | SQPnP(Raw): " << avg_sqpnp << endl;

        outFile << n << " " << avg_bapnp << " " << avg_epnp << " " << avg_cpnp << " " << avg_sqpnp << endl;
    }

    outFile.close();
    return 0;
}
