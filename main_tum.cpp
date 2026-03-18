#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm> 

#include <Eigen/Dense>

#include "bapnp.h"
#include "cpnp.h"
#include "epnp.h"
#include "sqpnp.h"

using namespace std;
using namespace Eigen;

const double THRESH_R_DEG = 3.0;
const double THRESH_T_M   = 0.1;
const int BENCHMARK_ITERATIONS = 1000; 

double calc_rot_err(const Matrix3d& R1, const Matrix3d& R2) {
    double tr = (R1 * R2.transpose()).trace();
    double val = (tr - 1.0) / 2.0;
    if (val > 1.0) val = 1.0; else if (val < -1.0) val = -1.0;
    return acos(val) * 180.0 / M_PI;
}

double calc_trans_err(const Vector3d& t1, const Vector3d& t2) {
    return (t1 - t2).norm();
}

struct MethodStats {
    string name;
    double total_time = 0;
    int success_count = 0;
    double sum_r_err = 0; 
    double sum_t_err = 0; 
};

template <typename Func>
double measure_median_time(Func func) {
    std::vector<double> times;
    times.reserve(BENCHMARK_ITERATIONS);
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        auto t1 = chrono::high_resolution_clock::now();
        func(); 
        auto t2 = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(t2 - t1).count());
    }
    std::sort(times.begin(), times.end());
    if (BENCHMARK_ITERATIONS % 2 == 0) {
        return (times[BENCHMARK_ITERATIONS / 2 - 1] + times[BENCHMARK_ITERATIONS / 2]) / 2.0;
    } else {
        return times[BENCHMARK_ITERATIONS / 2];
    }
}

int main() {
    ifstream inFile("tum_data_export.txt");
    if (!inFile.is_open()) {
        cerr << "Error: Cannot open tum_data_export.txt." << endl;
        return -1;
    }

    ofstream outFile("tum_results_final.txt");
    outFile << "Frame N BAPnP_Time EPnP_Time SQPnP_Time CPnP_Time "
            << "BAPnP_R EPnP_R SQPnP_R CPnP_R "
            << "BAPnP_t EPnP_t SQPnP_t CPnP_t" << endl;

    MethodStats s_bapnp   = {"BAPnP"};
    MethodStats s_epnp    = {"EPnP(Raw)"};
    MethodStats s_sqpnp   = {"SQPnP"};
    MethodStats s_cpnp    = {"CPnP"};

    string token;
    int total_frames = 0;
    long long total_points_all = 0;

    cout << "Starting Ultimate TUM Benchmark (Median Filtering: " << BENCHMARK_ITERATIONS << " runs)..." << endl;

    while (inFile >> token) {
        if (token != "FRAME") continue;

        int frame_id, n_points;
        inFile >> frame_id >> n_points;
        total_points_all += n_points;

        double fx, fy, cx, cy;
        inFile >> fx >> fy >> cx >> cy;

        Matrix3d R_gt; Vector3d t_gt;
        for(int r=0; r<3; ++r) inFile >> R_gt(r,0) >> R_gt(r,1) >> R_gt(r,2) >> t_gt(r);

        MatrixXd P_world(3, n_points);     
        MatrixXd pts_2d_norm(3, n_points); 

        vector<Vector3d> cpnp_P3D; cpnp_P3D.reserve(n_points);
        vector<Vector2d> cpnp_P2D_norm; cpnp_P2D_norm.reserve(n_points); 
        vector<double> epnp_u(n_points), epnp_v(n_points);

        for (int i = 0; i < n_points; ++i) {
            double X, Y, Z, u, v;
            inFile >> X >> Y >> Z >> u >> v;

            P_world(0, i) = X; P_world(1, i) = Y; P_world(2, i) = Z;
            double u_norm = (u - cx) / fx;
            double v_norm = (v - cy) / fy;
            
            pts_2d_norm(0, i) = u_norm;
            pts_2d_norm(1, i) = v_norm;
            pts_2d_norm(2, i) = 1.0;

            cpnp_P3D.emplace_back(X, Y, Z);
            cpnp_P2D_norm.emplace_back(u_norm, v_norm);
            epnp_u[i] = u;
            epnp_v[i] = v;
        }

        // ======================= 1. BAPnP =======================
        Matrix3d R_est_bapnp; Vector3d t_est_bapnp;
        double time_bapnp = measure_median_time([&]() {
            BAPnP::solve(pts_2d_norm, P_world, R_est_bapnp, t_est_bapnp);
        });

        // ======================= 2. EPnP =======================
        epnp PnP;
        PnP.set_internal_parameters(cx, cy, fx, fy);
        PnP.set_maximum_number_of_correspondences(n_points);
        Matrix3d R_epnp_e; Vector3d t_epnp_e;
        double time_epnp = measure_median_time([&]() {
            PnP.reset_correspondences();
            for(int i=0; i<n_points; i++) {
                PnP.add_correspondence(P_world(0,i), P_world(1,i), P_world(2,i), epnp_u[i], epnp_v[i]);
            }
            double R_est[3][3], t_est[3];
            PnP.compute_pose(R_est, t_est);
            for(int r=0; r<3; r++) {
                for(int c=0; c<3; c++) R_epnp_e(r,c) = R_est[r][c];
                t_epnp_e(r) = t_est[r];
            }
        });

        // ======================= 3. CPnP =======================
        Matrix3d R_cpnp_e = Matrix3d::Identity();
        Vector3d t_cpnp_e = Vector3d::Zero();
        std::vector<double> cpnp_params = {1.0, 1.0, 0.0, 0.0}; 
        Vector4d q_out, q_gn; Vector3d t_out, t_gn;

        double time_cpnp = measure_median_time([&]() {
             pnpsolver::CPnP(cpnp_P2D_norm, cpnp_P3D, cpnp_params, q_out, t_out, q_gn, t_gn);
        });

        Quaterniond q_c(q_gn(0), q_gn(1), q_gn(2), q_gn(3));
        q_c.normalize();
        R_cpnp_e = q_c.toRotationMatrix();
        t_cpnp_e = t_gn;

        // ======================= 4. SQPnP  =======================
        Matrix3d R_sqpnp_e = Matrix3d::Identity();
        Vector3d t_sqpnp_e = Vector3d::Zero();
        double time_sqpnp = measure_median_time([&]() {
            sqpnp::PnPSolver solver(cpnp_P3D, cpnp_P2D_norm);
            if (solver.IsValid() && solver.Solve() && solver.NumberOfSolutions() > 0) {
                const auto* sol = solver.SolutionPtr(0);
                R_sqpnp_e = Map<const Matrix<double, 3, 3, RowMajor>>(sol->r_hat.data());
                t_sqpnp_e = sol->t;
            }
        });

        
        auto check = [&](MethodStats& s, double t, const Matrix3d& R, const Vector3d& t_vec) {
            double r_err = calc_rot_err(R_gt, R);
            double t_err = calc_trans_err(t_gt, t_vec);
            s.total_time += t; 
            if (r_err < THRESH_R_DEG && t_err < THRESH_T_M) {
                s.success_count++;
                s.sum_r_err += r_err;
                s.sum_t_err += t_err;
            }
            return make_pair(r_err, t_err);
        };

        auto e_b  = check(s_bapnp, time_bapnp, R_est_bapnp, t_est_bapnp);
        auto e_ep = check(s_epnp,  time_epnp,  R_epnp_e,    t_epnp_e);
        auto e_c  = check(s_cpnp,  time_cpnp,  R_cpnp_e,    t_cpnp_e);
        auto e_sq = check(s_sqpnp, time_sqpnp, R_sqpnp_e,   t_sqpnp_e);

        outFile << frame_id << " " << n_points << " "
                << time_bapnp << " " << time_epnp << " " << time_sqpnp << " " << time_cpnp << " "
                << e_b.first << " " << e_ep.first << " " << e_sq.first << " " << e_c.first << " "
                << e_b.second << " " << e_ep.second << " " << e_sq.second << " " << e_c.second << endl;

        total_frames++;
        if (total_frames % 50 == 0) cout << "Processed " << total_frames << " frames..." << endl;
    }

   
    cout << "\n==================================================================================" << endl;
    cout << "  TUM BENCHMARK FINAL RESULTS (" << total_frames << " frames, " << BENCHMARK_ITERATIONS << " runs/frame)" << endl;
    
    double avg_points = (total_frames > 0) ? (double)total_points_all / total_frames : 0.0;
    cout << "  Avg Points per Frame: " << fixed << setprecision(1) << avg_points << endl;
    
    cout << "  Thresh: " << THRESH_R_DEG << " deg, " << THRESH_T_M << " m" << endl;
    cout << "==================================================================================" << endl;

    auto print_stat = [&](MethodStats s) {
        double succ_rate = (total_frames > 0) ? 100.0 * s.success_count / total_frames : 0;
        double avg_time = (total_frames > 0) ? s.total_time / total_frames : 0; 
        double avg_r = (s.success_count > 0) ? s.sum_r_err / s.success_count : -1;
        double avg_t = (s.success_count > 0) ? s.sum_t_err / s.success_count : -1;

        cout << left << setw(10) << s.name
             << " | Time: " << fixed << setprecision(3) << avg_time << " ms"
             << " | Succ: " << setprecision(1) << succ_rate << "%"
             << " | Err(R/t): " << setprecision(3) << avg_r << " deg / " << avg_t << " m"
             << endl;
    };

    print_stat(s_bapnp);
    print_stat(s_epnp);     
    print_stat(s_cpnp);
    print_stat(s_sqpnp);
    cout << "==================================================================================" << endl;

    inFile.close();
    outFile.close();
    return 0;
}
