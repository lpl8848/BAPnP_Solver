#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>

#define _USE_MATH_DEFINES
#include <cmath>

#include <Eigen/Dense>

#include "bapnp.h"
#include "cpnp.h"
#include "epnp.h"
#include "sqpnp.h"

using namespace std;
using namespace Eigen;

const double THRESH_R_DEG = 3.0;
const double THRESH_T_M = 0.1;

//注意：由于每个算法现在都要跑完整的 RANSAC 循环，将测速次数降至 50 次以防止跑废 CPU
const int BENCHMARK_ITERATIONS = 1;

// ======================= 误差计算辅助函数 =======================
double calc_rot_err(const Matrix3d& R1, const Matrix3d& R2) {
    double tr = (R1 * R2.transpose()).trace();
    double val = (tr - 1.0) / 2.0;
    if (val > 1.0) val = 1.0; else if (val < -1.0) val = -1.0;
    return acos(val) * 180.0 / M_PI;
}

double calc_trans_err(const Vector3d& t1, const Vector3d& t2) {
    return (t1 - t2).norm();
}

double calc_reproj_error(const Vector3d& P3D, const Vector2d& P2D_norm, const Matrix3d& R, const Vector3d& t) {
    Vector3d P_cam = R * P3D + t;
    if (P_cam(2) < 1e-5) return 1e6;
    Vector2d p_proj(P_cam(0) / P_cam(2), P_cam(1) / P_cam(2));
    return (p_proj - P2D_norm).norm();
}

// ======================= 泛型 RANSAC 框架 =======================
// 接受任意求解器 (SolverFunc)，独立执行 RANSAC + 最终优化
template <typename SolverFunc>
int run_custom_ransac(const vector<Vector3d>& P3D, const vector<Vector2d>& P2D_norm,
    SolverFunc solver,
    Matrix3d& best_R, Vector3d& best_t,
    int iterations, double reproj_thresh, int seed)
{
    int n = P3D.size();
    if (n < 6) return 0; // 点数不足，直接失败

    // 固定种子以确保在 measure_median_time 循环内，执行相同的随机序列
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> uni(0, n - 1);

    int max_inliers = -1;
    vector<int> best_inliers;
    Matrix3d temp_best_R = Matrix3d::Identity();
    Vector3d temp_best_t = Vector3d::Zero();

    // 1. RANSAC 循环
    for (int i = 0; i < iterations; ++i) {
        vector<Vector3d> sub_P3D;
        vector<Vector2d> sub_P2D;
        vector<int> sample_idx;

        while (sample_idx.size() < 6) {
            int idx = uni(rng);
            if (std::find(sample_idx.begin(), sample_idx.end(), idx) == sample_idx.end()) {
                sample_idx.push_back(idx);
                sub_P3D.push_back(P3D[idx]);
                sub_P2D.push_back(P2D_norm[idx]);
            }
        }

        Matrix3d R_est; Vector3d t_est;
        // 调用传入的具体求解器
        if (!solver(sub_P3D, sub_P2D, R_est, t_est)) continue;

        vector<int> current_inliers;
        for (int j = 0; j < n; ++j) {
            if (calc_reproj_error(P3D[j], P2D_norm[j], R_est, t_est) < reproj_thresh) {
                current_inliers.push_back(j);
            }
        }

        if ((int)current_inliers.size() > max_inliers) {
            max_inliers = current_inliers.size();
            best_inliers = current_inliers;
            temp_best_R = R_est;
            temp_best_t = t_est;
        }
    }

    // 2. 使用所有内点重新进行一次最终求解 (Refinement)
    if (max_inliers >= 6) {
        vector<Vector3d> inlier_P3D;
        vector<Vector2d> inlier_P2D;
        for (int idx : best_inliers) {
            inlier_P3D.push_back(P3D[idx]);
            inlier_P2D.push_back(P2D_norm[idx]);
        }
        if (solver(inlier_P3D, inlier_P2D, best_R, best_t)) {
            return max_inliers;
        }
        else {
            best_R = temp_best_R;
            best_t = temp_best_t;
            return max_inliers;
        }
    }

    return 0; // RANSAC 失败
}

// ======================= 测时结构与函数 =======================
struct MethodStats {
    string name;
    double total_time = 0;
    int success_count = 0;
    double sum_r_err = 0;
    double sum_t_err = 0;
    long long sum_inliers = 0;
    long long sum_points = 0;
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
    }
    else {
        return times[BENCHMARK_ITERATIONS / 2];
    }
}

int main(int argc, char** argv) {
    string data_file = "D:/Data/tum_data_export.txt";
    if (argc > 1) data_file = argv[1];
    ifstream inFile(data_file);
    if (!inFile.is_open()) {
        cerr << "Error: Cannot open " << data_file << endl;
        return -1;
    }

    ofstream outFile("tum_results_ransac.txt");
    outFile << "Frame Pts BAPnP_Inl EPnP_Inl SQPnP_Inl CPnP_Inl "
        << "BAPnP_Time EPnP_Time SQPnP_Time CPnP_Time "
        << "BAPnP_R EPnP_R SQPnP_R CPnP_R "
        << "BAPnP_t EPnP_t SQPnP_t CPnP_t" << endl;

    MethodStats s_bapnp = { "BAPnP+RANSAC" };
    MethodStats s_epnp = { "EPnP+RANSAC" };
    MethodStats s_sqpnp = { "SQPnP+RANSAC" };
    MethodStats s_cpnp = { "CPnP+RANSAC" };

    string token;
    int total_frames = 0;

    cout << "Starting Ultimate RANSAC Benchmark (" << BENCHMARK_ITERATIONS << " runs/frame, 500 RANSAC iters)..." << endl;

    // --- 定义各个求解器的 Lambda 包装函数，统一接口 ---

    auto solve_bapnp = [](const vector<Vector3d>& p3, const vector<Vector2d>& p2, Matrix3d& R, Vector3d& t) -> bool {
        int sz = p3.size();
        MatrixXd P(3, sz); MatrixXd p_img(3, sz);
        for (int i = 0; i < sz; ++i) {
            P.col(i) = p3[i]; p_img.col(i) << p2[i](0), p2[i](1), 1.0;
        }
        BAPnP::solve(p_img, P, R, t);
        return true;
        };

    auto solve_epnp = [](const vector<Vector3d>& p3, const vector<Vector2d>& p2, Matrix3d& R, Vector3d& t) -> bool {
        int sz = p3.size();
        epnp PnP;
        PnP.set_internal_parameters(0.0, 0.0, 1.0, 1.0); // 传入归一化坐标，所以内参设为单位阵
        PnP.set_maximum_number_of_correspondences(sz);
        PnP.reset_correspondences();
        for (int i = 0; i < sz; i++) {
            PnP.add_correspondence(p3[i](0), p3[i](1), p3[i](2), p2[i](0), p2[i](1));
        }
        double R_est[3][3], t_est[3];
        PnP.compute_pose(R_est, t_est);
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) R(r, c) = R_est[r][c];
            t(r) = t_est[r];
        }
        return true;
        };

    auto solve_cpnp = [](const vector<Vector3d>& p3, const vector<Vector2d>& p2, Matrix3d& R, Vector3d& t) -> bool {
        std::vector<double> cpnp_params = { 1.0, 1.0, 0.0, 0.0 };
        Vector4d q_out, q_gn; Vector3d t_out, t_gn;
        pnpsolver::CPnP(p2, p3, cpnp_params, q_out, t_out, q_gn, t_gn);
        Quaterniond q_c(q_gn(0), q_gn(1), q_gn(2), q_gn(3));
        q_c.normalize();
        R = q_c.toRotationMatrix();
        t = t_gn;
        return true;
        };

    auto solve_sqpnp = [](const vector<Vector3d>& p3, const vector<Vector2d>& p2, Matrix3d& R, Vector3d& t) -> bool {
        sqpnp::PnPSolver solver(p3, p2);
        if (solver.IsValid() && solver.Solve() && solver.NumberOfSolutions() > 0) {
            const auto* sol = solver.SolutionPtr(0);
            R = Map<const Matrix<double, 3, 3, RowMajor>>(sol->r_hat.data());
            t = sol->t;
            return true;
        }
        return false;
        };

    while (inFile >> token) {
        if (token != "FRAME") continue;

        int frame_id, n_points;
        inFile >> frame_id >> n_points;

        double fx, fy, cx, cy;
        inFile >> fx >> fy >> cx >> cy;

        Matrix3d R_gt; Vector3d t_gt;
        for (int r = 0; r < 3; ++r) inFile >> R_gt(r, 0) >> R_gt(r, 1) >> R_gt(r, 2) >> t_gt(r);

        vector<Vector3d> cpnp_P3D; cpnp_P3D.reserve(n_points);
        vector<Vector2d> cpnp_P2D_norm; cpnp_P2D_norm.reserve(n_points);

        for (int i = 0; i < n_points; ++i) {
            double X, Y, Z, u, v;
            inFile >> X >> Y >> Z >> u >> v;
            cpnp_P3D.emplace_back(X, Y, Z);
            cpnp_P2D_norm.emplace_back((u - cx) / fx, (v - cy) / fy);
        }

        if (n_points < 6) {
            cout << "Frame " << frame_id << " has < 6 points. Skipping." << endl;
            continue;
        }

        // 设置 RANSAC 阈值：2 像素对应归一化平面的阈值
        double ransac_thresh = 2.0 / ((fx + fy) / 2.0);

        // ======================= 独立测试各个算法 =======================
        int inl_b = 0, inl_ep = 0, inl_sq = 0, inl_c = 0;
        Matrix3d R_b, R_ep, R_sq, R_c;
        Vector3d t_b, t_ep, t_sq, t_c;

        double t_bapnp = measure_median_time([&]() {
            inl_b = run_custom_ransac(cpnp_P3D, cpnp_P2D_norm, solve_bapnp, R_b, t_b, 500, ransac_thresh, frame_id);
            });

        double t_epnp = measure_median_time([&]() {
            inl_ep = run_custom_ransac(cpnp_P3D, cpnp_P2D_norm, solve_epnp, R_ep, t_ep, 500, ransac_thresh, frame_id);
            });

        double t_sqpnp = measure_median_time([&]() {
            inl_sq = run_custom_ransac(cpnp_P3D, cpnp_P2D_norm, solve_sqpnp, R_sq, t_sq, 500, ransac_thresh, frame_id);
            });

        double t_cpnp = measure_median_time([&]() {
            inl_c = run_custom_ransac(cpnp_P3D, cpnp_P2D_norm, solve_cpnp, R_c, t_c, 500, ransac_thresh, frame_id);
            });

        // 误差统计辅助 (失败或内点不够按失败计)
        auto check = [&](MethodStats& s, double t_val, const Matrix3d& R, const Vector3d& t_vec, int inl) {
            if (inl < 6) return make_pair(-1.0, -1.0);
            s.sum_inliers += inl;
            s.sum_points += n_points;
            double r_err = calc_rot_err(R_gt, R);
            double t_err = calc_trans_err(t_gt, t_vec);
            s.total_time += t_val;
            if (r_err < THRESH_R_DEG && t_err < THRESH_T_M) {
                s.success_count++;
                s.sum_r_err += r_err;
                s.sum_t_err += t_err;
            }
            return make_pair(r_err, t_err);
            };

        auto err_b = check(s_bapnp, t_bapnp, R_b, t_b, inl_b);
        auto err_ep = check(s_epnp, t_epnp, R_ep, t_ep, inl_ep);
        auto err_sq = check(s_sqpnp, t_sqpnp, R_sq, t_sq, inl_sq);
        auto err_c = check(s_cpnp, t_cpnp, R_c, t_c, inl_c);

        outFile << frame_id << " " << n_points << " "
            << inl_b << " " << inl_ep << " " << inl_sq << " " << inl_c << " "
            << t_bapnp << " " << t_epnp << " " << t_sqpnp << " " << t_cpnp << " "
            << err_b.first << " " << err_ep.first << " " << err_sq.first << " " << err_c.first << " "
            << err_b.second << " " << err_ep.second << " " << err_sq.second << " " << err_c.second << endl;

        total_frames++;
        if (total_frames % 10 == 0) cout << "Processed " << total_frames << " frames..." << endl;
    }

    cout << "\n==================================================================================" << endl;
    cout << "  INDEPENDENT RANSAC BENCHMARK RESULTS (" << total_frames << " valid frames)" << endl;
    cout << "  RANSAC params: 500 iters, 6-pt sample, 2 px threshold" << endl;
    cout << "==================================================================================" << endl;

    auto print_stat = [&](MethodStats s) {
        double succ_rate = (total_frames > 0) ? 100.0 * s.success_count / total_frames : 0;
        double avg_time = (total_frames > 0) ? s.total_time / total_frames : 0;
        double avg_r = (s.success_count > 0) ? s.sum_r_err / s.success_count : -1;
        double avg_t = (s.success_count > 0) ? s.sum_t_err / s.success_count : -1;
        double avg_inl = (total_frames > 0) ? (double)s.sum_inliers / total_frames : 0;
        double avg_pts = (total_frames > 0) ? (double)s.sum_points / total_frames : 0;
        double inl_ratio = (avg_pts > 0) ? 100.0 * avg_inl / avg_pts : 0;

        cout << left << setw(15) << s.name
            << " | Time: " << fixed << setprecision(3) << avg_time << " ms"
            << " | Succ: " << setprecision(1) << succ_rate << "%"
            << " | Err(R/t): " << setprecision(3) << avg_r << " deg / " << avg_t << " m"
            << " | Inliers: " << setprecision(0) << avg_inl << "/" << avg_pts
            << " (" << setprecision(1) << inl_ratio << "%)"
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