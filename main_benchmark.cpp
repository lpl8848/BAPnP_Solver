#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <random>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "bapnp.h"
#include "cpnp.h"
#include "ops.h"

using namespace std;
using namespace Eigen;

bool run_cpnp_adapter(const MatrixXd& P_world, const MatrixXd& y_norm, 
                      Matrix3d& R_est, Vector3d& t_est) {
    int n = P_world.cols();

    std::vector<Eigen::Vector3d> pts_3d(n);
    std::vector<Eigen::Vector2d> pts_2d(n);

    for(int i=0; i<n; ++i) {
        pts_3d[i] = P_world.col(i);
        pts_2d[i] = y_norm.col(i).head<2>();
    }

    std::vector<double> params = {1.0, 1.0, 0.0, 0.0};

    Eigen::Vector4d q_out, q_gn;
    Eigen::Vector3d t_out, t_gn;

    bool success = pnpsolver::CPnP(pts_2d, pts_3d, params, q_out, t_out, q_gn, t_gn);

    if (success) {
        Eigen::Quaterniond q(q_gn(0), q_gn(1), q_gn(2), q_gn(3));
        q.normalize();
        R_est = q.toRotationMatrix();
        t_est = t_gn;
    }

    return success;
}

void eigen2cv(const MatrixXd& P_world, const MatrixXd& y_norm, 
              std::vector<cv::Point3d>& pts3d, std::vector<cv::Point2d>& pts2d) {
    int n = P_world.cols();
    pts3d.clear(); 
    pts2d.clear();
    pts3d.reserve(n); 
    pts2d.reserve(n);
    for(int i=0; i<n; ++i) {
        pts3d.emplace_back(P_world(0,i), P_world(1,i), P_world(2,i));
        pts2d.emplace_back(y_norm(0,i), y_norm(1,i));
    }
}

int main() {
    ofstream outFile("runtime_benchmark_result.txt");
    if (!outFile.is_open()) {
        cerr << "Error: Could not open output file." << endl;
        return -1;
    }
    
    outFile << "N BAPnP EPnP EPnP_LM CPnP SQPnP" << endl; 

    vector<int> N_list = {6, 10, 20, 50, 100, 200, 500, 1000, 2000};
    
    srand(2025); 

    bool has_sqpnp = false;
    #if CV_VERSION_MAJOR >= 4 && (CV_VERSION_MINOR > 5 || (CV_VERSION_MINOR == 5 && CV_VERSION_REVISION >= 3))
    has_sqpnp = true;
    cout << "[Info] OpenCV SQPnP support detected." << endl;
    #else
    cout << "[Warning] OpenCV version too old for SQPnP. Skipping SQPnP." << endl;
    #endif

    cout << "\n=========================================================" << endl;
    cout << "  RUNNING RUNTIME BENCHMARK (ms) - 10000 Trials per N" << endl;
    cout << "=========================================================" << endl;

    const double f_sim = 800.0;
    const double cx_sim = 320.0;
    const double cy_sim = 240.0;

    for (int n : N_list) {
        double t_bapnp = 0, t_epnp = 0, t_epnp_lm = 0, t_cpnp = 0, t_sqpnp = 0;
        int trials = 10000;

        std::vector<cv::Point3d> cv_pts3d; cv_pts3d.reserve(n);
        std::vector<cv::Point2d> cv_pts2d; cv_pts2d.reserve(n);
        
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);

        std::default_random_engine generator(2025 + n);
        std::normal_distribution<double> noise_dist(0.0, 2.0);

        for (int k = 0; k < trials; ++k) {
            MatrixXd P_world = MatrixXd::Random(3, n) * 10.0; 
            P_world.row(2).array() += 20.0; 
            
            Matrix3d R_gt = Quaterniond::UnitRandom().toRotationMatrix();
            Vector3d t_gt = Vector3d::Random();
            
            MatrixXd P_cam = (R_gt * P_world).colwise() + t_gt;
            MatrixXd y_norm(3, n);

            for(int i=0; i<n; ++i) {
                double X = P_cam(0, i);
                double Y = P_cam(1, i);
                double Z = P_cam(2, i);

                double u_clean = f_sim * (X / Z) + cx_sim;
                double v_clean = f_sim * (Y / Z) + cy_sim;

                double u_noisy = u_clean + noise_dist(generator);
                double v_noisy = v_clean + noise_dist(generator);

                y_norm(0, i) = (u_noisy - cx_sim) / f_sim;
                y_norm(1, i) = (v_noisy - cy_sim) / f_sim;
                y_norm(2, i) = 1.0;
            }

            eigen2cv(P_world, y_norm, cv_pts3d, cv_pts2d);

            Matrix3d R_est; Vector3d t_est;
            auto s1 = chrono::high_resolution_clock::now();
            BAPnP::solve(y_norm, P_world, R_est, t_est);
            auto e1 = chrono::high_resolution_clock::now();
            t_bapnp += chrono::duration<double, milli>(e1 - s1).count();

            cv::Mat rvec, tvec;
            auto s2 = chrono::high_resolution_clock::now();
            cv::solvePnP(cv_pts3d, cv_pts2d, K, D, rvec, tvec, false, cv::SOLVEPNP_EPNP);
            auto e2 = chrono::high_resolution_clock::now();
            double time_linear = chrono::duration<double, milli>(e2 - s2).count();
            t_epnp += time_linear;

            cv::Mat rvec_ref = rvec.clone();
            cv::Mat tvec_ref = tvec.clone();
            
            auto s3 = chrono::high_resolution_clock::now();
            cv::solvePnPRefineLM(cv_pts3d, cv_pts2d, K, D, rvec_ref, tvec_ref);
            auto e3 = chrono::high_resolution_clock::now();
            double time_refine = chrono::duration<double, milli>(e3 - s3).count();
            
            t_epnp_lm += (time_linear + time_refine);

            auto s4 = chrono::high_resolution_clock::now();
            run_cpnp_adapter(P_world, y_norm, R_est, t_est);
            auto e4 = chrono::high_resolution_clock::now();
            t_cpnp += chrono::duration<double, milli>(e4 - s4).count();

            if (has_sqpnp) {
                cv::Mat rvec_sq, tvec_sq;
                auto s5 = chrono::high_resolution_clock::now();
                cv::solvePnP(cv_pts3d, cv_pts2d, K, D, rvec_sq, tvec_sq, false, cv::SOLVEPNP_SQPNP);
                auto e5 = chrono::high_resolution_clock::now();
                t_sqpnp += chrono::duration<double, milli>(e5 - s5).count();
            }
        }

        double avg_bapnp    = t_bapnp / trials;
        double avg_epnp     = t_epnp / trials;
        double avg_epnp_lm  = t_epnp_lm / trials;
        double avg_cpnp     = t_cpnp / trials;
        double avg_sqpnp    = t_sqpnp / trials;

        cout << left << "N=" << setw(5) << n 
             << " | BAPnP: " << fixed << setprecision(4) << avg_bapnp 
             << " | EPnP: " << avg_epnp 
             << " | EPnP+LM: " << avg_epnp_lm 
             << " | CPnP: " << avg_cpnp 
             << " | SQPnP: " << avg_sqpnp << endl;

        outFile << n << " " << avg_bapnp << " " << avg_epnp << " " << avg_epnp_lm << " " << avg_cpnp << " " << avg_sqpnp << endl;
    }

    outFile.close();
    cout << "\nBenchmark finished. Results saved to runtime_benchmark_result.txt" << endl;
    return 0;
}
