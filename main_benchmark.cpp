#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>

// Eigen
#include <Eigen/Dense>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

//
#include "bapnp.h"
#include "cpnp.h" 
#include "ops.h"

using namespace std;
using namespace Eigen;

// ==========================================
//  辅助函数 1: 适配器 - 调用 CPnP
// ==========================================
// 将 Eigen 数据转换为 CPnP 需要的 std::vector 格式，并处理命名空间
bool run_cpnp_adapter(const MatrixXd& P_world, const MatrixXd& y_norm, 
                      Matrix3d& R_est, Vector3d& t_est) {
    int n = P_world.cols();

    // 1. 数据转换: Eigen -> std::vector
    std::vector<Eigen::Vector3d> pts_3d(n);
    std::vector<Eigen::Vector2d> pts_2d(n);

    for(int i=0; i<n; ++i) {
        pts_3d[i] = P_world.col(i);
        pts_2d[i] = y_norm.col(i).head<2>(); // 取前两维 (u, v)
    }

    // 2. 准备参数 (fx=1, fy=1, cx=0, cy=0 因为是归一化坐标)
    std::vector<double> params = {1.0, 1.0, 0.0, 0.0};

    // 3. 准备输出变量
    Eigen::Vector4d q_out, q_gn;
    Eigen::Vector3d t_out, t_gn;

    // 4. 调用算法 
    bool success = pnpsolver::CPnP(pts_2d, pts_3d, params, q_out, t_out, q_gn, t_gn);

    // 5. 结果转换: 选取优化后的结果 (q_gn, t_gn)
    if (success) {
        Eigen::Quaterniond q(q_gn(0), q_gn(1), q_gn(2), q_gn(3));
        q.normalize();
        R_est = q.toRotationMatrix();
        t_est = t_gn;
    }

    return success;
}

// ==========================================
//  辅助函数 2: 转换器 - Eigen 转 OpenCV
// ==========================================
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

// ==========================================
//  主函数
// ==========================================
int main() {
    // 设置输出文件
    ofstream outFile("runtime_benchmark_result.txt");
    if (!outFile.is_open()) {
        cerr << "Error: Could not open output file." << endl;
        return -1;
    }
    
    // 表头
    outFile << "N BAPnP EPnP EPnP_LM CPnP SQPnP" << endl; 

    // 测试点数序列
    vector<int> N_list = {6, 10, 20, 50, 100, 200, 500, 1000};
    
    // 固定随机种子，保证每次运行数据一致
    srand(2025); 

    // 检查 OpenCV 版本是否支持 SQPnP
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

    // --- 主循环：遍历不同的点数 N ---
    for (int n : N_list) {
        double t_bapnp = 0, t_epnp = 0, t_epnp_lm = 0, t_cpnp = 0, t_sqpnp = 0;
        int trials = 10000;

        // 预分配 OpenCV 容器内存，避免 Loop 内反复申请影响测速
        std::vector<cv::Point3d> cv_pts3d; cv_pts3d.reserve(n);
        std::vector<cv::Point2d> cv_pts2d; cv_pts2d.reserve(n);
        
        // OpenCV 相机参数 (归一化坐标下 K为单位阵, D为0)
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);

        // --- Trials 循环 ---
        for (int k = 0; k < trials; ++k) {
            // 1. 数据生成 (Data Generation)
            MatrixXd P_world = MatrixXd::Random(3, n) * 10.0; 
            P_world.row(2).array() += 20.0; // Z轴推远，确保点在相机前方
            
            Matrix3d R_gt = Quaterniond::UnitRandom().toRotationMatrix();
            Vector3d t_gt = Vector3d::Random();
            
            // 投影得到归一化坐标
            MatrixXd P_cam = (R_gt * P_world).colwise() + t_gt;
            MatrixXd y_norm(3, n);
            for(int i=0; i<n; ++i) y_norm.col(i) = P_cam.col(i) / P_cam(2, i);

            // 转换为 OpenCV 格式
            eigen2cv(P_world, y_norm, cv_pts3d, cv_pts2d);

            // ------------------------------------------------------
            // Algorithm 1: BAPnP (Ours) - Linear + GN
            // ------------------------------------------------------
            Matrix3d R_est; Vector3d t_est;
            auto s1 = chrono::high_resolution_clock::now();
            BAPnP::solve(y_norm, P_world, R_est, t_est);
            auto e1 = chrono::high_resolution_clock::now();
            t_bapnp += chrono::duration<double, milli>(e1 - s1).count();

            // ------------------------------------------------------
            // Algorithm 2: EPnP (OpenCV) - Linear Only
            // ------------------------------------------------------
            cv::Mat rvec, tvec;
            auto s2 = chrono::high_resolution_clock::now();
            cv::solvePnP(cv_pts3d, cv_pts2d, K, D, rvec, tvec, false, cv::SOLVEPNP_EPNP);
            auto e2 = chrono::high_resolution_clock::now();
            double time_linear = chrono::duration<double, milli>(e2 - s2).count();
            t_epnp += time_linear;

            // ------------------------------------------------------
            // Algorithm 3: EPnP + LM (OpenCV) - Linear + Refine
            // ------------------------------------------------------
            // 关键逻辑：必须串联 Linear 和 Refine 两个步骤
            // 首先深拷贝一份 Linear 结果作为初值
            cv::Mat rvec_ref = rvec.clone();
            cv::Mat tvec_ref = tvec.clone();
            
            auto s3 = chrono::high_resolution_clock::now();
            // 显式调用 LM Refine (注意：OpenCV 4.x+ 支持此函数)
            cv::solvePnPRefineLM(cv_pts3d, cv_pts2d, K, D, rvec_ref, tvec_ref);
            auto e3 = chrono::high_resolution_clock::now();
            double time_refine = chrono::duration<double, milli>(e3 - s3).count();
            
            // 总时间 = 线性初始化 + 迭代优化
            t_epnp_lm += (time_linear + time_refine);

            // ------------------------------------------------------
            // Algorithm 4: CPnP (State-of-the-Art)
            // ------------------------------------------------------
            auto s4 = chrono::high_resolution_clock::now();
            run_cpnp_adapter(P_world, y_norm, R_est, t_est);
            auto e4 = chrono::high_resolution_clock::now();
            t_cpnp += chrono::duration<double, milli>(e4 - s4).count();

            // ------------------------------------------------------
            // Algorithm 5: SQPnP (OpenCV) - Non-linear Global
            // ------------------------------------------------------
            if (has_sqpnp) {
                cv::Mat rvec_sq, tvec_sq;
                auto s5 = chrono::high_resolution_clock::now();
                cv::solvePnP(cv_pts3d, cv_pts2d, K, D, rvec_sq, tvec_sq, false, cv::SOLVEPNP_SQPNP);
                auto e5 = chrono::high_resolution_clock::now();
                t_sqpnp += chrono::duration<double, milli>(e5 - s5).count();
            }
        }

        // 计算平均耗时 (ms)
        double avg_bapnp   = t_bapnp / trials;
        double avg_epnp    = t_epnp / trials;
        double avg_epnp_lm = t_epnp_lm / trials;
        double avg_cpnp    = t_cpnp / trials;
        double avg_sqpnp   = t_sqpnp / trials;

        // 打印结果到控制台
        cout << left << "N=" << setw(5) << n 
             << " | BAPnP: " << fixed << setprecision(4) << avg_bapnp 
             << " | EPnP: " << avg_epnp 
             << " | EPnP+LM: " << avg_epnp_lm 
             << " | CPnP: " << avg_cpnp 
             << " | SQPnP: " << avg_sqpnp << endl;

        // 写入文件
        outFile << n << " " << avg_bapnp << " " << avg_epnp << " " << avg_epnp_lm << " " << avg_cpnp << " " << avg_sqpnp << endl;
    }

    outFile.close();
    cout << "\nBenchmark finished. Results saved to runtime_benchmark_result.txt" << endl;
    return 0;
}
