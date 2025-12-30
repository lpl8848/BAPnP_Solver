#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <cmath>
#include <iomanip>

// Eigen
#include <Eigen/Dense>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

// --- 引入算法头文件 ---
#include "bapnp.h"
#include "cpnp.h" 
#include "ops.h"

using namespace std;
using namespace Eigen;

// --- 评测标准 ---
const double THRESH_R_DEG = 3.0;  // 旋转成功阈值 (度)
const double THRESH_T_M   = 0.1;  // 平移成功阈值 (米)

// 辅助：计算旋转误差 (角度)
double calc_rot_err(const Matrix3d& R1, const Matrix3d& R2) {
    double tr = (R1 * R2.transpose()).trace();
    double val = (tr - 1.0) / 2.0;
    if (val > 1.0) val = 1.0; else if (val < -1.0) val = -1.0;
    return acos(val) * 180.0 / M_PI;
}

// 辅助：计算平移误差 (模长)
double calc_trans_err(const Vector3d& t1, const Vector3d& t2) {
    return (t1 - t2).norm();
}

// 结构体：方便统计结果
struct MethodStats {
    string name;
    double total_time = 0;
    int success_count = 0;
    double sum_r_err = 0; // 仅累加成功帧
    double sum_t_err = 0; // 仅累加成功帧
};

int main() {
    // 输入文件 (请确保文件名正确)
    ifstream inFile("tum_data_export.txt");
    if (!inFile.is_open()) {
        cerr << "Error: Cannot open tum_data_export.txt. Did you run 'dos2unix'?" << endl;
        return -1;
    }

    // 输出文件
    ofstream outFile("tum_results_final.txt");
    outFile << "Frame N BAPnP_Time EPnP_Time EPnPLM_Time SQPnP_Time CPnP_Time "
            << "BAPnP_R EPnP_R EPnPLM_R SQPnP_R CPnP_R "
            << "BAPnP_t EPnP_t EPnPLM_t SQPnP_t CPnP_t" << endl;

    // 初始化统计器
    MethodStats s_bapnp   = {"BAPnP"};
    MethodStats s_epnp    = {"EPnP"};
    MethodStats s_epnp_lm = {"EPnP+LM"}; // 新增: 显式统计 Refine 后的结果
    MethodStats s_sqpnp   = {"SQPnP"};
    MethodStats s_cpnp    = {"CPnP"};

    string token;
    int total_frames = 0;

    // 检查 SQPnP 支持
    bool has_sqpnp = false;
    #if CV_VERSION_MAJOR >= 4 && (CV_VERSION_MINOR > 5 || (CV_VERSION_MINOR == 5 && CV_VERSION_REVISION >= 3))
    has_sqpnp = true;
    #endif

    cout << "Starting Ultimate TUM Benchmark (Normalized CPnP Fix)..." << endl;

    while (inFile >> token) {
        if (token != "FRAME") continue;

        int frame_id, n_points;
        inFile >> frame_id >> n_points;

        // 1. 读取内参
        double fx, fy, cx, cy;
        inFile >> fx >> fy >> cx >> cy;

        // 2. 读取 GT Pose
        Matrix3d R_gt; Vector3d t_gt;
        for(int r=0; r<3; ++r) {
            inFile >> R_gt(r,0) >> R_gt(r,1) >> R_gt(r,2) >> t_gt(r);
        }

        // 3. 读取点云 & 准备数据容器
        MatrixXd P_world(3, n_points);     // for BAPnP & CPnP
        MatrixXd pts_2d_norm(3, n_points); // for BAPnP & CPnP (Normalized)

        // for OpenCV (EPnP, SQPnP 需要像素坐标)
        vector<cv::Point3d> cv_P3D; cv_P3D.reserve(n_points);
        vector<cv::Point2d> cv_P2D; cv_P2D.reserve(n_points);

        // for CPnP (Eigen vector)
        vector<Vector3d> cpnp_P3D; cpnp_P3D.reserve(n_points);
        vector<Vector2d> cpnp_P2D_norm; cpnp_P2D_norm.reserve(n_points); // 改为存归一化坐标

        for (int i = 0; i < n_points; ++i) {
            double X, Y, Z, u, v;
            inFile >> X >> Y >> Z >> u >> v;

            // BAPnP Data
            P_world(0, i) = X; P_world(1, i) = Y; P_world(2, i) = Z;

            //计算归一化坐标
            double u_norm = (u - cx) / fx;
            double v_norm = (v - cy) / fy;
            
            pts_2d_norm(0, i) = u_norm;
            pts_2d_norm(1, i) = v_norm;
            pts_2d_norm(2, i) = 1.0;

            // OpenCV Data (像素坐标)
            cv_P3D.emplace_back(X, Y, Z);
            cv_P2D.emplace_back(u, v);

            // CPnP Data (改为存归一化坐标)
            cpnp_P3D.emplace_back(X, Y, Z);
            cpnp_P2D_norm.emplace_back(u_norm, v_norm);
        }

        // OpenCV 参数
        cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat dist = cv::Mat::zeros(4,1, CV_64F);

        // ======================= 1. BAPnP (Ours) =======================
        Matrix3d R_bapnp; Vector3d t_bapnp;
        auto t1 = chrono::high_resolution_clock::now();
        BAPnP::solve(pts_2d_norm, P_world, R_bapnp, t_bapnp);
        auto t2 = chrono::high_resolution_clock::now();
        double time_bapnp = chrono::duration<double, milli>(t2 - t1).count();

        // ======================= 2. EPnP (Linear Only) =======================
        cv::Mat rvec_lin, tvec_lin;
        auto t3 = chrono::high_resolution_clock::now();
        cv::solvePnP(cv_P3D, cv_P2D, K, dist, rvec_lin, tvec_lin, false, cv::SOLVEPNP_EPNP);
        auto t4 = chrono::high_resolution_clock::now();
        double time_epnp = chrono::duration<double, milli>(t4 - t3).count();

        // 转换结果
        cv::Mat R_cv_lin; cv::Rodrigues(rvec_lin, R_cv_lin);
        Matrix3d R_epnp_e; Vector3d t_epnp_e;
        cv::cv2eigen(R_cv_lin, R_epnp_e);
        t_epnp_e << tvec_lin.at<double>(0), tvec_lin.at<double>(1), tvec_lin.at<double>(2);

        // ======================= 3. EPnP + LM (Refinement) =======================
        // 复用 Linear 结果作为初值 (深拷贝)
        cv::Mat rvec_ref = rvec_lin.clone();
        cv::Mat tvec_ref = tvec_lin.clone();

        auto t3_r = chrono::high_resolution_clock::now();
        // 显式调用 LM 优化
        cv::solvePnPRefineLM(cv_P3D, cv_P2D, K, dist, rvec_ref, tvec_ref);
        auto t4_r = chrono::high_resolution_clock::now();
        
        // 总时间 = Linear时间 + Refine时间
        double time_epnp_lm = time_epnp + chrono::duration<double, milli>(t4_r - t3_r).count();

        // 转换结果
        cv::Mat R_cv_ref; cv::Rodrigues(rvec_ref, R_cv_ref);
        Matrix3d R_epnplm_e; Vector3d t_epnplm_e;
        cv::cv2eigen(R_cv_ref, R_epnplm_e);
        t_epnplm_e << tvec_ref.at<double>(0), tvec_ref.at<double>(1), tvec_ref.at<double>(2);

        // ======================= 4. CPnP (Optimized Input) =======================
        Matrix3d R_cpnp_e = Matrix3d::Identity();
        Vector3d t_cpnp_e = Vector3d::Zero();
        double time_cpnp = 0;

        {
            // 参数设为单位内参，因为输入已经是归一化坐标了
            std::vector<double> cpnp_params = {1.0, 1.0, 0.0, 0.0};
            Vector4d q_out, q_gn;
            Vector3d t_out, t_gn;

            auto t7 = chrono::high_resolution_clock::now();
            // 传入 cpnp_P2D_norm (归一化坐标)
            pnpsolver::CPnP(cpnp_P2D_norm, cpnp_P3D, cpnp_params, q_out, t_out, q_gn, t_gn);
            auto t8 = chrono::high_resolution_clock::now();
            time_cpnp = chrono::duration<double, milli>(t8 - t7).count();

            Quaterniond q(q_gn(0), q_gn(1), q_gn(2), q_gn(3));
            q.normalize();
            R_cpnp_e = q.toRotationMatrix();
            t_cpnp_e = t_gn;
        }

        // ======================= 5. SQPnP (OpenCV) =======================
        double time_sqpnp = 0;
        Matrix3d R_sqpnp_e = Matrix3d::Identity();
        Vector3d t_sqpnp_e = Vector3d::Zero();

        if (has_sqpnp) {
            cv::Mat rvec_sq, tvec_sq;
            auto t5 = chrono::high_resolution_clock::now();
            cv::solvePnP(cv_P3D, cv_P2D, K, dist, rvec_sq, tvec_sq, false, cv::SOLVEPNP_SQPNP);
            auto t6 = chrono::high_resolution_clock::now();
            time_sqpnp = chrono::duration<double, milli>(t6 - t5).count();

            cv::Mat R_cv_sq; cv::Rodrigues(rvec_sq, R_cv_sq);
            cv::cv2eigen(R_cv_sq, R_sqpnp_e);
            t_sqpnp_e << tvec_sq.at<double>(0), tvec_sq.at<double>(1), tvec_sq.at<double>(2);
        }

        // --- 统计误差 ---
        // Helper Lambda
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

        auto e_b   = check(s_bapnp,   time_bapnp,   R_bapnp,    t_bapnp);
        auto e_ep  = check(s_epnp,    time_epnp,    R_epnp_e,   t_epnp_e);
        auto e_eplm= check(s_epnp_lm, time_epnp_lm, R_epnplm_e, t_epnplm_e); // 新增
        auto e_c   = check(s_cpnp,    time_cpnp,    R_cpnp_e,   t_cpnp_e);
        auto e_sq  = check(s_sqpnp,   time_sqpnp,   R_sqpnp_e,  t_sqpnp_e);

        // 写入文件
        outFile << frame_id << " " << n_points << " "
                << time_bapnp << " " << time_epnp << " " << time_epnp_lm << " " << time_sqpnp << " " << time_cpnp << " "
                << e_b.first << " " << e_ep.first << " " << e_eplm.first << " " << e_sq.first << " " << e_c.first << " "
                << e_b.second << " " << e_ep.second << " " << e_eplm.second << " " << e_sq.second << " " << e_c.second
                << endl;

        total_frames++;
        if (total_frames % 50 == 0) cout << "Processed " << total_frames << " frames..." << endl;
    }

    // --- 打印报表 ---
    cout << "\n==================================================================================" << endl;
    cout << "  TUM BENCHMARK FINAL RESULTS (" << total_frames << " frames)" << endl;
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
    print_stat(s_epnp_lm);  
    print_stat(s_cpnp);
    print_stat(s_sqpnp);
    cout << "==================================================================================" << endl;

    inFile.close();
    outFile.close();
    return 0;
}
