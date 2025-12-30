#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <chrono>
#include <cmath>
#include <iomanip>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

// 引入算法
#include "bapnp.h"
#include "cpnp.h" 
#include "ops.h"

using namespace std;
using namespace Eigen;

// --- 配置参数 ---
const double FILTER_ERR_THRESH = 1.0; // COLMAP 3D点过滤阈值 (px)
const int MIN_INLIERS = 6;            // 最少点数 (放宽一点以测试极少点情况)
const double THRESH_R = 2.0;          // 旋转成功阈值 (deg)
const double THRESH_T = 1.0;          // 平移成功阈值 (m)

// --- 数据结构 ---
struct Camera {
    int id;
    int width, height;
    Matrix3d K;
};

struct Point3D {
    int id;
    Vector3d xyz;
    double error;
};

struct Image {
    int id;
    string name;
    Matrix3d R_gt;
    Vector3d t_gt;
    int camera_id;
    vector<Vector2d> pts2d;   // 2D 观测
    vector<int> p3d_ids;      // 对应的 3D 点 ID
};

// --- 辅助函数：四元数转旋转矩阵 ---
Matrix3d quat2rot(double qw, double qx, double qy, double qz) {
    Quaterniond q(qw, qx, qy, qz);
    q.normalize();
    return q.toRotationMatrix();
}

// --- 核心函数：读取 COLMAP 数据 ---
bool read_colmap(const string& base_path,
                 map<int, Point3D>& points3d,
                 map<int, Camera>& cameras,
                 vector<Image>& images) {

    cout << "Loading COLMAP data from: " << base_path << endl;

    // 1. 读取 points3D.txt
    ifstream f_pts(base_path + "/points3D.txt");
    if (!f_pts.is_open()) return false;

    string line;
    while (getline(f_pts, line)) {
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        Point3D p;
        int r, g, b;
        ss >> p.id >> p.xyz(0) >> p.xyz(1) >> p.xyz(2) >> r >> g >> b >> p.error;

        if (p.error <= FILTER_ERR_THRESH) {
            points3d[p.id] = p;
        }
    }
    cout << "Loaded " << points3d.size() << " valid 3D points." << endl;

    // 2. 读取 cameras.txt
    ifstream f_cam(base_path + "/cameras.txt");
    if (!f_cam.is_open()) return false;
    while (getline(f_cam, line)) {
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        Camera cam;
        string model;
        ss >> cam.id >> model >> cam.width >> cam.height;

        double p1, p2, p3, p4;
        ss >> p1 >> p2 >> p3;

        cam.K = Matrix3d::Identity();
        if (model == "PINHOLE") {
            ss >> p4;
            cam.K(0,0) = p1; // fx
            cam.K(1,1) = p2; // fy
            cam.K(0,2) = p3; // cx
            cam.K(1,2) = p4; // cy
        } else {
            // SIMPLE_RADIAL: f, cx, cy, k
            cam.K(0,0) = p1; cam.K(1,1) = p1;
            cam.K(0,2) = p2; cam.K(1,2) = p3;
            // 注意：这里忽略了畸变 k，假设输入点已去畸变或畸变很小
            // 如果追求极致精度，应在这里读取 k，并在后续像素转归一化时进行去畸变
        }
        cameras[cam.id] = cam;
    }

    // 3. 读取 images.txt
    ifstream f_img(base_path + "/images.txt");
    if (!f_img.is_open()) return false;
    while (getline(f_img, line)) {
        if (line.empty() || line[0] == '#') continue;

        stringstream ss(line);
        Image img;
        double qw, qx, qy, qz, tx, ty, tz;
        ss >> img.id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> img.camera_id >> img.name;
        img.R_gt = quat2rot(qw, qx, qy, qz);
        img.t_gt = Vector3d(tx, ty, tz);

        getline(f_img, line);
        stringstream ss2(line);
        double x, y;
        int p3d_id;
        while (ss2 >> x >> y >> p3d_id) {
            if (p3d_id != -1 && points3d.count(p3d_id)) {
                img.pts2d.emplace_back(x, y);
                img.p3d_ids.push_back(p3d_id);
            }
        }

        if (img.pts2d.size() >= MIN_INLIERS) {
            images.push_back(img);
        }
    }
    cout << "Loaded " << images.size() << " valid images." << endl;
    return true;
}

// --- 统计工具 ---
struct Stats {
    string name;
    int success = 0;
    int total = 0;
    double time_sum = 0;
    double r_err_sum = 0;
    double t_err_sum = 0;
};

void update_stats(Stats& s, double t_ms, double r_err, double t_err) {
    s.total++;
    s.time_sum += t_ms;
    if (r_err < THRESH_R && t_err < THRESH_T) {
        s.success++;
        s.r_err_sum += r_err;
        s.t_err_sum += t_err;
    }
}

// ======================= 主程序 =======================
int main(int argc, char** argv) {
    string dataset_path = "/home/luo/projects/MyBAPnP/south-building";
    if (argc > 1) dataset_path = argv[1];

    // 1. 读取数据
    map<int, Point3D> points3d;
    map<int, Camera> cameras;
    vector<Image> images;

    if (!read_colmap(dataset_path, points3d, cameras, images)) {
        cerr << "Failed to read COLMAP data from: " << dataset_path << endl;
        cerr << "Usage: ./run_colmap_bench <path_to_colmap_folder>" << endl;
        return -1;
    }

    // 2. 初始化算法统计
    Stats s_bapnp   = {"BAPnP"};
    Stats s_epnp    = {"EPnP"};
    Stats s_epnp_lm = {"EPnP+LM"};
    Stats s_sqpnp   = {"SQPnP"};
    Stats s_cpnp    = {"CPnP"};

    cout << "\nStarting Benchmark on " << images.size() << " images..." << endl;

    // 3. 主循环
    int count = 0;
    for (const auto& img : images) {
        int N = img.pts2d.size();
        Matrix3d K = cameras[img.camera_id].K;

        // 准备数据容器
        MatrixXd P_world(3, N);
        MatrixXd pts_norm(3, N);
        
        vector<cv::Point3d> cv_P3D;
        vector<cv::Point2d> cv_P2D;
        
        vector<Vector3d> cpnp_P3D;
        vector<Vector2d> cpnp_P2D_norm; // 改用归一化坐标

        for (int i = 0; i < N; ++i) {
            Vector3d Xw = points3d[img.p3d_ids[i]].xyz;
            Vector2d uv = img.pts2d[i];

            // BAPnP (Matrix)
            P_world.col(i) = Xw;
            Vector3d uv_h(uv(0), uv(1), 1.0);
            Vector3d yn = K.inverse() * uv_h; // 归一化
            pts_norm.col(i) = yn;

            // OpenCV (cv::Point)
            cv_P3D.emplace_back(Xw(0), Xw(1), Xw(2));
            cv_P2D.emplace_back(uv(0), uv(1));

            // CPnP (std::vector<Eigen>)
            cpnp_P3D.emplace_back(Xw);
            // 这里存入归一化坐标，而非像素坐标
            cpnp_P2D_norm.emplace_back(yn(0), yn(1));
        }

        // --- A. 运行 BAPnP (Ours) ---
        Matrix3d R_est; Vector3d t_est;
        auto t1 = chrono::high_resolution_clock::now();
        BAPnP::solve(pts_norm, P_world, R_est, t_est);
        auto t2 = chrono::high_resolution_clock::now();

        double err_r_b = (AngleAxisd(img.R_gt.transpose() * R_est).angle()) * 180.0 / M_PI;
        double err_t_b = (t_est - img.t_gt).norm();
        update_stats(s_bapnp, chrono::duration<double, milli>(t2 - t1).count(), err_r_b, err_t_b);

        // --- B. 运行 EPnP (Standard Baseline) ---
        cv::Mat cv_K(3,3,CV_64F), rvec, tvec, dist=cv::Mat::zeros(4,1,CV_64F);
        cv::eigen2cv(K, cv_K); 

        auto t3 = chrono::high_resolution_clock::now();
        // Step 1: Linear Only
        cv::solvePnP(cv_P3D, cv_P2D, cv_K, dist, rvec, tvec, false, cv::SOLVEPNP_EPNP);
        auto t4 = chrono::high_resolution_clock::now();
        double time_epnp_lin = chrono::duration<double, milli>(t4 - t3).count();

        // 统计 Linear 结果
        cv::Mat R_cv; cv::Rodrigues(rvec, R_cv);
        Matrix3d R_epnp; cv::cv2eigen(R_cv, R_epnp);
        Vector3d t_epnp(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

        double err_r_e = (AngleAxisd(img.R_gt.transpose() * R_epnp).angle()) * 180.0 / M_PI;
        double err_t_e = (t_epnp - img.t_gt).norm();
        update_stats(s_epnp, time_epnp_lin, err_r_e, err_t_e);

        // --- B2. 运行 EPnP + LM (Robust Baseline) ---
        // 复用 Linear 结果作为初值 (深拷贝)
        cv::Mat rvec_ref = rvec.clone();
        cv::Mat tvec_ref = tvec.clone();

        auto t3_r = chrono::high_resolution_clock::now();
        // 显式 Refine
        cv::solvePnPRefineLM(cv_P3D, cv_P2D, cv_K, dist, rvec_ref, tvec_ref);
        auto t4_r = chrono::high_resolution_clock::now();
        double time_refine = chrono::duration<double, milli>(t4_r - t3_r).count();

        // 统计 Refine 结果
        cv::Mat R_cv_ref; cv::Rodrigues(rvec_ref, R_cv_ref);
        Matrix3d R_epnp_lm; cv::cv2eigen(R_cv_ref, R_epnp_lm);
        Vector3d t_epnp_lm(tvec_ref.at<double>(0), tvec_ref.at<double>(1), tvec_ref.at<double>(2));

        double err_r_elm = (AngleAxisd(img.R_gt.transpose() * R_epnp_lm).angle()) * 180.0 / M_PI;
        double err_t_elm = (t_epnp_lm - img.t_gt).norm();
        update_stats(s_epnp_lm, time_epnp_lin + time_refine, err_r_elm, err_t_elm);

        // --- C. 运行 CPnP ---
        {
            // 参数设为单位内参，因为输入已经是归一化坐标了
            std::vector<double> params = {1.0, 1.0, 0.0, 0.0};
            Eigen::Vector4d q_out, q_gn;
            Eigen::Vector3d t_out, t_gn;

            auto t5 = chrono::high_resolution_clock::now();
            // 传入 cpnp_P2D_norm (归一化坐标)
            pnpsolver::CPnP(cpnp_P2D_norm, cpnp_P3D, params, q_out, t_out, q_gn, t_gn);
            auto t6 = chrono::high_resolution_clock::now();

            Quaterniond q_c(q_gn(0), q_gn(1), q_gn(2), q_gn(3));
            q_c.normalize();
            Matrix3d R_cpnp = q_c.toRotationMatrix();
            Vector3d t_cpnp = t_gn;

            double err_r_c = (AngleAxisd(img.R_gt.transpose() * R_cpnp).angle()) * 180.0 / M_PI;
            double err_t_c = (t_cpnp - img.t_gt).norm();
            update_stats(s_cpnp, chrono::duration<double, milli>(t6 - t5).count(), err_r_c, err_t_c);
        }

        // --- D. 运行 SQPnP ---
        #if CV_VERSION_MAJOR >= 4 && (CV_VERSION_MINOR > 5 || (CV_VERSION_MINOR == 5 && CV_VERSION_REVISION >= 3))
        {
            cv::Mat rvec_sq, tvec_sq;
            auto t7 = chrono::high_resolution_clock::now();
            cv::solvePnP(cv_P3D, cv_P2D, cv_K, dist, rvec_sq, tvec_sq, false, cv::SOLVEPNP_SQPNP);
            auto t8 = chrono::high_resolution_clock::now();

            cv::Mat R_cv_sq; cv::Rodrigues(rvec_sq, R_cv_sq);
            Matrix3d R_sq; cv::cv2eigen(R_cv_sq, R_sq);
            Vector3d t_sq(tvec_sq.at<double>(0), tvec_sq.at<double>(1), tvec_sq.at<double>(2));

            double err_r_s = (AngleAxisd(img.R_gt.transpose() * R_sq).angle()) * 180.0 / M_PI;
            double err_t_s = (t_sq - img.t_gt).norm();
            update_stats(s_sqpnp, chrono::duration<double, milli>(t8 - t7).count(), err_r_s, err_t_s);
        }
        #endif

        if (++count % 50 == 0) cout << "Processed " << count << " images." << endl;
    }

    // 4. 打印报表
    cout << "\n=== South Building Benchmark Results ===" << endl;
    auto print = [](Stats s) {
        if (s.total == 0) return;
        double succ_rate = 100.0 * s.success / s.total;
        double mean_r = s.success > 0 ? s.r_err_sum / s.success : -1;
        double mean_t = s.success > 0 ? s.t_err_sum / s.success : -1;
        cout << left << setw(10) << s.name
        << " | Succ: " << fixed << setprecision(1) << succ_rate << "%"
        << " | Time: " << setprecision(3) << s.time_sum/s.total << " ms"
        << " | Err: " << setprecision(3) << mean_r << " deg / " << mean_t << " m" << endl;
    };
    print(s_bapnp);
    print(s_epnp);    // Linear Only
    print(s_epnp_lm); // Linear + Refine (Strong Baseline)
    print(s_cpnp);
    print(s_sqpnp);

    return 0;
}
