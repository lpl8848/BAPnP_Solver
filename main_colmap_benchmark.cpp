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
#include <algorithm> 
#include <random>   
#include <numeric>  

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>


#include "bapnp.h"
#include "cpnp.h"
#include "ops.h" 

using namespace std;
using namespace Eigen;


const double FILTER_ERR_THRESH = 1.0; 
const int MIN_INLIERS = 6;            
const double THRESH_R = 2.0;          
const double THRESH_T = 1.0;          
const int BENCHMARK_ITERATIONS = 1000; 


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
    vector<Vector2d> pts2d;   
    vector<int> p3d_ids;      
};

struct Stats {
    string name;
    int success = 0;
    int total = 0;
    double time_sum = 0;
    double r_err_sum = 0;
    double t_err_sum = 0;
    
    
    void reset() {
        success = 0; total = 0; time_sum = 0; r_err_sum = 0; t_err_sum = 0;
    }
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

Matrix3d quat2rot(double qw, double qx, double qy, double qz) {
    Quaterniond q(qw, qx, qy, qz);
    q.normalize();
    return q.toRotationMatrix();
}


bool read_colmap(const string& base_path,
                 map<int, Point3D>& points3d,
                 map<int, Camera>& cameras,
                 vector<Image>& images) {
    
    cout << "Loading COLMAP data from: " << base_path << endl;

    ifstream f_pts(base_path + "/points3D.txt");
    if (!f_pts.is_open()) return false;
    string line;
    while (getline(f_pts, line)) {
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        Point3D p;
        int r, g, b;
        ss >> p.id >> p.xyz(0) >> p.xyz(1) >> p.xyz(2) >> r >> g >> b >> p.error;
        if (p.error <= FILTER_ERR_THRESH) points3d[p.id] = p;
    }
    
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
            cam.K(0,0) = p1; cam.K(1,1) = p2; cam.K(0,2) = p3; cam.K(1,2) = p4;
        } else {
            cam.K(0,0) = p1; cam.K(1,1) = p1; cam.K(0,2) = p2; cam.K(1,2) = p3;
        }
        cameras[cam.id] = cam;
    }

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
        if (img.pts2d.size() >= MIN_INLIERS) images.push_back(img);
    }
    return true;
}

void update_stats(Stats& s, double t_ms, double r_err, double t_err) {
    s.total++;
    s.time_sum += t_ms; 
    if (r_err < THRESH_R && t_err < THRESH_T) {
        s.success++;
        s.r_err_sum += r_err;
        s.t_err_sum += t_err;
    }
}


void run_benchmark_on_sparse(
    const vector<Image>& images, 
    map<int, Point3D>& points3d, 
    map<int, Camera>& cameras,
    int target_N) 
{

    Stats s_bapnp   = {"BAPnP"};
    Stats s_epnp    = {"EPnP"};
    Stats s_epnp_lm = {"EPnP+LM"};
    Stats s_sqpnp   = {"SQPnP"};
    Stats s_cpnp    = {"CPnP"};

    cout << "------------------------------------------------------------" << endl;
    cout << "Running Sparse Benchmark with N = " << target_N << " points per frame" << endl;


    std::mt19937 rng(12345); 

    int valid_frames = 0;

    for (const auto& img : images) {
        int total_points = img.pts2d.size();
        
        if (total_points < target_N) continue;
        
        valid_frames++;
        Matrix3d K = cameras[img.camera_id].K;


        vector<int> indices(total_points);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng); 


        vector<int> selected_indices(indices.begin(), indices.begin() + target_N);


        int N = target_N;
        MatrixXd P_world(3, N);
        MatrixXd pts_norm(3, N);
        
        vector<cv::Point3d> cv_P3D;
        vector<cv::Point2d> cv_P2D;
        
        vector<Vector3d> cpnp_P3D;
        vector<Vector2d> cpnp_P2D_norm; 

        for (int i = 0; i < N; ++i) {
            int idx = selected_indices[i]; 
            
            Vector3d Xw = points3d[img.p3d_ids[idx]].xyz;
            Vector2d uv = img.pts2d[idx];

            // BAPnP
            P_world.col(i) = Xw;
            Vector3d uv_h(uv(0), uv(1), 1.0);
            Vector3d yn = K.inverse() * uv_h; 
            pts_norm.col(i) = yn;

            // OpenCV
            cv_P3D.emplace_back(Xw(0), Xw(1), Xw(2));
            cv_P2D.emplace_back(uv(0), uv(1));

            // CPnP
            cpnp_P3D.emplace_back(Xw);
            cpnp_P2D_norm.emplace_back(yn(0), yn(1));
        }

        // --- A. BAPnP ---
        Matrix3d R_est; Vector3d t_est;
        double t_bapnp = measure_median_time([&]() {
            BAPnP::solve(pts_norm, P_world, R_est, t_est);
        });
        double err_r_b = (AngleAxisd(img.R_gt.transpose() * R_est).angle()) * 180.0 / M_PI;
        double err_t_b = (t_est - img.t_gt).norm();
        update_stats(s_bapnp, t_bapnp, err_r_b, err_t_b);

        // --- B. EPnP ---
        cv::Mat cv_K(3,3,CV_64F), rvec, tvec, dist=cv::Mat::zeros(4,1,CV_64F);
        cv::eigen2cv(K, cv_K); 
        double t_epnp_lin = measure_median_time([&]() {
            cv::solvePnP(cv_P3D, cv_P2D, cv_K, dist, rvec, tvec, false, cv::SOLVEPNP_EPNP);
        });

        cv::Mat R_cv; cv::Rodrigues(rvec, R_cv);
        Matrix3d R_epnp; cv::cv2eigen(R_cv, R_epnp);
        Vector3d t_epnp(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
        double err_r_e = (AngleAxisd(img.R_gt.transpose() * R_epnp).angle()) * 180.0 / M_PI;
        double err_t_e = (t_epnp - img.t_gt).norm();
        update_stats(s_epnp, t_epnp_lin, err_r_e, err_t_e);

        // --- C. EPnP + LM ---
        cv::Mat rvec_ref = rvec.clone();
        cv::Mat tvec_ref = tvec.clone();

        double t_refine = measure_median_time([&]() {
            cv::Mat r_tmp = rvec.clone(); 
            cv::Mat t_tmp = tvec.clone();
            cv::solvePnPRefineLM(cv_P3D, cv_P2D, cv_K, dist, r_tmp, t_tmp);
        });
       
        cv::solvePnPRefineLM(cv_P3D, cv_P2D, cv_K, dist, rvec_ref, tvec_ref);
        cv::Mat R_cv_ref; cv::Rodrigues(rvec_ref, R_cv_ref);
        Matrix3d R_epnp_lm; cv::cv2eigen(R_cv_ref, R_epnp_lm);
        Vector3d t_epnp_lm(tvec_ref.at<double>(0), tvec_ref.at<double>(1), tvec_ref.at<double>(2));
        double err_r_elm = (AngleAxisd(img.R_gt.transpose() * R_epnp_lm).angle()) * 180.0 / M_PI;
        double err_t_elm = (t_epnp_lm - img.t_gt).norm();
        update_stats(s_epnp_lm, t_epnp_lin + t_refine, err_r_elm, err_t_elm);

        // --- D. CPnP ---
        {
            std::vector<double> params = {1.0, 1.0, 0.0, 0.0};
            Eigen::Vector4d q_out, q_gn;
            Eigen::Vector3d t_out, t_gn;
            double time_cpnp = measure_median_time([&]() {
                pnpsolver::CPnP(cpnp_P2D_norm, cpnp_P3D, params, q_out, t_out, q_gn, t_gn);
            });
            Quaterniond q_c(q_gn(0), q_gn(1), q_gn(2), q_gn(3));
            q_c.normalize();
            Matrix3d R_cpnp = q_c.toRotationMatrix();
            Vector3d t_cpnp = t_gn;
            double err_r_c = (AngleAxisd(img.R_gt.transpose() * R_cpnp).angle()) * 180.0 / M_PI;
            double err_t_c = (t_cpnp - img.t_gt).norm();
            update_stats(s_cpnp, time_cpnp, err_r_c, err_t_c);
        }

        // --- E. SQPnP ---
        #if CV_VERSION_MAJOR >= 4 && (CV_VERSION_MINOR > 5 || (CV_VERSION_MINOR == 5 && CV_VERSION_REVISION >= 3))
        {
            cv::Mat rvec_sq, tvec_sq;
            double t_sqpnp = measure_median_time([&]() {
                cv::solvePnP(cv_P3D, cv_P2D, cv_K, dist, rvec_sq, tvec_sq, false, cv::SOLVEPNP_SQPNP);
            });
            cv::Mat R_cv_sq; cv::Rodrigues(rvec_sq, R_cv_sq);
            Matrix3d R_sq; cv::cv2eigen(R_cv_sq, R_sq);
            Vector3d t_sq(tvec_sq.at<double>(0), tvec_sq.at<double>(1), tvec_sq.at<double>(2));
            double err_r_s = (AngleAxisd(img.R_gt.transpose() * R_sq).angle()) * 180.0 / M_PI;
            double err_t_s = (t_sq - img.t_gt).norm();
            update_stats(s_sqpnp, t_sqpnp, err_r_s, err_t_s);
        }
        #endif
        
        if (valid_frames % 20 == 0) cout << "\rProcessing frame " << valid_frames << "..." << flush;
    }
    cout << "\nFinished." << endl;

    
    auto print = [&](Stats s) {
        if (s.total == 0) return;
        double succ_rate = 100.0 * s.success / s.total;
        double mean_r = s.success > 0 ? s.r_err_sum / s.success : -1;
        double mean_t = s.success > 0 ? s.t_err_sum / s.success : -1;
        cout << left << setw(10) << s.name
        << " | Succ: " << fixed << setprecision(1) << setw(5) << succ_rate << "%"
        << " | Time: " << setprecision(4) << setw(7) << s.time_sum/s.total << " ms"
        << " | Err: " << setprecision(3) << mean_r << " deg / " << mean_t << " m" << endl;
    };
    
    cout << "\n>>> Results for N = " << target_N << " <<<" << endl;
    print(s_bapnp);
    print(s_epnp);   
    print(s_epnp_lm); 
    print(s_cpnp);
    print(s_sqpnp);
    cout << endl;
}

int main(int argc, char** argv) {
    string dataset_path = "/home/luo/projects/MyBAPnP/south-building";
    if (argc > 1) dataset_path = argv[1];

    map<int, Point3D> points3d;
    map<int, Camera> cameras;
    vector<Image> images;

    if (!read_colmap(dataset_path, points3d, cameras, images)) {
        cerr << "Failed to read COLMAP data." << endl;
        return -1;
    }

    cout << "\n==============================================" << endl;
    cout << "  Starting SPARSE Benchmark on South Building" << endl;
    cout << "==============================================" << endl;


    vector<int> test_Ns = {20, 50, 100};

    for (int N : test_Ns) {
        run_benchmark_on_sparse(images, points3d, cameras, N);
    }

    return 0;
}
