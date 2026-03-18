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

#include "bapnp.h"
#include "cpnp.h"
#include "epnp.h"
#include "sqpnp.h"

using namespace std;
using namespace Eigen;


const double FILTER_ERR_THRESH = 1.0; 
const int MIN_INLIERS = 6;            
const double THRESH_R = 2.0;          
const double THRESH_T = 1.0;          
const int BENCHMARK_ITERATIONS = 1000; 

struct Camera { int id; int width, height; Matrix3d K; };
struct Point3D { int id; Vector3d xyz; double error; };
struct Image {
    int id; string name;
    Matrix3d R_gt; Vector3d t_gt;
    int camera_id;
    vector<Vector2d> pts2d;   
    vector<int> p3d_ids;      
};
struct Stats {
    string name;
    int success = 0; int total = 0;
    double time_sum = 0; double r_err_sum = 0; double t_err_sum = 0;
    void reset() { success = 0; total = 0; time_sum = 0; r_err_sum = 0; t_err_sum = 0; }
};


double calc_rot_err(const Matrix3d& R1, const Matrix3d& R2) {
    double tr = (R1 * R2.transpose()).trace();
    double val = (tr - 1.0) / 2.0;
    if (val > 1.0) val = 1.0; else if (val < -1.0) val = -1.0;
    return acos(val) * 180.0 / M_PI;
}

double calc_trans_err(const Vector3d& t1, const Vector3d& t2) {
    return (t1 - t2).norm();
}

template <typename Func>
double measure_median_time(Func func) {
    std::vector<double> times;
    times.reserve(BENCHMARK_ITERATIONS);
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        auto t1 = chrono::high_resolution_clock::now();
        func(); 
        auto t2 = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, std::milli>(t2 - t1).count());
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

bool read_colmap(const string& base_path, map<int, Point3D>& points3d, map<int, Camera>& cameras, vector<Image>& images) {
    ifstream f_pts(base_path + "/points3D.txt");
    if (!f_pts.is_open()) return false;
    string line;
    while (getline(f_pts, line)) {
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        Point3D p; int r, g, b;
        ss >> p.id >> p.xyz(0) >> p.xyz(1) >> p.xyz(2) >> r >> g >> b >> p.error;
        if (p.error <= FILTER_ERR_THRESH) points3d[p.id] = p;
    }
    
    ifstream f_cam(base_path + "/cameras.txt");
    if (!f_cam.is_open()) return false;
    while (getline(f_cam, line)) {
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        Camera cam; string model;
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
        double x, y; int p3d_id;
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

void run_benchmark_on_sparse(const vector<Image>& images, map<int, Point3D>& points3d, map<int, Camera>& cameras, int target_N) {
    Stats s_bapnp   = {"BAPnP"};
    Stats s_epnp    = {"EPnP(Raw)"};
    Stats s_sqpnp   = {"SQPnP"};
    Stats s_cpnp    = {"CPnP"};

    cout << "------------------------------------------------------------" << endl;
    cout << "Running Sparse Benchmark with N = " << target_N << " (NEAR-COPLANAR MODE)" << endl;

    std::mt19937 rng(12345); 
    int valid_frames = 0;

    for (const auto& img : images) {
        int total_points = img.pts2d.size();
        if (total_points < target_N) continue;
        
        Matrix3d K = cameras[img.camera_id].K;
        Matrix3d K_inv = K.inverse(); 
        double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);


        vector<Vector3d> frame_pts3d(total_points);
        for (int i = 0; i < total_points; ++i) {
            frame_pts3d[i] = points3d[img.p3d_ids[i]].xyz;
        }


        int ransac_iters = 50;
        std::uniform_int_distribution<int> dist(0, total_points - 1);
        Vector3d best_normal = Vector3d::UnitZ();
        Vector3d best_p0 = Vector3d::Zero();
        int max_inliers = 0;
        double plane_thresh = 0.5; 

        for (int it = 0; it < ransac_iters; ++it) {
            int i1 = dist(rng), i2 = dist(rng), i3 = dist(rng);
            if (i1 == i2 || i1 == i3 || i2 == i3) continue;

            Vector3d p1 = frame_pts3d[i1];
            Vector3d p2 = frame_pts3d[i2];
            Vector3d p3 = frame_pts3d[i3];

            Vector3d normal = (p2 - p1).cross(p3 - p1);
            if (normal.norm() < 1e-6) continue;
            normal.normalize();

            int inliers = 0;
            for (const auto& pt : frame_pts3d) {
                if (std::abs(normal.dot(pt - p1)) < plane_thresh) inliers++;
            }

            if (inliers > max_inliers) {
                max_inliers = inliers;
                best_normal = normal;
                best_p0 = p1;
            }
        }


        vector<pair<double, int>> dist_idx_pairs;
        dist_idx_pairs.reserve(total_points);
        for (int i = 0; i < total_points; ++i) {
            double d = std::abs(best_normal.dot(frame_pts3d[i] - best_p0));
            dist_idx_pairs.push_back({d, i});
        }
        std::sort(dist_idx_pairs.begin(), dist_idx_pairs.end());


        int pool_size = std::min(total_points, static_cast<int>(target_N * 1.5));
        if (pool_size < target_N) continue; 

        vector<int> coplanar_pool;
        coplanar_pool.reserve(pool_size);
        for (int i = 0; i < pool_size; ++i) {
            coplanar_pool.push_back(dist_idx_pairs[i].second);
        }

        std::shuffle(coplanar_pool.begin(), coplanar_pool.end(), rng);
        vector<int> selected_indices(coplanar_pool.begin(), coplanar_pool.begin() + target_N);
        // ====================================================

        valid_frames++;

        int N = target_N;
        vector<Vector3d> raw_points3d(N);
        vector<Vector2d> raw_pixels(N);
        for (int i = 0; i < N; ++i) {
            int idx = selected_indices[i]; 
            raw_points3d[i] = points3d[img.p3d_ids[idx]].xyz;
            raw_pixels[i]   = img.pts2d[idx];
        }

        // --- A. BAPnP ---
        Matrix3d R_est_bapnp; Vector3d t_est_bapnp;
        double time_bapnp = measure_median_time([&]() {
            MatrixXd P_world(3, N);
            MatrixXd pts_norm(3, N);
            for(int i = 0; i < N; ++i) {
                P_world.col(i) = raw_points3d[i];
                Vector3d uv_h(raw_pixels[i](0), raw_pixels[i](1), 1.0);
                pts_norm.col(i) = K_inv * uv_h; 
            }
            BAPnP::solve(pts_norm, P_world, R_est_bapnp, t_est_bapnp);
        });
        update_stats(s_bapnp, time_bapnp, calc_rot_err(img.R_gt, R_est_bapnp), calc_trans_err(img.t_gt, t_est_bapnp));

        // --- B. EPnP---
        epnp PnP;
        PnP.set_internal_parameters(cx, cy, fx, fy);
        PnP.set_maximum_number_of_correspondences(N);
        Matrix3d R_epnp; Vector3d t_epnp;
        double time_epnp_raw = measure_median_time([&]() {
            PnP.reset_correspondences();
            for(int i = 0; i < N; i++) {
                PnP.add_correspondence(raw_points3d[i](0), raw_points3d[i](1), raw_points3d[i](2), 
                                       raw_pixels[i](0), raw_pixels[i](1));
            }
            double R_est[3][3], t_est[3];
            PnP.compute_pose(R_est, t_est);
            for(int r = 0; r < 3; r++) {
                for(int c = 0; c < 3; c++) R_epnp(r,c) = R_est[r][c];
                t_epnp(r) = t_est[r];
            }
        });
        update_stats(s_epnp, time_epnp_raw, calc_rot_err(img.R_gt, R_epnp), calc_trans_err(img.t_gt, t_epnp));

        // --- C. CPnP ---
        std::vector<double> params = {1.0, 1.0, 0.0, 0.0};
        Matrix3d R_cpnp; Vector3d t_cpnp;
        double time_cpnp = measure_median_time([&]() {
            vector<Vector2d> cpnp_p2d_norm(N);
            for(int i = 0; i < N; ++i) {
                Vector3d uv_h(raw_pixels[i](0), raw_pixels[i](1), 1.0);
                Vector3d norm_pt = K_inv * uv_h;
                cpnp_p2d_norm[i] = Vector2d(norm_pt(0), norm_pt(1));
            }
            Eigen::Vector4d q_out, q_gn; 
            Eigen::Vector3d t_out, t_gn;
            pnpsolver::CPnP(cpnp_p2d_norm, raw_points3d, params, q_out, t_out, q_gn, t_gn);
            
            Quaterniond q_c(q_gn(0), q_gn(1), q_gn(2), q_gn(3)); 
            q_c.normalize();
            R_cpnp = q_c.toRotationMatrix();
            t_cpnp = t_gn;
        });
        update_stats(s_cpnp, time_cpnp, calc_rot_err(img.R_gt, R_cpnp), calc_trans_err(img.t_gt, t_cpnp));

        // --- D. SQPnP---
        Matrix3d R_sq = Matrix3d::Identity(); Vector3d t_sq = Vector3d::Zero();
        double time_sqpnp = measure_median_time([&]() {
            vector<Vector2d> sq_p2d_norm(N);
            for(int i = 0; i < N; ++i) {
                Vector3d uv_h(raw_pixels[i](0), raw_pixels[i](1), 1.0);
                Vector3d norm_pt = K_inv * uv_h;
                sq_p2d_norm[i] = Vector2d(norm_pt(0), norm_pt(1));
            }
            sqpnp::PnPSolver solver(raw_points3d, sq_p2d_norm);
            if (solver.IsValid() && solver.Solve() && solver.NumberOfSolutions() > 0) {
                const auto* sol = solver.SolutionPtr(0);
                R_sq = Map<const Matrix<double, 3, 3, RowMajor>>(sol->r_hat.data());
                t_sq = sol->t;
            }
        });
        update_stats(s_sqpnp, time_sqpnp, calc_rot_err(img.R_gt, R_sq), calc_trans_err(img.t_gt, t_sq));
        
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

    vector<int> test_Ns = {20, 50, 100, 200};
    for (int N : test_Ns) {
        run_benchmark_on_sparse(images, points3d, cameras, N);
    }

    return 0;
}
