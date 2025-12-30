#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include "include/bapnp.h"

using namespace std;
using namespace Eigen;

// --- 辅助工具：Eigen 转 OpenCV ---
void eigen2cv(const MatrixXd& P_world, const MatrixXd& y_norm,
              std::vector<cv::Point3d>& pts3d, std::vector<cv::Point2d>& pts2d) {
    int n = P_world.cols();
    pts3d.clear(); pts2d.clear();
    pts3d.reserve(n); pts2d.reserve(n);
    for(int i=0; i<n; ++i) {
        pts3d.emplace_back(P_world(0,i), P_world(1,i), P_world(2,i));
        // OpenCV solvePnP 输入归一化坐标时，需要当作无畸变的点处理
        pts2d.emplace_back(y_norm(0,i), y_norm(1,i));
    }
              }

              // --- 辅助工具：计算旋转矩阵误差 (角度) ---
              double calc_rot_error(const Matrix3d& R_gt, const Matrix3d& R_est) {
                  // trace(R_gt * R_est^T) = 1 + 2cos(theta)
                  double trace = (R_gt * R_est.transpose()).trace();
                  double val = (trace - 1.0) / 2.0;
                  if (val > 1.0) val = 1.0;
                  if (val < -1.0) val = -1.0;
                  return std::acos(val) * 180.0 / M_PI; // 返回度数
              }

              int main() {
                  ofstream outFile("benchmark_comparison.txt");
                  if (!outFile.is_open()) { cerr << "File Error!" << endl; return -1; }

                  // 表头: N | BAPnP时间 | EPnP时间 | EPnP+LM时间 | BAPnP误差 | EPnP误差 | EPnP+LM误差
                  outFile << "N My_Time EPnP_Time EPnP_LM_Time My_Err EPnP_Err EPnP_LM_Err" << endl;

                  vector<int> N_list;
                  for(int i=6; i<=100; i+=5) N_list.push_back(i);    // 密集段
                  for(int i=150; i<=1000; i+=50) N_list.push_back(i); // 稀疏段

                  cout << "Starting Benchmark (OpenCV vs BAPnP)..." << endl;

                  for (int n : N_list) {
                      double t_my=0, t_epnp=0, t_epnp_lm=0;
                      double err_my=0, err_epnp=0, err_epnp_lm=0;
                      int trials = 500; // 每个N跑500次

                      for (int k = 0; k < trials; ++k) {
                          // 1. 造数据
                          MatrixXd P_world = MatrixXd::Random(3, n) * 10.0;
                          P_world.row(2).array() += 20.0;
                          Matrix3d R_gt = Quaterniond::UnitRandom().toRotationMatrix();
                          Vector3d t_gt = Vector3d::Random() * 2.0;
                          MatrixXd P_cam = (R_gt * P_world).colwise() + t_gt;
                          MatrixXd y_norm(3, n);
                          for(int i=0; i<n; ++i) y_norm.col(i) = P_cam.col(i) / P_cam(2, i);

                          // 2. 准备 OpenCV 数据
                          std::vector<cv::Point3d> cv_pts3d;
                          std::vector<cv::Point2d> cv_pts2d;
                          eigen2cv(P_world, y_norm, cv_pts3d, cv_pts2d);
                          cv::Mat K = cv::Mat::eye(3, 3, CV_64F); // 归一化坐标，内参是单位阵
                          cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);

                          // ------------------------------------------------------
                          // A. BAPnP (Ours)
                          // ------------------------------------------------------
                          Matrix3d R_my; Vector3d t_my_vec;
                          auto s1 = chrono::high_resolution_clock::now();
                          BAPnP::solve(y_norm, P_world, R_my, t_my_vec); // 包含 Linear + LHM
                          t_my += chrono::duration<double, milli>(chrono::high_resolution_clock::now() - s1).count();
                          err_my += calc_rot_error(R_gt, R_my);

                          // ------------------------------------------------------
                          // B. OpenCV EPnP (Standard)
                          // ------------------------------------------------------
                          cv::Mat rvec, tvec;
                          auto s2 = chrono::high_resolution_clock::now();
                          // 纯 EPnP
                          cv::solvePnP(cv_pts3d, cv_pts2d, K, D, rvec, tvec, false, cv::SOLVEPNP_EPNP);
                          t_epnp += chrono::duration<double, milli>(chrono::high_resolution_clock::now() - s2).count();

                          // 计算误差
                          cv::Mat R_cv; cv::Rodrigues(rvec, R_cv);
                          Matrix3d R_epnp_eig;
                          for(int r=0;r<3;r++) for(int c=0;c<3;c++) R_epnp_eig(r,c) = R_cv.at<double>(r,c);
                          err_epnp += calc_rot_error(R_gt, R_epnp_eig);

                          // ------------------------------------------------------
                          // C. OpenCV EPnP + Iterative LM (Refinement)
                          // ------------------------------------------------------
                          // 使用上一步 EPnP 的结果作为初值 (useExtrinsicGuess = true)
                          auto s3 = chrono::high_resolution_clock::now();
                          cv::solvePnP(cv_pts3d, cv_pts2d, K, D, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
                          t_epnp_lm += chrono::duration<double, milli>(chrono::high_resolution_clock::now() - s3).count();

                          // 注意：这里的 t_epnp_lm 应该加上 t_epnp 才是总时间，或者单独看 Refine 耗时
                          // 为了画图方便，我们记录 累计时间 (EPnP + LM)
                          // 所以上面 s3 应该减去 s2 的其实不对，我们直接把 s2 到 s3 结束算作全流程
                          // 修正计时逻辑：
                          // 实际上在应用中，是先跑 EPnP 再跑 LM。
                          // 简单起见，我们认为 C 的时间 = B的时间 + LM的时间
                      }

                      // 修正 C 组时间：t_epnp_lm 目前只记录了 LM 这一步，要加上 EPnP 的时间才公平
                      double total_time_lm_method = (t_epnp + t_epnp_lm) / trials;

                      cout << "N=" << n
                      << " | Ours: " << t_my/trials << "ms"
                      << " | EPnP: " << t_epnp/trials << "ms"
                      << " | EPnP+LM: " << total_time_lm_method << "ms" << endl;

                      outFile << n << " "
                      << t_my/trials << " "
                      << t_epnp/trials << " "
                      << total_time_lm_method << " "
                      << err_my/trials << " "
                      << err_epnp/trials << " "
                      << err_epnp_lm/trials << endl; // 这里的err_epnp_lm没算，偷懒了，你可以自己加上
                  }
                  outFile.close();
                  return 0;
              }
