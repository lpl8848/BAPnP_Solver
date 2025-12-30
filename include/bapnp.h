#ifndef BAPNP_H
#define BAPNP_H

#include <Eigen/Dense>
#include <vector>

class BAPnP {
public:
    // 为了方便外部使用，定义一些常用类型
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;

    /**
     * @brief BAPnP 主求解接口
     * @param y_norm   2xN (x,y) 或 3xN (x,y,1) 的归一化相机坐标
     * @param P_world  3xN 的世界坐标系下的 3D 点
     * @param R        输出：旋转矩阵 (3x3)
     * @param t        输出：平移向量 (3x1)
     */
    static void solve(const Eigen::MatrixXd& y_norm, 
                      const Eigen::MatrixXd& P_world, 
                      Eigen::Matrix3d& R, 
                      Eigen::Vector3d& t);

private:
    // 禁止实例化，因为这只是一个静态工具类
    BAPnP() = delete;

    // Stage I-III: 线性求解器 (Linear Initialization)
    static void linear_solver(const Eigen::MatrixXd& y_norm, 
                              const Eigen::MatrixXd& P_world, 
                              Eigen::Matrix3d& R_out, 
                              Eigen::Vector3d& t_out);

    // Stage IV: 高斯-牛顿迭代优化 (Iterative Refinement)
    static void refine_gn(const Eigen::Matrix3d& R_init, 
                          const Eigen::Vector3d& t_init,
                          const Eigen::MatrixXd& y_norm, 
                          const Eigen::MatrixXd& P_world,
                          Eigen::Matrix3d& R_opt, 
                          Eigen::Vector3d& t_opt);

    // 辅助函数: Procrustes 对齐 (Kabsch Algorithm)
    // 求解 R, t 使得 R * P_src + t ~ P_dst
    static void compute_procrustes(const Eigen::MatrixXd& P_src, 
                                   const Eigen::MatrixXd& P_dst, 
                                   Eigen::Matrix3d& R, 
                                   Eigen::Vector3d& t);
};

#endif // BAPNP_H
