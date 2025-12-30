cmake_minimum_required(VERSION 3.10)
project(BAPnP_Benchmark)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release) # 必须 Release，否则测速不准

# 1. 找包
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# 2. 头文件
include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# 3. 编译所有源文件 (BAPnP + CPnP)
add_executable(run_benchmark
    main_benchmark.cpp
    src/bapnp.cpp
   src/cpnp.cpp  # <--- 加入 CPnP
)

add_executable(run_tum main_tum.cpp src/bapnp.cpp src/cpnp.cpp) # 如果用到 CPnP 也要加上 src/cpnp.cpp



add_executable(run_colmap_bench main_colmap_benchmark.cpp src/bapnp.cpp src/cpnp.cpp)

target_link_libraries(run_colmap_bench ${OpenCV_LIBS} Eigen3::Eigen)

target_link_libraries(run_tum ${OpenCV_LIBS} Eigen3::Eigen)


# 4. 链接
target_link_libraries(run_benchmark
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

