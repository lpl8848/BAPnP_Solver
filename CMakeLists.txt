cmake_minimum_required(VERSION 3.10)
project(BAPnP_Benchmark)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
add_definitions(-D_USE_MATH_DEFINES)


add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# 寻找依赖项
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

# 1. 基础 Benchmark 目标
add_executable(run_benchmark
    main_benchmark.cpp
    src/bapnp.cpp
    src/cpnp.cpp
    src/epnp.cpp
    src/sqpnp.cpp
)

# 2. TUM 数据集原始评估目标
add_executable(run_tum
    main_tum.cpp
    src/bapnp.cpp
    src/cpnp.cpp
    src/epnp.cpp
    src/sqpnp.cpp
)

# 3. COLMAP Benchmark 目标
add_executable(run_colmap_bench
    main_colmap_benchmark.cpp
    src/bapnp.cpp
    src/cpnp.cpp
    src/epnp.cpp
    src/sqpnp.cpp
)

# 4. TUM 数据集 RANSAC 评估目标
add_executable(run_tum_ransac
    main_tum_ransac.cpp
    src/bapnp.cpp
    src/cpnp.cpp
    src/epnp.cpp
    src/sqpnp.cpp
)

# 5. TUM 数据集高外点 RANSAC 评估目标 (针对 extremal outlier 场景)
add_executable(run_tum_high_ransac
    main_tum_high_ransac.cpp
    src/bapnp.cpp
    src/cpnp.cpp
    src/epnp.cpp
    src/sqpnp.cpp
)

# 设置包含目录
target_include_directories(run_benchmark PRIVATE include)
target_include_directories(run_tum PRIVATE include)
target_include_directories(run_colmap_bench PRIVATE include)
target_include_directories(run_tum_ransac PRIVATE include)
target_include_directories(run_tum_high_ransac PRIVATE include)

# 链接库 (Eigen3 和 OpenCV)
target_link_libraries(run_benchmark PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
target_link_libraries(run_tum PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
target_link_libraries(run_colmap_bench PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
target_link_libraries(run_tum_ransac PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
target_link_libraries(run_tum_high_ransac PRIVATE Eigen3::Eigen ${OpenCV_LIBS})