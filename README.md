# BAPnP: A Robust Linear Initialization for PnP Based on Barycentric Affine Invariance

This repository contains the MATLAB simulations and C++ implementation for the paper: **"BAPnP: A Robust Linear Initialization for PnP Based on Barycentric Affine Invariance"**.

BAPnP is an efficient $O(n)$ solver that leverages geometry-guided base selection to maximize the reference basis volume, providing a reliable initialization for Gauss-Newton refinement, especially in quasi-planar configurations.

## 1. MATLAB Simulations

The MATLAB code is located in the `simulations/` directory. It includes the algorithm implementation, ablation studies, and comparisons with state-of-the-art methods.

### 1.1 Prerequisites & Setup

To run the comparisons, you need to download the baseline algorithms and add them to your MATLAB path:

1. **MLPnP and other algorithms**: Download from [urbste/MLPnP_matlab_toolbox](https://github.com/urbste/MLPnP_matlab_toolbox).
2. **CPnP**: Download from [LIAS-CUHKSZ/CPnP-A-Consistent-PnP-Solver](https://github.com/LIAS-CUHKSZ/CPnP-A-Consistent-PnP-Solver).
3. **SRPnP**: Download from (https://github.com/pingwangsky/PnP_tool)

**Setup:**
Unzip these toolboxes and add their folders (and subfolders) to your MATLAB working path before running the experiments.

### 1.2 Core Algorithms

We provide unified interfaces for different solvers:
* `BAPnP.m`: Our proposed method (Linear Initialization + Gauss-Newton Refinement).
* `pnp_linear_only.m`: Our proposed method (Linear Initialization only).
* `run_cpnp.m`: Wrapper for the CPnP solver.
* *(Other wrappers included in the folder)*

### 1.3 Reproducing Paper Figures

Use the following scripts to reproduce the figures presented in the paper:

| Figure in Paper | Description | MATLAB Script |
| :--- | :--- | :--- |
| **Fig. 1** | Geometric Comparison (Tetrahedron Volume) | `Tetrahedron.m` |
| **Fig. 2** | Ablation Studies| `Ablation1.m`, `Ablation2.m` |
| **Fig. 3** | Robustness to Image Noise | `exp1.m` |
| **Fig. 4** | Robustness to Point Density | `exp2.m` |
| **Fig. 5** | Computational Efficiency Plot | `plot_time.m` |
| **Fig. 6** | Quasi-Planar Stability & Spectral Gap Analysis | `test_spectral_gap_comparison.m`<br>`test_pnp_planarity_performance.m` |

---

## 2. C++ Implementation

The C++ source code is located in the `src/` directory. It is designed for real-time performance evaluation and benchmark datasets.

### 2.1 Source Files

* `src/bapnp.cpp`: The C++ implementation of the BAPnP algorithm.
* `main_benchmark.cpp`: Runtime comparison against OpenCV implementations.
* `main_colmap_benchmark.cpp`: Evaluation on the **South Building Dataset**.
* `main_tum.cpp`: Evaluation on the **TUM RGB-D Dataset**.

### 2.2 Dataset Preparation

Before running the real-world benchmarks, please download the required datasets. Due to size constraints, they are not included in this repository.

1. **South Building Dataset**:
   * **Download**: Visit [COLMAP Datasets](https://colmap.github.io/datasets.html) and download "South Building".
   * **Setup**: Extract the dataset and ensure the path matches the configuration in `main_colmap_benchmark.cpp` .

2. **TUM RGB-D Dataset**:
   * **Download**: Visit [TUM RGB-D Benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download#freiburg1_desk).
   * **Sequence**: We use the `freiburg1_desk` sequence for evaluation.
   * **Setup**: Download the sequence and place it in the working directory or update the path in `main_tum.cpp`.
     
### 2.3 Build and Run

Ensure you have a C++ compiler (supports C++11 or higher) and CMake installed.

**Build:**

```bash
cd build
# Assuming CMake is used to generate the Makefile
cmake .. 
make -j

Runtime Benchmark:
./run_benchmark

TUM RGB-D Experiment:
./run_tum

South Building (COLMAP) Experiment:
./run_colmap_bench
