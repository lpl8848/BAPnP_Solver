# BAPnP: A Barycentric Affine Invariant Linear Solver for Robust and Efficient Perspective-n-Point Pose Estimation

[![Manuscript Status](https://img.shields.io/badge/Manuscript-Under%20Review-blue)](https://github.com/lpl8848/BAPnP_Solver)

This repository contains the MATLAB simulations and C++ implementation for the paper:

> **"BAPnP: A Barycentric Affine Invariant Linear Solver for Robust and Efficient Perspective-$n$-Point Pose Estimation"**  
> *Under review at **The Visual Computer** (Springer).*

BAPnP is an efficient $O(n)$ solver that leverages geometry-guided base selection to maximize the reference basis volume, providing a reliable initialization for Gauss-Newton refinement, especially in quasi-planar configurations. It maintains 100% success rate down to strict coplanarity while executing in just $4.4\,\mu s$ at $N=10$ in C++.

---

## 3. Supplementary Materials

`supplementary.pdf` in the repository root archives three reviewer-requested experiments that are not included in the main manuscript:

| Section | Content | Description |
| :--- | :--- | :--- |
| §2 | Localized Basis-Point Corruption | Pure-linear oDLT comparison under targeted basis-point noise (0--20 px) |
| §3 | Minimal-Configuration Analysis | Performance of all 9 PnP methods at $N=4,5,6$ ($\sigma=3$ px) |
| §4 | Extremal Outlier Injection | RANSAC-integrated comparison under 10%--50% synthetic periphery outliers |

The C++ sources for the supplementary experiments (`main_tum_ransac.cpp`, `main_tum_high_ransac.cpp`) and the plotting script (`plot_outlier.py`) are included in this repository.

---

## Citation

If you use this code in your research, please cite the corresponding manuscript:

```bibtex
@article{luo2026bapnp,
  title={BAPnP: A Barycentric Affine Invariant Linear Solver for Robust and Efficient Perspective-n-Point Pose Estimation},
  author={Luo, Peilin and Guo, Yang},
  journal={Under review at The Visual Computer},
  year={2026}
}
```

---

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
* `BAPnP_Coplanar.m`: Our proposed method
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
* `main_benchmark.cpp`: Runtime comparison against state-of-the-art implementations.
* `main_colmap_benchmark.cpp`: Evaluation on the **South Building Dataset**.
* `main_tum.cpp`: Raw evaluation on the **TUM RGB-D Dataset** (without outlier rejection).
* `main_tum_ransac.cpp`: RANSAC-integrated evaluation on the **TUM RGB-D Dataset** (500 iterations, 6-pt sampling, 2 px threshold).
* `main_tum_high_ransac.cpp`: RANSAC evaluation with synthetic extremal outlier injection (10%--50% ratios), testing robustness under adversarial periphery outliers.

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
mkdir build
cd build
cmake .. 
make -j
```

**Runtime Benchmark:**
```bash
./run_benchmark
```

**TUM RGB-D Experiment:**
```bash
./run_tum
```

**TUM RANSAC Experiment:**
```bash
./run_tum_ransac
```

**TUM RANSAC with Extremal Outlier Injection:**
```bash
./run_tum_high_ransac [outlier_ratio]
# e.g., ./run_tum_high_ransac 0.3  for 30% injected outliers
```

**South Building (COLMAP) Experiment:**
```bash
./run_colmap_bench
```
