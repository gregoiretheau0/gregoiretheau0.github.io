---
layout: post
title: "Perspective-N-Points (PnP) for Vision-Based Landing (VBL)"
date: 2026-07-29
use_math: true
---

In safety-critical aeronautical applications, vision-based navigation serves as a crucial redundant positioning system when primary signals such as GPS or Instrument Landing Systems (ILS) are degraded or unavailable. A common Vision-Based Landing (VBL) architecture relies on a hybrid pipeline: a deep neural network detects 2D keypoints of a runway, and a geometric Perspective-n-Point (PnP) solver computes the 6-DoF position and attitude of the aircraft relative to the touchdown target.

This post breaks down the mathematics behind PnP algorithms, details their integration into an end-to-end VBL pipeline, and presents a quantitative benchmark evaluating five major PnP formulations across 1,000 synthetic flight approach scenarios from the LARDv2 dataset [[Ducoffe et al., 2023](#ref-lard)].

---

### Table of Contents
* TOC
{:toc}

---

## The Vision-Based Landing (VBL) Architecture

The objective of an end-to-end VBL pipeline is to estimate the 3D aircraft pose (attitude and translation) relative to the runway coordinate system from a single image.

![End-to-End VBL Pipeline Architecture](/assets/images/posts/pnp-vbl/vbl_pipeline.png)

The system operates in three main functional stages:

1. **2D Landmark Detection**: A keypoint regression network (such as YOLOv8-Pose) processes the input image and isolates the 2D pixel coordinates $\hat{k} \in \mathbb{R}^{4 \times 2}$ corresponding to the key features of the runway (typically the four runway corners).
2. **Candidate Selection**: A filtering step retains the highest-confidence candidate detection above an operational threshold.
3. **Geometric Pose Estimation**: A Perspective-n-Point ($f_{PnP}$) algorithm combines the regressed 2D keypoints with known 3D runway coordinates $K_{3D} \in \mathbb{R}^{4 \times 3}$ to derive the final 6-DoF aircraft pose.

---

## Mathematical Formulation of Perspective-n-Point (PnP)

The Perspective-n-Point problem consists of determining the relative rotation $\mathbf{R} \in \mathrm{SO}(3)$ and translation $\mathbf{t} \in \mathbb{R}^3$ between a 3D object coordinate frame and a camera coordinate frame, given $N$ corresponding 3D-to-2D point pairs.

![PnP Coordinate Transformation Geometry](/assets/images/posts/pnp-vbl/2dto3d.png)

Using the pinhole camera model, a 3D point $\mathbf{X}_w = [X_w, Y_w, Z_w, 1]^\top$ expressed in the world coordinate system projects to image pixel coordinates $\mathbf{u} = [u, v, 1]^\top$ according to:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

where $s$ is an arbitrary scaling factor and $\mathbf{K}$ represents the internal calibration matrix of the camera defined by focal lengths $(f_x, f_y)$ and principal point $(c_x, c_y)$:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

Expanding the extrinsic matrix $[\mathbf{R} \mid \mathbf{t}]$, the transformation mapping world coordinates to normalized image coordinates yields:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

Solving for $\mathbf{R}$ and $\mathbf{t}$ requires minimizing the reprojection error between the observed keypoint projections $\hat{\mathbf{u}}_i$ and projected 3D coordinates:

$$\arg\min_{\mathbf{R}, \mathbf{t}} \sum_{i=1}^{N} \left\Vert{} \hat{\mathbf{u}}_i - \pi\left(\mathbf{K} (\mathbf{R} \mathbf{X}_{w,i} + \mathbf{t})\right) \right\Vert{}_2^2$$

where $\pi([X, Y, Z]^\top) = [X/Z, Y/Z]^\top$ denotes the standard perspective projection operator.

---

## Evaluated PnP Solvers

We evaluate five distinct PnP implementations to identify the optimal configuration for real-time runway pose estimation:

* **Iterative PnP (ITER)**: A non-linear optimization approach based on the Levenberg-Marquardt algorithm [[Levenberg, 1944](#ref-levenberg)]. It minimizes reprojection error iteratively starting from an initial pose guess.
* **P3P (Perspective-3-Point)**: A closed-form solver (Gao / Kneip formulation) utilizing three point correspondences to derive up to four potential solutions [[Gao et al., 2003](#ref-p3p)], disambiguated using a fourth point.
* **IPPE (Infinitesimal Plane-Based Pose Estimation)**: A non-iterative solver designed specifically for planar targets [[Collins & Bartoli, 2014](#ref-ippe)] (such as runway thresholds). It solves the pose directly from homography decomposition without iterative local minima risks.
* **SQPNP (Sequential Quadratic Programming PnP)**: A solver that reformulates PnP non-linear least squares as a sequential quadratic program [[Terzakis & Lourakis, 2020](#ref-sqpnp)], handling coplanar and non-coplanar points consistently.
* **BPnP (Backpropagatable PnP)**: A differentiable PnP formulation using the Implicit Function Theorem [[Chen et al., 2020](#ref-bpnp)]. While primarily designed for end-to-end trainable networks and backpropagation attacks, it acts as a baseline surrogate for standard OpenCV implementations that lack PyTorch automatic differentiation bindings.

---

## Experimental Setup & Implementation

The benchmark script evaluates solver resilience under two conditions:
1. **Ground Truth (GT)**: Evaluating numerical solver accuracy using exact ground-truth 2D annotations.
2. **YOLO Predictions (YOLO)**: Testing performance under real keypoint regression noise from a YOLOv8-Pose model trained on the LARDv2 dataset.

The evaluation metric measured is the relative 3D translation error percentage against the true slant distance $d_{true}$:

$$\text{Error}_{\%} = \frac{\vert{} d_{true} - \Vert{} \hat{\mathbf{t}} \Vert{}_2 \vert{}}{d_{true}} \times 100$$

Here is the Python implementation used to benchmark the algorithms across 1,000 images:

```python
import cv2
import numpy as np
import torch
import pandas as pd

def evaluate_all_pnps(obj_pts, img_pts, cam_mat, dist_c, dev):
    """Calculates translation norm across multiple PnP implementations."""
    results = {}
    cv2_solvers = {
        'ITER': cv2.SOLVEPNP_ITERATIVE,
        'P3P': cv2.SOLVEPNP_P3P,
        'IPPE': cv2.SOLVEPNP_IPPE,
        'SQPNP': cv2.SOLVEPNP_SQPNP
    }
    
    # Standard OpenCV Solvers
    for name, flag in cv2_solvers.items():
        success, _, tvec = cv2.solvePnP(obj_pts, img_pts, cam_mat, dist_c, flags=flag)
        results[name] = np.linalg.norm(tvec) if success else None
        
    # Differentiable BPnP Solver
    success_bpnp, tvec_bpnp = solve_pnp_bpnp(obj_pts, img_pts, cam_mat, dev)
    results['BPNP'] = np.linalg.norm(tvec_bpnp) if success_bpnp else None
    
    return results
```

## Benchmark Results & Analysis

The quantitative comparison across 1,000 flight approach scenarios highlights structural differences in how geometric solvers handle planar keypoint configurations and neural network noise.

![Relative Translation Error Distribution Across 5 Solvers](/assets/images/posts/pnp-vbl/pnp_benchmark_all.png)

### 1. Performance Under Ground-Truth Keypoints (GT)

When provided with exact keypoint annotations, solvers exhibit distinct performance baselines::

* **ITER, BPnP, and SQPNP** demonstrate the highest precision, with median errors around **2.2%–2.3%** (**69.6 m** for ITER, **69.9 m** for BPnP, and **73.5 m** for SQPNP) and low Mean Absolute Errors (MAE $\le 3.2\%$).
* **IPPE** achieves strong numerical accuracy with a median error of **3.0% (93.7 m)** and MAE of **3.8% (143.6 m)**.
* **P3P** exhibits significantly higher error and variance (**8.8% / 242.3 m** median, **10.9% / 342.5 m** MAE). This behavior stems from geometric ambiguity when selecting subsets of 3 points from planar rectangular configurations.

### 2. Performance Under Predicted Keypoints (YOLOv8-Pose)

Incorporating keypoint estimation noise from YOLOv8-Pose reveals how solvers how solvers degrade under real-world detection errors:

* **Iterative PnP (ITER)** maintains the lowest median translation error at **5.9% (201.3 m)** with a Mean Absolute Error (MAE) of **14.6% (368.2 m)**.
* **SQPNP** achieves the best overall variance control, yielding a median error of **6.7% (219.2 m)** and the lowest MAE under noise at **9.3% (366.8 m)**.
* **IPPE** offers stable planar performance with a median error of **8.3% (241.5 m)** and an MAE of **10.6% (394.3 m)**.
* **P3P** degrades severely under prediction noise, recording a median error of **16.0% (518.7 m)** and an extreme MAE of **85.0% (927.4 m)**.

### 3. Evaluating BPnP as a Differentiable Evaluation Surrogate

Substituting OpenCV solvers—which lack native automatic differentiation support—with backpropagatable implementations like BPnP is required for gradient-based robustness evaluation (such as APGD adversarial validation [[Croce & Hein, 2020](#ref-apgd)]). Our benchmark evaluates how closely BPnP mirrors standard OpenCV solvers:

* **Nominal Equivalence**: On ground-truth keypoints, BPnP is performs on par with to standard Iterative PnP (**2.2% / 69.9 m** median error vs. **2.2% / 69.6 m** for ITER).
* **Sensitivity to Outliers**: Under noisy predictions, while BPnP retains a median error close to ITER (**6.8% / 238.0 m** vs. **5.9% / 201.3 m**), it exhibits high sensitivity to severe keypoint perturbations. This causes its MAE to spike to **58.6% (592.2 m)** compared to **14.6% (368.2 m)** for ITER.
* **Implication**: BPnP serves as a reliable differentiable proxy for gradient-based evaluation, provided effective outlier filtering or robust loss weighting is applied.

### PnP Solvers Comparison Summary

| Solver | Formulation Type | Coplanar Safe ? | Differentiable Implementation ? | Median Error (GT) | Median Error (YOLO Noise) | Primary Use Case |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **ITER** | Non-linear (Levenberg-Marquardt) | ✅ | ❌ (Standard OpenCV) | 2.2% (69.6 m) | **5.9% (201.3 m)** | General baseline, strong noise tolerance |
| **SQPNP** | Sequential Quadratic Prog. | ✅ | ❌ (Standard OpenCV) | 2.3% (73.5 m) | **6.7% (219.2 m)** | Best overall variance control under noise |
| **IPPE** | Homography Decomposition | ✅ | ❌ (Standard OpenCV) | 3.0% (93.7 m) | **8.3% (241.5 m)** | Fast & direct solver tailored for planar targets |
| **BPnP** | Implicit Function Theorem | ✅ | ✅ (PyTorch Native) | 2.2% (69.9 m) | **6.8% (238.0 m)** | Gradient-based evaluation & end-to-end training |
| **P3P** | Closed-form (3 points) | ❌ | ❌ (Standard OpenCV) | 8.8% (242.3 m) | **16.0% (518.7 m)** | Non-coplanar 3D targets only (Avoid for runways) |

---

> ## Key Takeaways for Avionics Perception Design
>
> * **Prefer SQPNP, ITER, or IPPE for Planar Targets**: For 4-point planar runway configurations, SQPNP, Iterative PnP, and IPPE provide optimal numerical stability and noise tolerance.
> * **Avoid P3P for Planar Runway Keypoints**: P3P formulations suffer from geometric disambiguation issues when keypoints are strictly coplanar.
> * **Use BPnP as a Differentiable Evaluation Surrogate**: BPnP acts as a valid, backpropagatable proxy for standard OpenCV solvers during gradient-based security evaluation (such as APGD adversarial validation), provided keypoint outlier filtering is maintained.

---

## References

### PnP Solvers & Formulations
* <a id="ref-ippe"></a>**IPPE**: Collins, T., & Bartoli, A. (2014). *Infinitesimal Plane-Based Pose Estimation*. International Journal of Computer Vision (IJCV), 109(3), 252-286.
* <a id="ref-sqpnp"></a>**SQPNP**: Terzakis, G., & Lourakis, M. (2020). *A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem*. European Conference on Computer Vision (ECCV).
* <a id="ref-bpnp"></a>**BPnP**: Chen, B., Parra, A., Cao, J., Li, N., & Chin, T. J. (2020). *End-to-End Learnable Geometric Vision by Backpropagating PnP Optimization*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
* <a id="ref-p3p"></a>**P3P**: Gao, X. S., Hou, X. R., Tang, J., & Cheng, H. F. (2003). *Complete solution classification for the perspective-three-point problem*. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 25(8), 930-943.
* <a id="ref-levenberg"></a>**Iterative PnP**: Levenberg, K. (1944). *A method for the solution of certain non-linear problems in least squares*. Quarterly of Applied Mathematics, 2(2), 164-168.

### Datasets & Evaluation Frameworks
* <a id="ref-lard"></a>**LARD Dataset**: Ducoffe, M., Carrere, M., Féliers, L., Gauffriau, A., Mussot, V., Pagetti, C., & Sammour, T. (2023). *LARD: Landing Approach Runway Detection dataset for vision based landing*.
* <a id="ref-apgd"></a>**APGD / Robustness Evaluation**: Croce, F., & Hein, M. (2020). *Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks*. International Conference on Machine Learning (ICML).
