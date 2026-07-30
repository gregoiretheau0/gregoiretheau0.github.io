---
layout: post
title: "Perspective-N-Points (PnP) for Vision-Based Landing (VBL)"
date: 2026-07-29
use_math: true
---

In safety-critical aeronautical applications, vision-based navigation serves as a crucial redundant positioning system when primary signals such as GPS or Instrument Landing Systems (ILS) are degraded or unavailable. A common Vision-Based Landing (VBL) architecture relies on a hybrid pipeline: a deep neural network detects 2D keypoints of a runway, and a geometric Perspective-n-Point (PnP) solver computes the 6-DoF position and attitude of the aircraft relative to the touchdown target.

This post breaks down the mathematics behind PnP algorithms, details their integration into an end-to-end VBL pipeline, and presents a quantitative benchmark evaluating five major PnP formulations across 1,000 synthetic flight approach scenarios from the LARDv2 dataset.

---

## The Vision-Based Landing (VBL) Architecture

The objective of an end-to-end VBL pipeline is to estimate the 3D aircraft attitude and translation vector $y_{3D} \in \mathbb{R}^3$ relative to the runway coordinate system from a single image.

![End-to-End VBL Pipeline Architecture](/assets/images/posts/pnp-vbl/vbl_pipeline.png)

The system operates in three main functional stages:

1. **2D Landmark Detection**: A keypoint regression network (such as YOLOv8-Pose) processes the input image and isolates the 2D pixel coordinates $\hat{k} \in \mathbb{R}^{4 \times 2}$ corresponding to the key features of the runway (typically the four runway corners).
2. **Candidate Selection**: A filtering step retains the highest-confidence candidate detection above a operational threshold.
3. **Geometric Pose Estimation**: A Perspective-n-Point ($f_{PnP}$) algorithm combines the regressed 2D keypoints with known 3D runway coordinates $K_{3D} \in \mathbb{R}^{4 \times 3}$ to derive the final 6-DoF aircraft pose.

---

## Mathematical Formulation of Perspective-n-Point (PnP)

The Perspective-n-Point problem consists of determining the relative rotation $\mathbf{R} \in \mathrm{SO}(3)$ and translation $\mathbf{t} \in \mathbb{R}^3$ between a 3D object coordinate frame and a camera coordinate frame, given $N$ corresponding 3D-to-2D point pairs.

![PnP Coordinate Transformation Geometry](/assets/images/posts/pnp-vbl/pnp_geometry.png)

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

* **Iterative PnP (ITER)**: A non-linear optimization approach based on the Levenberg-Marquardt algorithm. It minimizes reprojection error iteratively starting from an initial pose guess.
* **P3P (Perspective-3-Point)**: A closed-form solver (Gao / Kneip formulation) utilizing three point correspondences to derive up to four potential solutions, disambiguated using a fourth point.
* **IPPE (Infinitesimal Plane-Based Pose Estimation)**: A non-iterative solver designed specifically for planar targets (such as runway thresholds). It solves the pose directly from homography decomposition without iterative local minima risks.
* **SQPNP (Sequential Quadratic Programming PnP)**: A solver that reformulates PnP non-linear least squares as a sequential quadratic program, handling coplanar and non-coplanar points consistently.
* **BPnP (Backpropagatable PnP)**: A differentiable PnP formulation using the Implicit Function Theorem. While primarily designed for end-to-end trainable networks and backpropagation attacks, it acts as a baseline surrogate for non-differentiable solvers.

---

## Experimental Setup & Implementation

The benchmark script evaluates solver resilience under two conditions:
1. **Ground Truth (GT)**: Evaluating numerical solver accuracy using exact ground-truth 2D annotations.
2. **YOLO Predictions (PR)**: Testing performance under real keypoint regression noise from a YOLOv8-Pose model trained on the LARDv2 dataset.

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

The quantitative comparison highlights clear structural differences in how geometric solvers handle planar keypoint configurations and neural network noise.

![Relative Translation Error Distribution Across 5 Solvers](/assets/images/posts/pnp-vbl/pnp_benchmark_all.png)
### Performance Under Ground-Truth Keypoints

When provided with exact keypoints, solvers exhibit distinct geometric characteristics:

* **ITER, IPPE, SQPNP, and BPnP** demonstrate high numerical precision, achieving a median relative translation error of **~2.2%** (corresponding to approximately **69.6 meters** of translation error at multi-kilometer approach distances).
* **P3P** displays significantly higher variance and median error (**~8.8%**). This behavior stems from the geometric ambiguity of planar configurations when selecting subsets of 3 points from planar rectangle corners.

### Performance Under Deep Learning Keypoint Noise (YOLOv8-Pose)

When incorporating real keypoint estimation noise from YOLOv8-Pose:

* **Iterative PnP (ITER)** proves to be the most robust general solver, maintaining a median error of **5.9% (201.3 m)** and a Mean Absolute Error (MAE) of **14.6% (368.2 m)**.
* **IPPE** and **SQPNP** offer near-identical robustness to ITER, benefiting from optimization formulations tailored to planar targets.
* **P3P** degrades significantly under prediction noise, yielding a median error of **15.8%** with extreme outliers exceeding 60% relative error.

---

## Iterative PnP vs. Differentiable BPnP Deep-Dive

In research contexts involving gradient-based validation or end-to-end training, substituting non-differentiable solvers with differentiable counterparts like BPnP is common practice. We evaluated whether BPnP provides an accurate surrogate for OpenCV's standard `SOLVEPNP_ITERATIVE`.

![OpenCV Iterative PnP vs Differentiable BPnP Comparison](/assets/images/posts/pnp-vbl/iter_vs_bpnp.png)

### Comparative Findings

* **Exact Equivalence on Nominal Input**: On ground-truth keypoints, BPnP matches Iterative PnP perfectly, recording identical median errors (**2.2%** vs **2.2%**) and close MAE bounds (**2.9%** vs **3.2%**).
* **Divergence Under Large Prediction Outliers**: Under noisy YOLO predictions, while the median performance remains close (**5.9%** for ITER vs **6.8%** for BPnP), BPnP exhibits higher sensitivity to large keypoint perturbations, increasing its MAE to **58.6%** compared to **14.6%** for OpenCV ITER.

---

## Key Takeaways for Avionics Perception Design

* **Prefer IPPE or ITER for Planar Keypoints**: For 4-point planar target configurations (such as runway corners), **IPPE** and **Iterative PnP** provide optimal numerical stability and noise tolerance.
* **Avoid P3P for Planar Targets**: P3P formulations should be avoided when keypoints are strictly coplanar due to geometric disambiguation ambiguity.
* **Use BPnP as a Differentiable Surrogate**: BPnP acts as a valid, differentiable proxy for iterative algorithms during gradient-based security evaluation (such as APGD adversarial validation), provided high keypoint outlier filtering is maintained.

