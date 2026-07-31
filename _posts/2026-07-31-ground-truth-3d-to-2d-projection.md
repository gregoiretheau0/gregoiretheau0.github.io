---
layout: post
title: "Synthetic Data Pipeline: Projecting 3D Ground Truth Keypoints to 2D Image Space"
date: 2026-07-31
use_math: true
---

In vision-based navigation systems, evaluating keypoint regression networks like YOLOv8-Pose requires exact 2D pixel annotations. Flight simulators such as FLSim naturally operate with 3D global positions (ECEF/WGS84) and aircraft attitude (Euler angles).

To automatically generate pixel-accurate ground truth annotations—or to debug keypoint regression before running a Perspective-n-Point (PnP) solver—we must perform the inverse geometric pipeline: projecting 3D runway world coordinates onto the 2D camera image plane.

This post breaks down the mathematics, coordinate frame transformations, and Python implementation required to map 3D global runway coordinates down to 2D image pixels.

---

### Table of Contents
* TOC
{:toc}

---

## The Inverse Geometric Problem

While PnP estimates the 6-DoF aircraft pose from known 2D keypoints, **3D-to-2D projection** solves the forward problem:

Given:
* 3D keypoint coordinates in Earth-Centered, Earth-Fixed (ECEF) space $K_{3D} \in \mathbb{R}^{N \times 3}$.
* Aircraft position $(X_{ac}, Y_{ac}, Z_{ac})_{ECEF}$ or $(\text{lat}, \text{lon}, \text{alt})$.
* Aircraft attitude angles $(\psi, \theta, \phi)$ (Yaw, Pitch, Roll).
* Camera intrinsic specifications (Field of View and resolution).

Compute:
* The projected pixel coordinates $(u, v) \in \mathbb{R}^{N \times 2}$ on the image sensor.

---

## Why Is Simple Projection Not Enough? The Coordinate Frame Challenge

A common pitfall when working with simulator telemetry is passing ECEF coordinates directly into standard pinhole camera equations. Pinhole projection assumes coordinates are expressed in an **optical frame** where $+Z$ points forward along the camera line-of-sight, $+X$ points right, and $+Y$ points downwards.

However, flight simulator telemetry operates across radically different reference frames:

![ECEF vs Local NED Frame](/assets/images/posts/3d-to-2d-projection/coordinate_systems.png)

1. **ECEF Frame**: A global geocentric coordinate frame centered at the Earth's center of mass ($X, Y, Z$).
2. **Local NED Frame (North-East-Down)**: A local tangent plane aligned with local gravity and North ($\text{xNorth}, \text{yEast}, \text{zDown}$).
3. **Aircraft Body Frame**: Axis alignment fixed to the aircraft structure ($+X$ along nose, $+Y$ right wing, $+Z$ belly).
4. **Camera Optical Frame**: Lens-centric coordinate system required by optics equations ($+Z$ depth, $+X$ right, $+Y$ down).

Mapping a 3D point from ECEF to the camera image requires chaining 3 rigid transformations:

![Three-Step Transformation Pipeline](/assets/images/posts/3d-to-2d-projection/6_issue_images.jpg)

---

## Step-by-Step Coordinate Frame Transformations

### Step 1: ECEF to Local NED Frame (Earth Curvature Alignment)

To convert global ECEF coordinates into a local planar frame centered at the aircraft location $(\phi_{lat}, \lambda_{lon})$, we apply the rotation matrix $\mathbf{R}_{ecef \to ned}$:

$$\mathbf{R}_{ecef \to ned} = \begin{bmatrix} -\sin\phi_{lat} \cos\lambda_{lon} & -\sin\phi_{lat} \sin\lambda_{lon} & \cos\phi_{lat} \\ -\sin\lambda_{lon} & \cos\lambda_{lon} & 0 \\ -\cos\phi_{lat} \cos\lambda_{lon} & -\cos\phi_{lat} \sin\lambda_{lon} & -\sin\phi_{lat} \end{bmatrix}$$

Given a 3D point $\mathbf{X}_{ECEF}$ and camera position $\mathbf{C}_{ECEF}$, the vector in local NED coordinates is:

$$\mathbf{X}_{NED} = \mathbf{R}_{ecef \to ned} \cdot (\mathbf{X}_{ECEF} - \mathbf{C}_{ECEF})$$

---

### Step 2: Local NED to Aircraft Body Frame (Attitude Rotation)

The aircraft attitude provided by flight simulators consists of standard aeronautical Euler angles: **Yaw ($\psi$)**, **Pitch ($\theta$)**, and **Roll ($\phi$)**. 

We construct the rotation matrix mapping local NED to the aircraft body frame by composing individual axis rotations:

$$\mathbf{R}_{yaw} = \begin{bmatrix} \cos\psi & \sin\psi & 0 \\ -\sin\psi & \cos\psi & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{R}_{pitch} = \begin{bmatrix} \cos\theta & 0 & -\sin\theta \\ 0 & 1 & 0 \\ \sin\theta & 0 & \cos\theta \end{bmatrix}, \quad \mathbf{R}_{roll} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\phi & \sin\phi \\ 0 & -\sin\phi & \cos\phi \end{bmatrix}$$

$$\mathbf{R}_{ned \to body} = \mathbf{R}_{roll} \cdot \mathbf{R}_{pitch} \cdot \mathbf{R}_{yaw}$$

Applying this rotation tilts the point vector according to the aircraft orientation:

$$\mathbf{X}_{body} = \mathbf{R}_{ned \to body} \cdot \mathbf{X}_{NED}$$

---

### Step 3: Aircraft Body to Camera Optical Frame (Axis Swapping)

In an aircraft body frame, $+X$ points forward, $+Y$ points right, and $+Z$ points down. In standard camera optics, $+Z$ is the optical axis (depth), $+X$ points right, and $+Y$ points down.

We bridge this convention gap by swapping the coordinate axes using a permutation matrix $\mathbf{R}_{body \to cam}$:

$$\mathbf{R}_{body \to cam} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{bmatrix}$$

$$\mathbf{X}_{cam} = \mathbf{R}_{body \to cam} \cdot \mathbf{X}_{body} = \begin{bmatrix} X_{body, Y} \\ X_{body, Z} \\ X_{body, X} \end{bmatrix}$$

Combining these steps yields the total extrinsic transformation mapping global 3D points directly into the optical frame:

$$\mathbf{R}_{ext} = \mathbf{R}_{body \to cam} \cdot \mathbf{R}_{ned \to body} \cdot \mathbf{R}_{ecef \to ned}$$

---

## Analytical Camera Intrinsics Estimation

In simulation environments, physical camera calibration using checkerboard patterns is unnecessary. We derive the intrinsic matrix $\mathbf{K}$ analytically using the simulator's vertical Field of View ($\theta_v$) and image resolution $(W, H)$.

![Analytical Camera Intrinsics Geometry](/assets/images/posts/3d-to-2d-projection/intrinsics_matrix_justification.png)

First, compute the horizontal Field of View ($\theta_h$):

$$\theta_h = 2 \cdot \arctan\left( \tan\left(\frac{\theta_v}{2}\right) \cdot \frac{W}{H} \right)$$

Focal lengths $(f_x, f_y)$ and principal points $(c_x, c_y)$ are then calculated as:

$$f_x = \frac{W}{2 \cdot \tan(\theta_h / 2)}, \quad f_y = \frac{H}{2 \cdot \tan(\theta_v / 2)}$$

$$c_x = \frac{W}{2}, \quad c_y = \frac{H}{2}$$

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

---

## Complete Python Implementation

Here is the complete Python script chaining all coordinate transformations to project 3D ECEF runway threshold points directly onto 2D image pixels:

```python
import numpy as np

def compute_intrinsics(fov_v_deg, width, height):
    """Computes camera intrinsics K analytically from simulator FOV."""
    fov_v = np.radians(fov_v_deg)
    fov_h = 2 * np.arctan(np.tan(fov_v / 2) * (width / height))
    
    fx = width / (2 * np.tan(fov_h / 2))
    fy = height / (2 * np.tan(fov_v / 2))
    cx, cy = width / 2.0, height / 2.0
    
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]])

def ecef_to_ned_matrix(lat_deg, lon_deg):
    """Computes rotation matrix from ECEF to local NED tangent plane."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    s_lat, c_lat = np.sin(lat), np.cos(lat)
    s_lon, c_lon = np.sin(lon), np.cos(lon)
    
    return np.array([
        [-s_lat * c_lon, -s_lat * s_lon,  c_lat],
        [-s_lon,          c_lon,          0    ],
        [-c_lat * c_lon, -c_lat * s_lon, -s_lat]
    ])

def ned_to_body_matrix(yaw_deg, pitch_deg, roll_deg):
    """Computes rotation matrix from NED to Aircraft Body frame."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    
    R_yaw   = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    R_roll  = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
    
    return R_roll @ R_pitch @ R_yaw

def project_3d_to_2d(pts_3d_ecef, ac_ecef, ac_ll, ac_att, K):
    """
    Projects 3D ECEF points onto 2D image coordinates.
    
    :param pts_3d_ecef: (N, 3) array of 3D keypoints in ECEF
    :param ac_ecef: (3,) aircraft ECEF position [X, Y, Z]
    :param ac_ll: (2,) aircraft [lat, lon] in degrees
    :param ac_att: (3,) aircraft attitude [yaw, pitch, roll] in degrees
    :param K: (3, 3) camera intrinsics matrix
    :return: (N, 2) projected 2D pixel coordinates
    """
    R_ecef2ned = ecef_to_ned_matrix(ac_ll[0], ac_ll[1])
    R_ned2body = ned_to_body_matrix(ac_att[0], ac_att[1], ac_att[2])
    R_body2cam = np.array([[0, 1, 0], 
                           [0, 0, 1], 
                           [1, 0, 0]])
    
    # Combined Extrinsic Rotation Matrix
    R_total = R_body2cam @ R_ned2body @ R_ecef2ned
    
    # Transform points to Camera Optical Frame
    pts_trans = pts_3d_ecef - ac_ecef
    pts_cam = (R_total @ pts_trans.T).T
    
    # Perspective Division and Intrinsic Projection
    pts_2d_hom = (K @ pts_cam.T).T
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:]
    
    return pts_2d
```

## Results & Verification

To validate our projection pipeline, we project 3D runway threshold keypoints extracted from `runways_db_V2_GEarth.json` using simulator metadata and overlay them onto generated synthetic images.

![Projection Results Verification](/assets/images/posts/3d-to-2d-projection/10_result.png)

As demonstrated above:
* **Green dots**: Ground-truth keypoint annotations provided in the simulator metadata CSV.
* **Red lines/dots**: Analytically projected 2D keypoints computed via our reference frame transformation chain.

The perfect alignment confirms that the extrinsic transformation matrix correctly bridges ECEF, NED, Aircraft Body, and Camera Optical frames without geometric distortion or alignment offsets.

---

> ## Key Takeaways
>
> * **Always chain transformations explicitly**: Do not skip the NED or Body frame when converting global geocentric coordinates (ECEF) to camera projection space.
> * **Watch axis definitions**: Flight body frames ($X$ front, $Y$ right, $Z$ down) differ from computer vision camera frames ($Z$ front, $X$ right, $Y$ down). The swap matrix $\mathbf{R}_{body \to cam}$ is critical.
> * **Analytical intrinsics are sufficient**: For synthetic data generated via engines like FLSim, camera intrinsic matrices can be derived directly from Field of View without empirical board calibration.