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

## Input Data Format (JSON & CSV)

Before setting up the geometric equations, let's examine how the simulator structures its input telemetry and ground truth data.

### 1. 3D World Coordinates (`runways_db_V2_FLSim.json`)
The database contains 3D global ECEF coordinates $(X, Y, Z)$ for the four threshold corners ($A, B, C, D$) of each runway:

```json
{
  "CYEG": {
    "20": {
      "A": { "position": { "x": -1885420.5, "y": -4274921.2, "z": 5098231.1 } },
      "B": { "position": { "x": -1885410.1, "y": -4274940.8, "z": 5098210.4 } },
      "C": { "position": { "x": -1883900.2, "y": -4275800.0, "z": 5097600.0 } },
      "D": { "position": { "x": -1883910.0, "y": -4275780.5, "z": 5097620.1 } }
    }
  }
}
```

### 2. Simulator Telemetry & 2D Keypoints (`metadata_flsim_train.csv`)
The dataset CSV provides camera pose parameters (latitude, longitude, altitude, Euler angles) along with reference 2D pixel annotations ($u, v$) for keypoint validation:

```csv
image;width;height;airport;runway;lat;lon;alt;yaw;pitch;roll;x_TR;y_TR;x_TL;y_TL;x_BL;y_BL;x_BR;y_BR
CYEG-20_000.jpg;1024;1024;CYEG;20;53.376196;-113.519482;1010.36;-149.64;87.63;-1.54;576;506;570;506;562;522;571;523
```

The 3D keypoints ($A, B, C, D$) map to 2D image annotations as follows:
* **Top-Right (TR)** $\leftrightarrow$ Corner $A$
* **Top-Left (TL)** $\leftrightarrow$ Corner $B$
* **Bottom-Left (BL)** $\leftrightarrow$ Corner $C$
* **Bottom-Right (BR)** $\leftrightarrow$ Corner $D$

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

![Three-Step Transformation Pipeline](/assets/images/posts/3d-to-2d-projection/6_issue_images.png)

---

## Step-by-Step Coordinate Frame Transformations

### A Note on Simulator Earth Models (Geodetic to ECEF)

Before executing coordinate frame rotations, telemetry coordinates (Latitude, Longitude, Altitude) must be converted into the global Cartesian ECEF system. 

Depending on the physics engine of the flight simulator, the virtual earth might be modeled as a perfect sphere rather than a standard WGS84 ellipsoid. For instance, in our specific pipeline, we map Geodetic to ECEF assuming a constant Earth radius ($R = 6371010$ meters). If high-precision real-world data is required, libraries like `pyproj` should be used to account for ellipsoidal eccentricity.

### Step 1: ECEF to Local NED Frame (Earth Curvature Alignment)

To convert global ECEF coordinates into a local planar frame centered at the aircraft location $(\phi_{lat}, \lambda_{lon})$, we apply the rotation matrix $\mathbf{R}_{ecef \to ned}$:

$$\mathbf{R}_{ecef \to ned} = \begin{bmatrix} -\sin\phi_{lat} \cos\lambda_{lon} & -\sin\phi_{lat} \sin\lambda_{lon} & \cos\phi_{lat} \\ -\sin\lambda_{lon} & \cos\lambda_{lon} & 0 \\ -\cos\phi_{lat} \cos\lambda_{lon} & -\cos\phi_{lat} \sin\lambda_{lon} & -\sin\phi_{lat} \end{bmatrix}$$

Given a 3D point $\mathbf{X}_{ECEF}$ and camera position $\mathbf{C}_{ECEF}$, the vector in local NED coordinates is:

$$\mathbf{X}_{NED} = \mathbf{R}_{ecef \to ned} \cdot (\mathbf{X}_{ECEF} - \mathbf{C}_{ECEF})$$

---

### Step 2: Local NED to Aircraft Body Frame (Attitude Rotation)

The aircraft attitude provided by flight simulators consists of standard aeronautical Euler angles: **Yaw ($\psi$)**, **Pitch ($\theta$)**, and **Roll ($\phi$)**. 

> **Pitch Angle Convention Warning**
> 
> Flight simulators like FLSim often express the raw pitch angle relative to the **zenith** ($0^\circ$ pointing straight up, $90^\circ$ at horizontal) rather than standard aeronautical pitch ($0^\circ$ at horizontal). 
> 
> Before constructing the rotation matrix, you must convert zenith pitch to standard pitch:
>
> $$\theta_{true} = \theta_{raw} - 90^\circ$$
>
> For instance, a simulator value of `pitch = 87.64°` actually represents a nose-down pitch of $-2.36^\circ$.

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

def project_3d_to_2d(pts_3d_ecef, cam_pos_ecef, ac_ll, ac_att, K):
    """
    Projects 3D ECEF points onto 2D image coordinates using homogeneous matrices.
    
    :param pts_3d_ecef: List of 3D keypoints in ECEF [X, Y, Z]
    :param cam_pos_ecef: (3,) camera ECEF position [X, Y, Z]
    :param ac_ll: (2,) aircraft [lat, lon] in degrees
    :param ac_att: (3,) aircraft attitude [yaw, pitch, roll] in degrees
    :param K: (3, 3) camera intrinsics matrix
    :return: (N, 2) projected 2D pixel coordinates
    """
    R_ecef2ned = ecef_to_ned_matrix(ac_ll[0], ac_ll[1])
    R_ned2body = ned_to_body_matrix(ac_att[0], ac_att[1], ac_att[2])
    
    # Axis swap: Aircraft Body to Camera Optical Frame
    R_body2cam = np.array([
        [0, 1, 0], 
        [0, 0, 1], 
        [1, 0, 0]
    ])
    
    # Combined Extrinsic Rotation Matrix (R)
    R_ext = R_body2cam @ R_ned2body @ R_ecef2ned
    
    # Translation vector t = -R * C
    t_ext = -R_ext @ cam_pos_ecef
    
    # Construct [3x4] Extrinsic Matrix E = [R | t]
    E = np.zeros((3, 4))
    E[:3, :3] = R_ext
    E[:3, 3] = t_ext
    
    pts_2d = []
    for pt in pts_3d_ecef:
        # 1. Convert to Homogeneous 3D coordinates [X, Y, Z, 1]^T
        Xw = np.array([pt[0], pt[1], pt[2], 1.0]).reshape(4, 1)
        
        # 2. Transform to Camera Optical Frame
        Xw_c = E @ Xw
        
        # 3. Guard against points behind or too close to the camera (Z <= 0)
        # In a full pipeline, segment clipping is required here.
        if Xw_c[2, 0] < 1.0:
            continue
            
        # 4. Intrinsic Projection and Perspective Division
        u = K @ Xw_c
        px = int(u[0, 0] / u[2, 0])
        py = int(u[1, 0] / u[2, 0])
        pts_2d.append([px, py])
        
    return np.array(pts_2d)
```

> **Handling Points Behind the Camera**
> 
> In a simulator environment, the aircraft might fly extremely close to or directly over the runway threshold. If a 3D keypoint falls behind the camera's focal plane (where $Z \le 0$ in the optical frame), standard perspective division ($X/Z$, $Y/Z$) will mathematically flip the point or cause a division by zero. 
> 
> To prevent this, the implementation must explicitly filter or adjust points where $Z < 1.0$ before applying the intrinsic projection matrix.

### Putting It All Together

Using the data from our JSON and CSV examples, the full pipeline execution looks like this:

```python
# 1. Extract metadata from CSV
lat, lon, alt = 53.376196, -113.519482, 1010.36
raw_yaw, raw_pitch, raw_roll = -149.64, 87.63, -1.54

# 2. Convert Geodetic to ECEF (Spherical model approximation)
# For strict WGS84, use pyproj.Transformer
R_earth = 6371010
cam_pos_ecef = llh2ecef_spherical(lat, lon, alt, R_earth)

# 3. Correct Zenith Pitch to standard aeronautical Pitch
ac_att = [raw_yaw, raw_pitch - 90.0, raw_roll]
ac_ll = [lat, lon]

# 4. Generate Intrinsics and Project
K = compute_intrinsics(fov_v_deg=60, width=1024, height=1024)

# 3D points from JSON (Corners A, B, C, D)
pts_3d = [
    [-1885420.5, -4274921.2, 5098231.1],
    [-1885410.1, -4274940.8, 5098210.4],
    [-1883900.2, -4275800.0, 5097600.0],
    [-1883910.0, -4275780.5, 5097620.1]
]

projected_pixels = project_3d_to_2d(pts_3d, cam_pos_ecef, ac_ll, ac_att, K)

## Results & Verification

To validate our projection pipeline, we project 3D runway threshold keypoints extracted from `runways_db_V2_FLSim.json` using simulator metadata and overlay them onto generated synthetic images.

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