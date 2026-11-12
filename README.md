# Hybrid 3D Reconstruction

A lightweight pipeline that reconstructs an approximate **3D mesh from a single 2D image**. It combines monocular deep-learning depth estimation (**MiDaS**) with classical geometry processing (point clouds, **ICP** alignment, **Poisson surface reconstruction**) using Open3D.

Developed as a B.Tech project at the School of Computer Engineering, KIIT Deemed to be University, Bhubaneswar (2025-26), under the guidance of Dr. Ajay Anand.

---

## Table of Contents

* [Overview](#overview)
* [Pipeline](#pipeline)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Key Parameters](#key-parameters)
* [Results](#results)
* [Limitations](#limitations)
* [Future Scope](#future-scope)
* [Team](#team)
* [References](#references)
* [License](#license)

---

## Overview

Classical 3D reconstruction (voxel carving, silhouette methods, multi-view stereo) needs several calibrated photos taken in a controlled setup. This project takes a different route:

1. Predict a dense **relative depth map** from one RGB image with MiDaS.
2. Back-project every pixel into 3D with a pinhole camera model to get a **coloured point cloud**.
3. Simulate extra viewpoints by **rotating** the input image (a "pseudo-multiview" step inspired by multi-view stereo), run the same depth and point-cloud steps on each view, and merge the clouds with **ICP**.
4. Clean the merged cloud and turn it into a surface with **Poisson reconstruction**.

The goal is a pipeline that runs on standard hardware (CUDA optional) and produces a visually meaningful, approximate 3D model.

---

## Pipeline

```text
input.jpg
   │
   ▼
Multi-view generation   (rotations of -10°, 0°, +10°)          multiview.py
   │
   ▼
Depth estimation        (MiDaS DPT_Hybrid, normalise, blur)    depth.py
   │
   ▼
Point cloud generation  (pinhole back-projection, colours)     pointcloud.py
   │
   ▼
ICP alignment + fusion  (each view → first view)               main.py
   │
   ▼
Cleaning                (plane removal, voxel downsample,
   │                     statistical + radius outlier removal,
   │                     normal estimation)                      main.py
   ▼
Poisson reconstruction  (depth = 9, crop to bounding box)      mesh.py
   │
   ▼
Smoothing + export      (multiview.ply, multiview.obj, viewer)  main.py
```

### Stage details

| Stage | What it does |
| ----- | ------------ |
| **Multi-view** | Rotates the image about its centre by each angle in `[-10, 0, 10]` and stores the inverse affine transform for each view. |
| **Depth** | Resizes to 384×384, runs MiDaS `DPT_Hybrid`, min-max normalises, applies a gamma of 1.5, resizes back and applies a 3×3 Gaussian blur. |
| **Point cloud** | Uses `fx = fy = max(h, w)` with the principal point at the image centre. Depth is normalised, inverted and scaled; border pixels (30 px margin), very dark background pixels and out-of-range depths are skipped. RGB colour is kept per point. |
| **ICP fusion** | The first view is the reference. Every other view is aligned to the merged cloud with point-to-point ICP (threshold 0.05) and appended. |
| **Cleaning** | Removes the dominant plane with RANSAC only if it holds more than 30% of the points, then voxel-downsamples (0.01), removes statistical and radius outliers, and estimates and orients normals. |
| **Meshing** | Poisson surface reconstruction (depth 9), cropped to the point cloud's bounding box, followed by 3 iterations of simple smoothing. |

---

## Project Structure

```text
.
├── main.py          # Entry point: runs the full pipeline, ICP fusion, cleaning, export
├── depth.py         # MiDaS model loading, depth inference, rembg background-removal helper
├── multiview.py     # Pseudo-multiview generation (image rotations + inverse transforms)
├── pointcloud.py    # Depth map + image → coloured Open3D point cloud
├── mesh.py          # Point cloud → mesh using Poisson reconstruction
├── LICENSE          # MIT
└── README.md
```

---

## Installation

Requirements: Python 3 and the packages below. A CUDA GPU is optional; the code falls back to CPU automatically.

```bash
pip install torch torchvision opencv-python numpy open3d rembg pillow
```

On the first run, `torch.hub` downloads the MiDaS model, so an internet connection is needed once.

---

## Usage

1. Put your image in the project folder and name it `input.jpg` (or change `IMAGE_PATH` in `main.py`).
2. Run:

```bash
python main.py
```

3. The script prints progress for each view, then writes:

| Output | Description |
| ------ | ----------- |
| `multiview.ply` | The cleaned, fused, coloured point cloud |
| `multiview.obj` | The reconstructed triangle mesh |
| `temp_0.jpg`, `temp_1.jpg`, `temp_2.jpg` | Temporary rotated views (can be deleted) |

Finally an Open3D window opens so you can inspect the mesh.

---

## Key Parameters

These are hand-tuned values, so expect to adjust them per image.

| Where | Parameter | Current value | Effect |
| ----- | --------- | ------------- | ------ |
| `multiview.py` | `angles` | `[-10, 0, 10]` | Rotations used to simulate extra views |
| `depth.py` | gamma | `1.5` | Contrast of the depth map |
| `pointcloud.py` | depth scale | `1.5` | Overall depth extent (try 1.5–3.0) |
| `pointcloud.py` | `margin` | `30` px | Border pixels ignored (try 15–40) |
| `pointcloud.py` | depth cut-offs | `0.1 < Z < 1.9` | Background rejection |
| `main.py` | ICP `threshold` | `0.05` | Maximum correspondence distance |
| `main.py` | plane removal | `> 30%` of points | Only removes large planes (e.g. table/floor) |
| `main.py` | `voxel_size` | `0.01` | Point cloud density |
| `mesh.py` | Poisson `depth` | `9` | Mesh resolution / detail |

---

## Results

Example inputs and the resulting meshes viewed in Open3D:

| Input | Reconstructed mesh |
| ----- | ------------------ |
| `docs/images/mug_input.jpg` | `docs/images/mug_mesh.png` |
| `docs/images/box_input.jpg` | `docs/images/box_mesh.png` |

<!-- Add the images to docs/images/ and they will show up here. -->

The reconstructions capture the overall shape (the mug body and handle, the box's top face and side pattern), but fine detail is limited. See [Limitations](#limitations).

---

## Limitations

* **Relative depth only.** MiDaS predicts relative, not metric, depth, so the global scale of the model is not accurate.
* **Pseudo-views add limited information.** Rotating a single image does not reveal unseen geometry, so the back and hidden sides are not truly reconstructed.
* **Background and surface leakage.** Parts of the supporting surface or background can end up in the mesh, and Poisson reconstruction can create holes or blobby regions on complex or textured areas.
* **Baked-in lighting.** Colours come straight from the photo, so shading and highlights are part of the texture. There is no UV unwrapping or texture mapping.
* **Slow point-cloud step.** `create_point_cloud` loops over pixels in Python and is not vectorised.
* **Sensitive to input quality** and to the hand-tuned thresholds above.

---

## Future Scope

* Use real multi-view images instead of synthetic rotations
* Improve depth accuracy (newer depth models, metric depth)
* Add proper texture mapping and UV unwrapping
* Better mesh refinement and smoothing
* Explore neural reconstruction models
* Vectorise point-cloud generation for speed

---

## Team

Developed by students of the School of Computer Engineering, KIIT Deemed to be University:

* Yash Kumar
* Arya Animesh Swain
* Nilaj Jana
* Anik Adhikari
* Ankur Bandopadhyay
* Jyotirmay Padhiary

**Project guide:** Dr. Ajay Anand

---

## References

1. A. Y. Mulayim, U. Yilmaz, M. V. Atalay, "Silhouette-based 3-D model reconstruction from multiple images," *IEEE Trans. Systems, Man, and Cybernetics, Part B*, 33, 2003.
2. S. Sinha, M. Pollefeys, "Multi-View Reconstruction Using Photo-consistency and Exact Silhouette Constraints: A Maximum-Flow Formulation," *ICCV*, 2005.
3. M. Jeyakumar et al., "3D Model Generator: AI-powered 2D-to-3D reconstruction using generative models and web-based visualization," *ICAAIC-2025*, IEEE.
4. MiDaS monocular depth estimation: <https://github.com/isl-org/MiDaS>
5. Open3D: <http://www.open3d.org/>

---

## License

Released under the [MIT License](LICENSE).
