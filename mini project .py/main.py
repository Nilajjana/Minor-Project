import cv2
import open3d as o3d
from depth import get_depth, load_model
from pointcloud import create_point_cloud
from mesh import create_mesh
from multiview import generate_views
import torch
import copy
import numpy as np

model, transform, device = load_model()

IMAGE_PATH = "input.jpg"

print("Loading image...")
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Generate multiple views
views, view_transforms = generate_views(img)

all_pcd = o3d.geometry.PointCloud()

print("Processing multiple views...")

for i, (view, view_transform) in enumerate(zip(views, view_transforms)):
    print(f"View {i+1}/{len(views)}")

    # Save temporary view
    temp_path = f"temp_{i}.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))

    depth, _ = get_depth(temp_path, model, transform, device)
    print(depth.shape, view.shape)

    pcd = create_point_cloud(depth, view, view_transform)

    # Merge point clouds
    if i == 0:
      all_pcd = pcd
    else:
        threshold = 0.05
        reg = o3d.pipelines.registration.registration_icp(
            pcd, all_pcd,
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        pcd.transform(reg.transformation)
        all_pcd += pcd

    torch.cuda.empty_cache()

plane_model, inliers = all_pcd.segment_plane(
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=1000
)

#Only removing plane if its large
if len(inliers) > 0.3 * len(all_pcd.points):
   all_pcd = all_pcd.select_by_index(inliers, invert=True)

# Downsample for stability
all_pcd = all_pcd.voxel_down_sample(voxel_size=0.01)
all_pcd, _ = all_pcd.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)

all_pcd, _ = all_pcd.remove_radius_outlier(
    nb_points=16,
    radius=0.05
)

print("Estimating normals...")
all_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
)
all_pcd.orient_normals_consistent_tangent_plane(100)

print("Saving point cloud...")
o3d.io.write_point_cloud("multiview.ply", all_pcd)

print("Creating mesh...")
mesh = create_mesh(all_pcd)

mesh = mesh.filter_smooth_simple(number_of_iterations=3)

o3d.io.write_triangle_mesh("multiview.obj", mesh)

print("Done!")

o3d.visualization.draw_geometries([mesh])
