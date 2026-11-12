import open3d as o3d
import numpy as np

def create_mesh(pcd):
    print("Estimating normals...")
    pcd.estimate_normals()

    # Step 1: Poisson reconstruction
    print("Creating mesh (Poisson)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )

    # Step 2: Convert densities to numpy
    densities = np.asarray(densities)

    # Step 3: Remove low-density vertices (IMPORTANT)
    threshold = np.quantile(densities, 0.35)  # remove lowest 10%
    vertices_to_remove = densities < threshold

    # clean mesh
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    return mesh
