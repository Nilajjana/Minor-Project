import open3d as o3d
import numpy as np
import cv2

def create_point_cloud(depth, image, transform=None):
    #ensuring same size
    h_d, w_d = depth.shape
    h_i, w_i, _ = image.shape

    if (h_d != h_i) or (w_d != w_i):
        image = cv2.resize(image, (w_d, h_d))

    h, w = depth.shape

    # STEP 1: Better camera model
    fx = fy = max(h, w)
    cx = w / 2
    cy = h / 2

    # STEP 2: Fix depth scaling (VERY IMPORTANT)
    z = depth.copy()
    z = (z - z.min()) / (z.max() - z.min())  # normalize
    z = 1.0 - z                              # invert
    z = z * 1.5                              # scale depth (tune 1.5–3.0)

    # Ensure RGB only
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Normalize colors
    image = image.astype(np.float32) / 255.0

    points = []
    colors = []

    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    margin = 30 #pixels (tune 15-40)

    for y in range(h):
        for x in range(w):

            if x < margin or x > w - margin or y < margin or y > h - margin:
                 continue

            Z = z[y, x]

            if mask[y, x] == 0 and Z > 1.2:
                 continue

            # STEP 3: Remove background using depth mask
            if Z > 1.9 or Z < 0.1:   # threshold (tune between 1.5–2.0)
                continue

            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy

            point = np.array([X, Y, Z])

            # apply inverse transform (align views)
            if transform is not None:
                px = x
                py = y
                tx = transform[0,0]*px + transform[0,1]*py + transform[0,2]
                ty = transform[1,0]*px + transform[1,1]*py + transform[1,2]

                point[0] += (tx - cx) / fx
                point[1] += (ty - cy) / fy

            points.append(point)
            colors.append(image[y, x])
 
    # Convert for Open3D
    points = np.array(points, dtype=np.float64)
    colors = np.array(colors, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
