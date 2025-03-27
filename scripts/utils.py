import math
import numpy as np
import pybullet as p
import cv2
import open3d as o3d
import open3d_plus as o3dp
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from env.constants import WORKSPACE_LIMITS, PIXEL_SIZE


reconstruction_config = {
    'nb_neighbors': 50,
    'std_ratio': 2.0,
    'voxel_size': 0.0015,
    'icp_max_try': 5,
    'icp_max_iter': 2000,
    'translation_thresh': 3.95,
    'rotation_thresh': 0.02,
    'max_correspondence_distance': 0.02
}

graspnet_config = {
    'graspnet_checkpoint_path': 'models/graspnet/logs/log_rs/checkpoint.tar',
    'refine_approach_dist': 0.01,
    'dist_thresh': 0.05,
    'angle_thresh': 15,
    'mask_thresh': 0.5
}

def get_heightmap(points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
        points: HxWx3 float array of 3D points in world coordinates.
        colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
        bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
            region in 3D space to generate heightmap in world coordinates.
        pixel_size: float defining size of each pixel in meters.
    Returns:
        heightmap: HxW float array of height (from lower z-bound) in meters.
        colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
    """
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[px, py] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[px, py, c] = colors[:, c]
    return heightmap, colormap


def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.
    Args:
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
        points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
        points: HxWx3 float array of 3D points in camera coordinates.
        transform: 4x4 float array representing a rigid transformation matrix.
    Returns:
        points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding, "constant", constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points


def reconstruct_heightmaps(color, depth, configs, bounds, pixel_size):
    """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
    heightmaps, colormaps = [], []
    for color, depth, config in zip(color, depth, configs):
        intrinsics = config["intrinsics"]
        xyz = get_pointcloud(depth, intrinsics)
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        xyz = transform_pointcloud(xyz, transform)
        heightmap, colormap = get_heightmap(xyz, color, bounds, pixel_size)
        print("Heightmap shape:", heightmap.shape)  # Check size
        print("Colormap shape:", colormap.shape)
        heightmaps.append(heightmap)
        colormaps.append(colormap)

    return heightmaps, colormaps


def get_fuse_heightmaps(obs, configs, bounds, pixel_size):
    """Reconstruct orthographic heightmaps with segmentation masks."""
    heightmaps, colormaps = reconstruct_heightmaps(
        obs["color"], obs["depth"], configs, bounds, pixel_size
    )
    colormaps = np.float32(colormaps)
    heightmaps = np.float32(heightmaps)

    # Fuse maps from different views.
    valid = np.sum(colormaps, axis=3) > 0
    repeat = np.sum(valid, axis=0)
    repeat[repeat == 0] = 1
    cmap = np.sum(colormaps, axis=0) / repeat[Ellipsis, None]
    cmap = np.uint8(np.round(cmap))
    hmap = np.max(heightmaps, axis=0)  # Max to handle occlusions.

    return cmap, hmap


def get_true_heightmap(env):
    """Get RGB-D orthographic heightmaps and segmentation masks in simulation."""

    # Capture near-orthographic RGB-D images and segmentation masks.
    color, depth, segm = env.render_camera(env.oracle_cams[0])

    # Combine color with masks for faster processing.
    color = np.concatenate((color, segm[Ellipsis, None]), axis=2)
    # print("Color shape after concatenation:", color.shape)

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = reconstruct_heightmaps(
        [color], [depth], env.oracle_cams, env.bounds, env.pixel_size
    )

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis, :3]
    # print("cmap shape:", cmap.shape)
    hmap = np.float32(hmaps)[0, Ellipsis]
    mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()

    return cmap, hmap, mask


def get_heightmap_from_real_image(color, depth, segm, env):
    # Combine color with masks for faster processing.
    color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = reconstruct_heightmaps(
        [color], [depth], env.camera.configs, env.bounds, env.pixel_size
    )

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis, :3]
    hmap = np.float32(hmaps)[0, Ellipsis]
    mask = np.uint8(cmaps)[0, Ellipsis, 3:].squeeze()

    return cmap, hmap, mask


def process_pcds(pcds, reconstruction_config):
    trans = dict()
    pcd = pcds[0]
    pcd.estimate_normals()
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors = reconstruction_config['nb_neighbors'],
        std_ratio = reconstruction_config['std_ratio']
    )
    for i in range(1, len(pcds)):
        voxel_size = reconstruction_config['voxel_size']
        income_pcd, _ = pcds[i].remove_statistical_outlier(
            nb_neighbors = reconstruction_config['nb_neighbors'],
            std_ratio = reconstruction_config['std_ratio']
        )
        income_pcd.estimate_normals()
        income_pcd = income_pcd.voxel_down_sample(voxel_size)
        transok_flag = False
        for _ in range(reconstruction_config['icp_max_try']): # try 5 times max
            reg_p2p = o3d.pipelines.registration.registration_icp(
                income_pcd,
                pcd,
                reconstruction_config['max_correspondence_distance'],
                np.eye(4, dtype = np.float64),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(reconstruction_config['icp_max_iter'])
            )
            if (np.trace(reg_p2p.transformation) > reconstruction_config['translation_thresh']) \
                and (np.linalg.norm(reg_p2p.transformation[:3, 3]) < reconstruction_config['rotation_thresh']):
                # trace for transformation matrix should be larger than 3.5
                # translation should less than 0.05
                transok_flag = True
                break
        if not transok_flag:
            reg_p2p.transformation = np.eye(4, dtype = np.float32)
        income_pcd = income_pcd.transform(reg_p2p.transformation)
        trans[i] = reg_p2p.transformation
        pcd = o3dp.merge_pcds([pcd, income_pcd])
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals()
    return trans, pcd


def get_fuse_pointcloud(env):
    pcds = []
    configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    # configs = [env.oracle_cams[0], env.agent_cams[0]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        color, depth, _ = env.render_camera(config)
        xyz = get_pointcloud(depth, config["intrinsics"])
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        points = transform_pointcloud(xyz, transform)
        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
        iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
        iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = color[valid]
        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd.voxel_down_sample(reconstruction_config['voxel_size'])
        # # visualization
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        # the first pcd is the one for start fusion
        pcds.append(pcd)

    _, fuse_pcd = process_pcds(pcds, reconstruction_config)
    # visualization
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([fuse_pcd, frame])

    return fuse_pcd


def get_true_bboxs(env, color_image, depth_image, mask_image):
    # get mask of all objects
    bbox_images = []
    bbox_positions = []
    for obj_id in env.obj_ids["rigid"]:
        mask = np.zeros(mask_image.shape).astype(np.uint8)
        mask[mask_image == obj_id] = 255
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        stats = stats[stats[:,4].argsort()]
        if stats[:-1].shape[0] > 0:
            bbox = stats[:-1][0]
            # for bbox
            # |(y0, x0)         |   
            # |                 |
            # |                 |
            # |         (y1, x1)|
            x0, y0 = bbox[0], bbox[1]
            x1 = bbox[0] + bbox[2]
            y1 = bbox[1] + bbox[3]

            # visualization
            start_point, end_point = (x0, y0), (x1, y1)
            color = (0, 0, 255) # Red color in BGR
            thickness = 1 # Line thickness of 1 px 
            mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
            # cv2.imwrite('mask_bboxs.png', mask_bboxs)

            bbox_image = color_image[y0:y1, x0:x1]
            bbox_images.append(bbox_image)
            
            pixel_x = (x0 + x1) // 2
            pixel_y = (y0 + y1) // 2
            bbox_pos = [
                pixel_y * PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
                pixel_x * PIXEL_SIZE + WORKSPACE_LIMITS[1][0],
                depth_image[pixel_y][pixel_x] + WORKSPACE_LIMITS[2][0],
            ]
            bbox_positions.append(bbox_pos)

    return bbox_images, bbox_positions

# Convert image coordinates to workspace coordinates
def image_to_workspace(x, y, z):
    workspace_x = x * PIXEL_SIZE + WORKSPACE_LIMITS[0][0]
    workspace_y = y * PIXEL_SIZE + WORKSPACE_LIMITS[1][0]
    workspace_z = z  
    return workspace_x, workspace_y, workspace_z

def crop_pointcloud(pcd, cropping_box, color_image, depth_image):
    # Convert the 2D cropping box coordinates to 3D coordinates
    x1, y1, x2, y2 = cropping_box
    depth_crop = depth_image[y1:y2, x1:x2]
    color_crop = color_image[y1:y2, x1:x2]
    mask = (depth_crop > 0)
    points = []
    colors = []

    for i in range(depth_crop.shape[0]):
        for j in range(depth_crop.shape[1]):
            if mask[i, j]:
                x = j + x1
                y = i + y1
                z = depth_crop[i, j]
                points.append((x, y, z))
                colors.append(color_crop[i, j] / 255.0)  # Normalize the color values

    points = np.array(points)
    colors = np.array(colors)

    image_height, image_width = depth_image.shape
    workspace_points = [image_to_workspace(x, y, z) for x, y, z in points]
    workspace_points = np.array(workspace_points)

    # Create a grid of points for the entire workspace with z=0
    grid_x, grid_y = np.meshgrid(
        np.linspace(WORKSPACE_LIMITS[0][0], WORKSPACE_LIMITS[0][1], image_width),
        np.linspace(WORKSPACE_LIMITS[1][0], WORKSPACE_LIMITS[1][1], image_height)
    )
    grid_z = np.zeros_like(grid_x)
    full_workspace_points = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3)
    full_colors = np.zeros((full_workspace_points.shape[0], 3))

    # Update the points in the cropped region with actual height
    for idx, (x, y, z) in enumerate(workspace_points):
        # Convert x, y from workspace coordinates back to image coordinates to find the corresponding index
        image_x = int((x - WORKSPACE_LIMITS[0][0]) / PIXEL_SIZE)
        image_y = int((y - WORKSPACE_LIMITS[1][0]) / PIXEL_SIZE)

        # Ensure the indices are within bounds
        if 0 <= image_x < image_width and 0 <= image_y < image_height:
            grid_index = image_x * image_height + image_y  # Correct index calculation
            full_workspace_points[grid_index, 2] = z
            full_colors[grid_index] = colors[idx]

    # Create a new point cloud from the full workspace points
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(full_workspace_points)
    full_pcd.colors = o3d.utility.Vector3dVector(full_colors)

    return full_pcd
