import numpy as np
import open3d as o3d
import open3d_plus as o3dp
from scipy.spatial.transform import Rotation as R
import copy

from models.graspnet.graspnet_baseline import GraspNetBaseLine
from scripts.utils import graspnet_config


# Modified from https://github.com/OCRTOC/OCRTOC_software_package/blob/master/ocrtoc_perception/src/ocrtoc_perception/perceptor.py
class Graspnet:
    def __init__(self):
        self.config = graspnet_config
        self.graspnet_baseline = GraspNetBaseLine(checkpoint_path = self.config['graspnet_checkpoint_path'])

    def convert_grasps_to_pose_set(self, gg):
        """
        Convert a GraspGroup to a list of grasp poses without any filtering
        
        Args:
            gg (GraspGroup): Grasp group to convert
        
        Returns:
            list: List of grasp poses in the same format as grasp_pose_set
        """
        grasp_pose_set = []
        
        ts = gg.translations
        rs = gg.rotation_matrices
        depths = gg.depths
        
        # move the center to the eelink frame
        ts = ts + rs[:,:,0] * (np.vstack((depths, depths, depths)).T)
        eelink_rs = np.zeros(shape=(len(rs), 3, 3), dtype=np.float32)
        
        # the coordinate systems are different in graspnet and ocrtoc
        eelink_rs[:,:,0] = rs[:,:,2]
        eelink_rs[:,:,1] = -rs[:,:,1]
        eelink_rs[:,:,2] = rs[:,:,0]
        
        # Convert each grasp to a pose
        for i in range(len(gg)):
            grasp_rotation_matrix = eelink_rs[i]
            if np.linalg.norm(np.cross(grasp_rotation_matrix[:,0], grasp_rotation_matrix[:,1]) - grasp_rotation_matrix[:,2]) > 0.1:
                grasp_rotation_matrix[:,0] = -grasp_rotation_matrix[:, 0]
            
            grasp_pose = np.zeros(7)
            grasp_pose[:3] = [ts[i][0], ts[i][1], ts[i][2]]
            r = R.from_matrix(grasp_rotation_matrix)
            grasp_pose[-4:] = r.as_quat()
            
            grasp_pose_set.append(grasp_pose)
        
        return grasp_pose_set



    def compute_grasp_pose(self, full_pcd):
      points, _ = o3dp.pcd2array(full_pcd)
      grasp_pcd = copy.deepcopy(full_pcd)
      grasp_pcd.points = o3d.utility.Vector3dVector(-points)

      # Generating grasp poses
      gg_before = self.graspnet_baseline.inference(grasp_pcd)
    #   print(f"DEBUGGING: Grasp candidates before collision detection: {len(gg_before)}")
      
      gg_before.translations = -gg_before.translations
      gg_before.rotation_matrices = -gg_before.rotation_matrices
      gg_before.translations = gg_before.translations + gg_before.rotation_matrices[:, :, 0] * self.config['refine_approach_dist']
      
      gg = self.graspnet_baseline.collision_detection(gg_before, points)
    #   print(f"DEBUGGING: Grasp candidates after collision detection: {len(gg)}")

      return gg_before, gg

    def sort_grasps_by_angle(self, gg):
        """
        Sort grasps based on their angle from the vertical axis
        
        Args:
            gg (GraspGroup): Input grasp group
        
        Returns:
            Sorted grasp group with grasps ordered from most vertical to least vertical
        """
        # Calculate angles from vertical axis
        # Assuming rs[:, 2, 0] represents the z-component of the approach vector
        angles = np.arccos(-gg.rotation_matrices[:, 2, 0]) * 180 / np.pi
        
        # Get sorting indices (from smallest angle to largest)
        sort_indices = np.argsort(angles)
        
        # Sort the grasp group using these indices
        sorted_gg = gg[sort_indices]
        
        # print("DEBUGGING: Sorted grasp angles (degrees):")
        # for angle in angles[sort_indices]:
        #     print(f"  {angle:.2f}")
        
        return sorted_gg
    
    def grasp_detection(self, full_pcd, object_poses=None):
        # Generate initial grasps
        gg_before, gg = self.compute_grasp_pose(full_pcd)
        
        # Sort grasps by angle
        sorted_gg = self.sort_grasps_by_angle(gg)
        
        # Convert sorted grasps to pose set
        sorted_grasp_pose_set = self.convert_grasps_to_pose_set(sorted_gg)
        

        return sorted_grasp_pose_set


