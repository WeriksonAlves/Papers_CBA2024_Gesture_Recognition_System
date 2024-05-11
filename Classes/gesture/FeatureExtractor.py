import numpy as np

class FeatureExtractor:
    """
    Class responsible for extracting features from data.
    """
    @staticmethod
    def calculate_joint_xy(pose_data, joint_index):
        """
        Extracts the x and y coordinates of a specific joint from pose data.

        Args:
            pose_data: Data structure containing information about a person's pose.
            joint_index: Index of the joint to extract from the `pose_data`.

        Returns:
            np.ndarray: Array containing the x and y coordinates of the specified joint.
        """
        joint = pose_data.landmark[joint_index]
        return np.array([joint.x, joint.y])
    
    @staticmethod
    def calculate_joint_xyz(pose_data, joint_index):
        """
        Extracts the x, y, and z coordinates of a specific joint from pose data.

        Args:
            pose_data: Data structure containing information about a person's pose.
            joint_index: Index of the joint to extract from the `pose_data`.

        Returns:
            np.ndarray: Array containing the x, y, and z coordinates of the specified joint.
        """
        joint = pose_data.landmark[joint_index]
        return np.array([joint.x, joint.y, joint.z])

