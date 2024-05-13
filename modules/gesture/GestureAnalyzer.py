import numpy as np

from typing import Tuple
from sklearn.decomposition import PCA
from .FeatureExtractor import FeatureExtractor

class GestureAnalyzer:
    """
    Class responsible for analyzing gestures.
    """
    @staticmethod
    def calculate_ref_pose(data: np.ndarray, joints: np.ndarray, dimension: int = 2) -> np.ndarray:
        """
        Calculates the reference pose based on input data and joint positions in either 2D or 3D dimensions.

        Args:
            data (np.ndarray): Input data containing joint positions.
            joints (np.ndarray): Indices of joints in the skeleton.
            dimension (int, optional): Dimensionality for pose calculation (2D or 3D). Defaults to 2.

        Returns:
            np.ndarray: Reference pose calculated based on the input data and joints.
        """
        pose_vector = []
        for joint in joints:
            if dimension == 3:
                pose_vector.append(FeatureExtractor.calculate_joint_xyz(data, joint))
            elif dimension == 2:
                pose_vector.append(FeatureExtractor.calculate_joint_xy(data, joint))
            else:
                raise ValueError("Invalid dimension parameter")
        reference_pose = np.mean(pose_vector, axis=0)
        return reference_pose
    
    @staticmethod
    def check_trigger_enabled(storage_trigger: np.ndarray, length: int = 30, dist: float = 0.03) -> Tuple[bool, np.ndarray, float]:
        """
        Checks if a trigger is enabled based on the input array, length, and distance criteria.

        Args:
            storage_trigger (np.ndarray): Array containing trigger data points.
            length (int, optional): Minimum number of elements in the `storage_trigger` array. Defaults to 30.
            dist (float, optional): Threshold distance value. Defaults to 0.03.

        Returns:
            Tuple[bool, np.ndarray, float]: Boolean indicating whether the trigger is enabled, a subset of `storage_trigger`, and the calculated distance of the virtual point.
        """
        if len(storage_trigger) < length:
            return False, storage_trigger, 1
        
        storage_trigger = storage_trigger[-length:]
        dimension = np.shape(storage_trigger)
        media_coordinates_fingers = np.mean(storage_trigger, axis=0).reshape(int(dimension[1] / 2), 2)
        std_fingers_xy = np.std(media_coordinates_fingers, axis=0)
        dist_virtual_point = np.sqrt((std_fingers_xy[0] ** 2) + (std_fingers_xy[1] ** 2))
        
        if dist_virtual_point < dist:
            return True, storage_trigger[-1:], dist_virtual_point
        else:
            return False, storage_trigger[-length:], dist_virtual_point
    
    @staticmethod
    def calculate_pca(data: np.ndarray, n_component: int = 3, show: bool = True) -> Tuple[PCA, np.ndarray]:
        """
        Calculates Principal Component Analysis (PCA) on a given dataset.

        Args:
            data (np.ndarray): Dataset for PCA analysis.
            n_component (int, optional): Number of principal components to be retained. Defaults to 3.
            show (bool, optional): Whether to display output. Defaults to True.

        Returns:
            Tuple[PCA, np.ndarray]: PCA model object and covariance matrix.
        """
        pca_model = PCA(n_components=n_component)
        pca_model.fit(data)
        covariance_matrix = pca_model.get_covariance()
        print(f"\n\n\nCumulative explained variance: {pca_model.explained_variance_} Explained variance per PC: {pca_model.explained_variance_ratio_}")
        return pca_model, covariance_matrix
