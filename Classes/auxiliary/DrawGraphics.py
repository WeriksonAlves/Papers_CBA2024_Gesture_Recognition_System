import matplotlib.pyplot as plt
import numpy as np

from typing import Type
from sklearn.decomposition import PCA

class DrawGraphics:
    @staticmethod
    def tracked_xyz(storage_pose_tracked: np.ndarray):
        """
        Plot the smoothed trajectory of tracked points using their X and Y coordinates.
        
        Args:
            storage_pose_tracked (np.ndarray): Array containing the smoothed trajectory of tracked points.
        """
        plt.figure(figsize=(8, 6))
        plt.title('Smoothed Trajectory of Tracked Points')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.grid(True)
        plt.plot(storage_pose_tracked[:, 0], storage_pose_tracked[:, 1], color='red', marker='o', linestyle='-')
        plt.plot(storage_pose_tracked[:, 3], storage_pose_tracked[:, 4], color='blue', marker='x', linestyle='-')
        plt.grid(True)
        plt.show()

    @staticmethod
    def results_pca(pca_model: Type[PCA], n_component: int = 3):
        """
        Visualize the explained variance ratio per principal component using a bar plot and cumulative step plot.
        
        Args:
            pca_model: Instance of a Principal Component Analysis (PCA) model.
            n_component (int, optional): Number of principal components to display. Defaults to 3.
        """
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, n_component + 1), pca_model.explained_variance_ratio_, alpha=0.5, align='center')
        plt.step(range(1, n_component + 1), np.cumsum(pca_model.explained_variance_ratio_), where='mid')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Per Principal Component')
        plt.show()
