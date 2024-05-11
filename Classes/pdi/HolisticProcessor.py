from .interfaces import InterfaceFeature
import mediapipe as mp
import cv2
import numpy as np

class HolisticProcessor(InterfaceFeature):
    """
    MediaPipe processor class for feature extraction.
    """
    def __init__(self, hands_model: mp.solutions.hands.Hands, pose_model: mp.solutions.pose.Pose):
        """
        Initialize MediaPipe processor.

        Args:
            hands_model (mp.solutions.hands.Hands): The hand detection model.
            pose_model (mp.solutions.pose.Pose): The pose detection model.
        """
        self.hands_model = hands_model
        self.pose_model = pose_model
    
    def find_features(self, projected_window: np.ndarray) -> tuple[mp.solutions.hands.Hands, mp.solutions.pose.Pose]:
        """
        Find features in a projected window.

        Args:
            projected_window (np.ndarray): The projected window.

        Returns:
            Tuple: A tuple containing the found hand and pose features.
        """
        hands_results = self.hands_model.process(cv2.cvtColor(projected_window, cv2.COLOR_BGR2RGB))
        pose_results = self.pose_model.process(cv2.cvtColor(projected_window, cv2.COLOR_BGR2RGB))
        return hands_results, pose_results
    
    def draw_features(self, projected_window: np.ndarray, hands_results: mp.solutions.hands.Hands, pose_results: mp.solutions.pose.Pose) -> np.ndarray:
        """
        Draw features on a projected window.

        Args:
            projected_window (np.ndarray): The projected window.
            hands_results (mp.solutions.hands.Hands): The hand detection results.
            pose_results (mp.solutions.pose.Pose): The pose detection results.
            
        Returns:
            np.ndarray: The modified projected window.
        """
        projected_window.flags.writeable = True
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    projected_window,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
        if pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                projected_window,
                pose_results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return projected_window
