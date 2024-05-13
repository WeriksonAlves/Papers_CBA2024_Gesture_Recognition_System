from .interfaces import InterfaceTrack
from ultralytics import YOLO
from collections import defaultdict

import numpy as np
import cv2

class YoloProcessor(InterfaceTrack):
    """
    YOLO processor class for tracking.
    """
    def __init__(self, file_YOLO: str):
        """
        Initialize YOLO processor.

        Args:
            yolo_model (YOLO): The YOLO model.
        """
        self.yolo_model = YOLO(file_YOLO)
        self.track_history = defaultdict(list)
    
    def find_people(self, captured_frame: np.ndarray, persist: bool = True, verbose: bool = False) -> list:
        """
        Find people in captured frame.

        Args:
            captured_frame (np.ndarray): The captured frame.
            persist (bool): Whether to persist the results.
            verbose (bool): Whether to output verbose information.

        Returns:
            List: A list of people found in the frame.
        """
        return self.yolo_model.track(captured_frame, persist=persist, verbose=verbose)
    
    def identify_operator(self, results_people: list) -> np.ndarray:
        """
        Identify operator.

        Args:
            results_people: Results of people found in the frame.
            
        Returns:
            np.ndarray: Image results.
        """
        return results_people[0].plot()
    
    def track_operator(self, results_people: list, results_identifies: np.ndarray, captured_frame: np.ndarray, length: int = 90) -> np.ndarray:
        """
        Track operator.

        Args:
            results_people: Results of people found in the frame.
            results_identifies (np.ndarray): Image results.
            captured_frame (np.ndarray): The captured frame.
            length (int): Length of the track.

        Returns:
            np.ndarray: Tracked operator.
        """
        boxes = results_people[0].boxes.xywh.cpu()
        track_ids = results_people[0].boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = map(int, box)
            track = self.track_history[track_id]
            track.append((x + w // 2, y + h // 2))
            track = track[-length:]
            points = np.vstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(results_identifies, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            person_roi = captured_frame[(y - h // 2):(y + h // 2), (x - w // 2):(x + w // 2)]
            break
        return cv2.flip(person_roi, 1)
