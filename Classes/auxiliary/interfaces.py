from abc import ABC, abstractmethod

import numpy as np

class InterfaceTime(ABC):
    @abstractmethod
    def tic(self) -> float:
        """Return the current time."""
        pass

    @abstractmethod
    def toc(self, start_time: float) -> float:
        """Return the elapsed time since a given starting time."""
        pass


class InterfaceFile(ABC):
    @abstractmethod
    def initialize_database(self, database: dict) -> tuple[list[str], np.ndarray]:
        """Initialize the database."""
        pass

    @abstractmethod
    def save_database(self, sample: dict, database: dict, file_path: str):
        """Save sample dictionary to a specified file path within a database dictionary."""
        pass

    @abstractmethod
    def load_database(self, current_folder: str, files_name: list[str], proportion: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load the database from files."""
        pass

    @abstractmethod
    def save_results(self, y_true: list[str], y_predict: list[str], target_names: list[str],  file_path_val: str):
        """Save true and predicted values to a specified file path."""
        pass

