import cv2
import numpy as np

class InitializeConfig:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, id: int = 0, fps: int = 5, dist: float = 0.025, length: int = 15) -> None:
        self.cap = cv2.VideoCapture(id)
        self.fps = fps
        self.dist = dist
        self.length = length

class ModeFactory:
    @staticmethod
    def create_mode(mode_type, **kwargs):
        """
        The function `create_mode` dynamically creates instances of different mode classes based on the
        specified `mode_type`.
        """
        if mode_type == 'dataset':
            return ModeDataset(**kwargs)
        elif mode_type == 'validate':
            return ModeValidate(**kwargs)
        elif mode_type == 'real_time':
            return ModeRealTime(**kwargs)
        else:
            raise ValueError("Invalid mode type")

class ModeDataset:
    def __init__(self, database: dict[str, list], file_name_build: str, max_num_gest: int = 50, 
                    dist: float = 0.025, length: int = 15) -> None:
        """
        This function initializes an object with specified parameters including a database, file name,
        maximum number of gestures, distance, and length.
        """
        self.mode = 'D'
        self.database = database
        self.file_name_build = file_name_build
        self.max_num_gest = max_num_gest
        self.dist = dist
        self.length = length

class ModeValidate:
    def __init__(self, files_name: list[str], database: dict[str, list], name_val: str,                    
                    proportion: float = 0.7, n_class: int = 5, n_sample_class: int = 10) -> None:
        """
        This function initializes various attributes including file names, database, proportion, and
        calculates a value based on input parameters.
        """
        self.mode = 'V'
        self.files_name = files_name
        self.database = database
        self.proportion = proportion
        self.k = int(np.round(np.sqrt(int(len(self.files_name) * self.proportion * n_class * n_sample_class))))
        self.file_name_val = self.rename(n_class, n_sample_class, name_val)

    def rename(self, n_class, n_sample_class, name_val):
        """
        The `rename` function generates a file name based on input parameters such as class, sample
        size, proportion, and a custom name value.
        """
        c = n_class
        s = int(len(self.files_name) * (1 - self.proportion) * n_class * n_sample_class)
        ma_p = int(10 * self.proportion)
        me_p = int(10 * (1 - self.proportion))
        return f"Results\C{c}_S{s}_p{ma_p}{me_p}_k{self.k}_{name_val}"

class ModeRealTime:
    def __init__(self, files_name: list[str], database: dict[str, list], proportion: float = 0.7,            
                    n_class: int = 5, n_sample_class: int = 10) -> None:
        """
        This function initializes an object with specified parameters for files, database, proportion,
        number of classes, and number of samples per class.
        """
        self.mode = 'RT'
        self.files_name = files_name
        self.database = database
        self.proportion = proportion
        self.k = int(np.round(np.sqrt(int(len(self.files_name) * self.proportion * n_class * n_sample_class))))
