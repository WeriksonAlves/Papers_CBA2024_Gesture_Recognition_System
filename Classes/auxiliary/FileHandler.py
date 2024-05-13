from .interfaces import InterfaceFile
from .TimeFunctions import TimeFunctions
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import numpy as np
import json
import os
import matplotlib.pyplot as plt

class FileHandler(InterfaceFile):
    def initialize_database(self, database: dict) -> tuple[list[str], np.ndarray]:
        """
        Initialize the database and return a list of gesture classes and a NumPy array of true labels.
        
        Args:
            database (dict): Dictionary representing the database with gesture classes as keys 
                            and data for each class as values.
        
        Returns:
            tuple: A tuple containing a list of gesture classes and a NumPy array of true labels.
        """
        target_names = list(database.keys()) + ['Z']
        y_true = np.array(['I'] * 10 + ['L'] * 10 + ['F'] * 10 + ['T'] * 10 + ['P'] * 10)
        print(f"\n\nVoltar aqui: file_manager.py\n\n")
        return target_names, y_true

    def save_database(self, sample: dict, database: dict, file_path: str):
        """
        Save a database to a file in JSON format after converting certain values to lists.
        
        Args:
            sample (dict): Data sample to be saved to the database.
            database (dict): Database dictionary where the data will be saved.
            file_path (str): File path where the database will be saved as a JSON file.
        """
        sample['data_pose_track'] = sample['data_pose_track'].tolist()
        sample['data_reduce_dim'] = sample['data_reduce_dim'].tolist()
    
        with open(file_path, "w") as file:
            json.dump(database, file)
    
    @TimeFunctions.run_timer
    def load_database(self, current_folder: str, files_name: list[str], proportion: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from multiple files, split it into training and validation sets, and calculate 
        average collection time per class.
        
        Args:
            current_folder (str): Path to the folder where the database files are located.
            files_name (list): List of file names to be loaded.
            proportion (float): Ratio of data samples for training compared to validation.
        
        Returns:
            tuple: A tuple containing training and validation data arrays.
        """
        X_train, Y_train, X_val, Y_val = [], [], [], []
        time_reg = np.zeros(5)

        for file_name in files_name:
            file_path = os.path.join(current_folder, file_name)

            with open(file_path, "r") as file:
                database = json.load(file)

            g = 0
            for _, value in database.items():
                np.random.shuffle(value)
                num = 0
                if g == 5:
                    g = 0
                for sample in value:
                    if num < int(proportion * len(value)):
                        X_train.append(np.array(sample['data_reduce_dim']).flatten().tolist())
                        Y_train.append(sample['answer_predict'])
                    else:
                        X_val.append(np.array(sample['data_reduce_dim']).flatten().tolist())
                        Y_val.append(sample['answer_predict'])
                    num += 1
                time_reg[g] += sample['time_gest']
                g += 1
        
        X_train, X_val = np.array(X_train), np.array(X_val)
        
        print(f"\n\nTraining => Samples: {X_train.shape} Class: {len(Y_train)} \nValidation => Samples: {X_val.shape} Class: {len(Y_val)}")
        length = len(X_train) / 5 + len(X_val) / 5
        print(f"Average collection time per class: {time_reg[0] / length:.3f}, {time_reg[1] / length:.3f}, {time_reg[2] / length:.3f}, {time_reg[3] / length:.3f}, {time_reg[4] / length:.3f}\n\n")
        
        return X_train, np.array(Y_train), X_val, np.array(Y_val)
    
    @TimeFunctions.run_timer
    def save_results(self, y_true: list[str], y_predict: list[str], time_classifier: list[float], target_names: list[str],  file_path_val: str):
        """
        Save true and predicted values to a specified file path.
        
        Args:
            y_true (list): List of true labels.
            y_predict (list): List of predicted labels.
            target_names (list): List of target names for classification.
            file_path_val (str): File path where the results will be saved.
        """
        results_validate = np.array([y_true, y_predict, time_classifier]).tolist()

        with open(file_path_val, "w") as file:
            json.dump(results_validate, file)

        cm_pc = confusion_matrix(y_true, y_predict, labels=target_names, normalize='true')
        disp_pc = ConfusionMatrixDisplay(confusion_matrix=cm_pc, display_labels=target_names)
        disp_pc.plot()
        plt.savefig(file_path_val+'_pc.jpg')

        cm_abs = confusion_matrix(y_true, y_predict, labels=target_names)
        disp_abs = ConfusionMatrixDisplay(confusion_matrix=cm_abs, display_labels=target_names)
        disp_abs.plot()
        plt.savefig(file_path_val+'_abs.jpg')

        print(classification_report(y_true, y_predict, target_names=target_names, zero_division=0))
