from sklearn.neighbors import KNeighborsClassifier
from .interfaces import InterfaceClassifier
from ..auxiliary.TimeFunctions import TimeFunctions


import numpy as np

class KNN(InterfaceClassifier):
    def __init__(self, initializer: KNeighborsClassifier):
        self.neigh = initializer
    
    @TimeFunctions.run_timer
    def fit(self, X_train: np.ndarray, Y_train: list) -> None:
        """
        Fit a KNN model using the input training data X_train and corresponding target labels Y_train.
        
        Args:
            X_train (np.ndarray): The input training data.
            Y_train (list): The corresponding target labels.
        
        Returns:
            None
        """
        self.neigh.fit(X_train, Y_train)
    
    def my_predict(self, reduced_data: np.ndarray, prob_min: float = 0.6) -> str:
        """
        Predict the class label for a given sample based on a minimum probability threshold.
        
        Args:
            reduced_data (np.ndarray): The input sample.
            prob_min (float): The minimum probability threshold.
        
        Returns:
            str: The predicted class label.
        """
        reduced_data = np.array(reduced_data).flatten().tolist() 
        
        probability = self.neigh.predict_proba(np.array([reduced_data]))
        if max(probability[0]) > prob_min:
            Y_predict = self.neigh.predict(np.array([reduced_data]))[0]
        else:
            Y_predict = 'Z'
        return Y_predict
    
    @TimeFunctions.run_timer
    def validate(self, x_val: np.ndarray) -> tuple[list[str], list[float]]:
        t_func = TimeFunctions()
        y_predict = []
        time_classifier = []
        for i in range(len(x_val)):
            t_ini = t_func.tic()
            y_predict.append(self.my_predict(x_val[i]))
            time_classifier.append(t_func.toc(t_ini))        
        return y_predict, time_classifier
            