"""
Research Line: Classification of gestures for vehicle control in inspection routines
Author: Wérikson Alves
Initial Date: 25/04/2024  => Final Date: /07/2024

...............................................................................................
Description
    Operation mode:
        Build:     Creates a new database and saves it in json format
        Recognize: Load the database, create the classifier and classify the actions

    Operation stage:
        0 - Processes the image and analyzes the operator's hand
        1 - Processes the image and analyzes the operator's body
        2 - Reduces the dimensionality of the data
        3 - Updates and save the database
        4 - Performs classification from kMeans
...............................................................................................
""" 

from GestureRecognitionSystem import *

import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier


files_name= ['Datasets\DataBase_(5-10)_Guilherme.json',
            'Datasets\DataBase_(5-10)_Hiago.json',
            'Datasets\DataBase_(5-10)_Lucas.json',
            'Datasets\DataBase_(5-10)_Mateus.json',
            'Datasets\DataBase_(5-10)_Thayron.json',
            'Datasets\DataBase_(5-10)_Werikson_1.json',
            'Datasets\DataBase_(5-10)_Werikson_2.json',
            'Datasets\DataBase_(5-10)_Werikson_3.json',
            'Datasets\DataBase_(5-10)_Werikson_4.json',
            'Datasets\DataBase_(5-10)_Werikson_5.json',
            'Datasets\DataBase_(5-10)_Werikson_6.json',
            'Datasets\DataBase_(5-10)_Werikson_7.json',
            'Datasets\DataBase_(5-10)_Werikson_8.json',
            'Datasets\DataBase_(5-10)_Werikson_9.json',
            'Datasets\DataBase_(5-10)_Werikson_10.json'
            ]
database = {'F': [], 'I': [], 'L': [], 'P': [], 'T': []}
name_val=f"val99"

# mode = ModeRealTime(files_name, database)
mode = ModeValidate(files_name, database, name_val)
if __name__ == "__main__":
    grs = GestureRecognitionSystem(
        config=InitializeConfig(),
        operation=mode,
        tracking_processor=YoloProcessor('yolov8n-pose.pt'), 
        file_handler=FileHandler(),
        data_processor=DataProcessor(), 
        time_functions=TimeFunctions(), 
        gesture_analyzer=GestureAnalyzer(),
        classifier=KNN(
            KNeighborsClassifier(
                n_neighbors=mode.k, 
                algorithm='auto', 
                weights='uniform'
                )
            ),
        feature=HolisticProcessor(
            mp.solutions.hands.Hands(
                static_image_mode=False, 
                max_num_hands=1, 
                model_complexity=1, 
                min_detection_confidence=0.75, 
                min_tracking_confidence=0.75
                ),
            mp.solutions.pose.Pose(
                static_image_mode=False, 
                model_complexity=1, 
                smooth_landmarks=True, 
                enable_segmentation=False, 
                smooth_segmentation=True, 
                min_detection_confidence=0.75, 
                min_tracking_confidence=0.75
                )
            )
        )
    
    grs.run()

