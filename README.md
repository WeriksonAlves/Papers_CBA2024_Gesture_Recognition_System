# CBA2024

## Title: 

**A Gesture Recognition System for Robot Guidance Applications**

The gesture classification system is a software application designed to analyze and interpret human gestures captured through images or videos. Its purpose is to identify and categorize specific gestures, such as hand movements or body poses, into predefined classes or categories. This system finds applications in various fields, including human-computer interaction, sign language recognition, and motion analysis. By accurately recognizing and classifying gestures, the system enables intuitive and natural interaction between humans and machines, facilitating tasks such as controlling devices, interpreting sign language, and analyzing human behavior.

## Overview of Gesture Classification System

1. **Data Acquisition**:
   - Involves capturing images or videos of gestures using cameras or sensors.
   - Images or video frames are collected in real-time (for live classification) or from existing datasets.

2. **Feature Extraction**:
   - Extracting relevant features from the captured data that characterize different gestures.
   - Features could include hand positions, angles, movement trajectories, or any other distinguishing characteristics.

3. **Pre-processing**:
   - Cleaning and preparing the data for classification.
   - This may involve resizing images, normalizing pixel values, removing noise, or augmenting the dataset to improve model performance.

4. **Data Classification**:
   - Utilizing a machine learning model to classify gestures based on the extracted features.
   - Common approaches include supervised learning algorithms such as Support Vector Machines (SVM), Convolutional Neural Networks (CNN), or Recurrent Neural Networks (RNN).

5. **Saved Data Management**:
   - Handling and storing the results of gesture classification.
   - This may involve saving classification results to a database, logging them to files, or visualizing them for further analysis.

By following these steps, the gesture classification system can accurately recognize and classify various gestures, enabling applications such as gesture-based control systems, sign language recognition, and human-computer interaction.

## Installation Instructions

To set up the gesture classification system, you need to install the necessary Python libraries and dependencies used in the system. You can do this easily using `pip`, which is Python's package installer.

To install the dependencies from the `requirements.txt` file, navigate to the project directory in the terminal or at the command prompt and run the following command:

   ```
   pip install -r requirements.txt
   ```

This will automatically install all the libraries listed in the requirements.txt file. By following these simple steps, you will have all the libraries and dependencies you need to run the gesture classification system on your system.

## Usage Instructions

The gesture classification system can be used to recognize and classify gestures in various scenarios. Here's how you can use the system effectively:

1. **Running the System with Different Modes:**
   - The system supports three different modes of operation:
     - **Dataset (D)**: To create a dataset to be used as a reference for gestures during classification.
     - **Real-Time (RT)**: For performing gesture classification in real-time from a live camera feed.
     - **Validate (V)**: For validating the performance of the classifier on a validation dataset.

2. **Configuring Parameters and Options:**
    - Depending on the mode of operation, you may need to configure different parameters and options:
        - For Dataset mode, define the classes that will be used and the name that will be used to save the dataset. Other parameters are optional.
        - For Real-Time mode, define the names of the reference data and the classes that must be used. Other parameters are optional.
        - For Validate mode, define the names of the reference data, the classes that will be used and the name that will be used to save the results. Other parameters are optional.
    
    - Next, initialize the 'GestureRecognitionSystem()' class, passing the parameters needed to run the system. And execute the `run()` method.

3. **Example Code Snippets or Usage Scenarios:**
   - Below are some example code snippets demonstrating how to use the gesture classification system:

     ```python
     database = {'F': [], 'I': [], 'L': [], 'P': [], 'T': []}
     file_name_build = f"path\name_file.json"
     files_name= ["path\name_file1.json", "path\name_file2.json", ...]
     name_val=f"val99"

     # Example usage in Dataset mode
     dataset_mode = ModeFactory.create_mode('dataset', database=database, file_name_build=file_name_build)

     # Example usage in Real-Time mode
     real_time_mode = ModeFactory.create_mode('real_time', files_name=files_name, database=database)

     # Example usage in Validation mode
     validate_mode = ModeFactory.create_mode('validate', files_name=files_name, database=database, name_val=name_val)
     ```

     ```python
     # Specify the operating mode
     mode = real_time_mode

     grs = GestureRecognitionSystem(
            config=InitializeConfig(),
            operation=mode,
            file_handler=FileHandler(),
            current_folder=os.path.dirname(__file__),
            data_processor=DataProcessor(), 
            time_functions=TimeFunctions(), 
            gesture_analyzer=GestureAnalyzer(),
            tracking_processor=YoloProcessor('yolov8n-pose.pt'), 
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
                ),
            classifier=KNN(KNeighborsClassifier( n_neighbors=mode.k, algorithm='auto', weights='uniform'))
            # or classifier=None
            )
     grs.run()
     ```

   - Warning: if you are using the database creation method, remember to remove the 'classifier' parameter or pass it as `None'.
   - Ensure that the necessary libraries and dependencies are installed before running the system.

By following these instructions and examples, you can effectively use the gesture classification system to recognize and classify gestures in your applications. Adjust the configurations and parameters as needed to suit your specific requirements and use cases.

## Examples

As an example of how to use the classification system, you can use the dataset available in the "Dataset" folder for Real-Time mode. To do this, run the main.py code. To initiate the gesture, bring the fingertips of the tracked hand together until their distance is less than 0.025 (this value is displayed in the real-time window). The gestures included in the database are shown in the figure below.

![alt text](Paper/Images/gesture_class.png)

Please note that the main.py script must be run without any changes.

*NOTE*: The preview window also indicates the stages of the action (S0 for awaiting trigger and S1 for storing gesture information) and shows the number of actions performed so far.

## Contact Information

For more information, any questions or suggestions for improvement, please get in touch.
