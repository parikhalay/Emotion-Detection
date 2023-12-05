# GITHUB LINK
https://github.com/parikhalay/Emotion-Detection

# DATASET LINK
https://drive.google.com/drive/folders/1wOocyriwibxSKxcRlbfkaWyLlso7r3Rb

# A.I.ducation Analytics Project Readme

In this project, we are analysing student facial responses thus shaping the future of AI-driven education. Our innovative system empowers educators with insights to create dynamic and engaging learning experiences.

## Project Introduction

As AI lectures present complex algorithms, instructors are no longer in the dark about student engagement. Our system examines students' facial responses in real-time, distinguishing the curious from the overwhelmed. A sleek desing of the system offers instructors immediate insights, such as engaged, neutral, bored etc. Smart AI suggestions in real-time ensures lectures evolve to meet learners' needs. As graduate students on this pioneering project, we are not just coding; we are sculpting the next phase of dynamic, AI-enhanced education.

## Project Objective

The primary objective of this project is to develop a Deep Learning Convolutional Neural Network (CNN) using PyTorch that can analyze images of students in a classroom or online meeting setting and categorize them into distinct states or activities. The system should be capable of recognizing four classes:

1. **Neutral**: Students displaying relaxed facial features with neither active engagement nor disengagement.
2. **Focused**: Students demonstrating signs of active concentration with sharp and attentive eyes.
3. **Tired**: Students exhibiting signs of weariness or a lack of interest, possibly with droopy eyes or vacant stares.
4. **Angry**: Signs of agitation or displeasure, which might manifest as tightened facial muscles, a tight-lipped frown, or narrowed eyes.

## Project Part I: Data Collection and Preparation

### Training Data

In this part, we collected suitable training data and performed exploratory data analysis (EDA). We followed these guidelines:

- **Training Data**: Created datasets for training and testing our AI system. Provided provenance information about the source of each image in  dataset. We reused existing datasets, but we ensured proper referencing.
- **Data Size**: We have a minimum of 1500 training images and 500 testing images (across all classes), totaling a minimum of 2000 images for the four classes, before applying data augmentation strategies. Ensured balanced datasets, with roughly the same number of images per class.
- **Used Real Data**: Used real training data; neither synthetic nor generated data.

### Data Cleaning

Images can vary in sizes, resolutions, or lighting conditions. We standardized the dataset by:

- Resizing images to a consistent dimension.
- Applying light processing for increased robustness (e.g., slight rotations, brightness adjustments, minor cropping).

### Labeling

If datasets are not pre-labeled or if there's ambiguity, manual labeling is required. So we mapped single or multiple classes from different datasets to suitable training classes for our system. We considerd using platforms like Labelbox for assistance.

### Dataset Visualization

Visualized dataset to ensure an even class distribution and to understand the data's nature. This is crucial before diving into model training, as imbalanced datasets can affect model performance. We used Matplotlib to show:

- Class distribution.
- A few sample images from different classes.
- Pixel intensity distribution for the images.

Gaining these insights early on will allowed us to make informed decisions about any additional preprocessing or cleaning that the dataset might require.

## Get Started

## Purpose of Each File

1. **dataCleaning.py**: This Python script is responsible for cleaning and preprocessing image data. It resizes images to a consistent dimension, applies brightness adjustments, and saves the processed images. The purpose is to prepare the data for machine learning models.

2. **dataVisualisation.py**: This script is used for data visualization. It creates a class distribution bar graph and displays a grid of sample images from different classes, along with their pixel intensity distribution. This step is crucial to understand the dataset and its class distribution before model training.

3. **pngTojpg.py**: This script is designed to convert PNG image files to JPG format. It iterates through directories, identifies PNG files, converts them to JPG, and replaces the original PNG files.

4. **datasetShrinking.py**: This script selects and copies a specific number of random images from a source directory to a target directory. It's useful for creating smaller subsets of a dataset for testing and experimentation.

## Data Cleaning

**dataCleaning.py** performs the following data cleaning tasks:

1. Resizes images to a consistent dimension (100x100 pixels) while preserving the aspect ratio and quality of the Images.
2. Applies brightness adjustments to images.
3. Standardizes the dataset to ensure uniformity in terms of size and quality.

To execute the data cleaning process:

**Step 1**: Set up the environment
Ensured the required libraries are installed. We needed OpenCV (cv2) and NumPy. Used Used `pip` to install them as follows:

```bash
pip install opencv-python-headless numpy
```

**Step 2**: Organized the directory structure
Made sure the dataset is organized in a directory structure similar to the one expected by the code. Specifically, we have two directories: ../dataset/train/ (input directory) and ../dataCleaning/train/ (output directory). The code will process images from the input directory and save the processed images in the output directory.

**Step 3**: Executed the code
Ran the Python script by running:

```bash
python dataCleaning.py
```

This will start the data cleaning process. The code will loop through the subdirectories in the input directory, resize the images to the specified target dimensions, apply brightness adjustments, and save the processed images in the output directory.

After executing these steps dataset will be standardized, with all images resized to a uniform dimension and brightness adjustments applied. The processed images will be stored in the output directory for further use in the project.

## Data Visualization

**dataVisualization.py** is used for visualizing the dataset and provides insights into the data distribution. It includes the following visualizations:

1. Class distribution bar graph.
2. A 5x5 grid of random sample images from different classes.
3. Pixel intensity distribution in RGB channels for the sample images.

To execute the data visualization:

**Step 1:** Set up the environment
Ensured that the required libraries are installed. We will need OpenCV (cv2), NumPy, and Matplotlib. Used `pip` to install them:

```bash
pip install opencv-python-headless numpy matplotlib
```

**Step 2:** Organized  directory structure
Made sure we have already executed the data cleaning code and have our dataset ready in the expected directory structure. Because code assumes that we have the standardized dataset in the `../dataCleaning/train/` directory.

**Step 3:** Executed the code
Ran the Python script:

```bash
python dataVisualization.py
```

This will start the data visualization process. The code will generate a bar graph showing the class distribution, display 25 random sample images from different classes, and plot intensity distributions for these images.

After executing these steps, we will have a better understanding of your dataset's class distribution, content, and pixel intensity variations. The visualizations can help us identify any imbalances or anomalies in our dataset, which is valuable for further analysis and model training.


## Data Processing done before Data Cleaning
**NOTE**: Below processes are already done no need to do it. It's just for reference.
### Converting PNG Files to JPG Files

**pngTojpg.py** is designed to convert PNG images to JPG format. It iterates through directories, identifies PNG files, converts them to JPG, and replaces the original PNG files.

To execute the PNG to JPG conversion:

```bash
python pngTojpg.py.py
```

### Data Shrinking

**datasetShrinking.py** is used for selecting and copying a specific number of random images from a source directory to a target directory. This is useful for creating smaller subsets of a dataset for testing and experimentation.

To execute the data shrinking process:

```bash
python datasetShrinking.py
```

# Develop and Load a CNN model
### Main Model Code (training_model.py):

This script in Python is designed for the Main Model's training process.
It encompasses defining the model's structure, handling data, executing the training and validation phases, and implementing early stopping mechanisms.

### Model Evaluation Script (test_model.py):
A dedicated Python script for evaluating the performance of the trained model on a test dataset. It includes functionality for loading the model, running it on the test data, and computing evaluation metrics like accuracy, precision, recall, and F1-score.

### Load Model Code (loadCNN.py):

This Python script is tasked with loading a previously trained model for making predictions on new or individual images.
It includes functionalities for model loading, batch prediction, and individual image prediction.

### Variant 1 Code (variant1CL.py):

This script is responsible for training the Variant 1 Model.
It follows a similar structure to the Main Model but incorporates an extra convolutional layer to detect more intricate features.

### Variant 2 Code (variant2KS.py):

This script facilitates the training of the Variant 2 Model.
It varies the kernel sizes in convolutional layers to explore the identification of broader facial features.

## Procedure for Executing the Code
### For Model Training

1. Install required libraries: numpy, torch, torchvision, matplotlib, sklearn, seaborn.

2. Place your training data in the specified directory as per the data_path in the script.

3. Execute the script using a Python interpreter, for example:

```bash
python training_model.py
```

4. The script will train the model across several epochs, with early stopping if validation loss does not improve.

5. It will save best model with lowest validation loss and a final model e.g., facial_recognition_final_model.pth.

### For Model Evaluation
1. Use the test_model.py script for evaluating the trained model.

2. Ensure the model file and test dataset are correctly placed as specified in the script.

3. Run the evaluation script:

```bash
python test_model.py
```
4. The script will load the model, apply it to the test data, and output evaluation metrics.

### For Predictions Using a Trained Model

1. Install Necessary Libraries: Make sure you have Python installed along with the required libraries: PyTorch, torchvision, PIL (Python Imaging Library), numpy, and Tkinter. These can be installed via pip or conda.

2. Place the Model File: Ensure the trained model file facial_recognition_best_model.pth is located in the savemodel directory relative to the script.

3. Launch the App: Run the Python script to start the application:
   
```bash
python loadCNN.py
```

4. Use the App:

   1. Once the application window opens, you will see an input field.
   2. Enter the path of the image you want to classify or use the file dialog to select an image.
   3. Press Submit. The application will display the selected image and its predicted class, along with the true class based on the image's file path.
   4. The application uses the CNN model to classify the image into one of the four facial expressions.

### K-Fold Cross-Validation
1. Install Required Libraries: Ensure all necessary Python libraries (PyTorch, torchvision, NumPy, scikit-learn) are installed.

2. Prepare Your Dataset: Place your dataset in the dataset/datacleaning/train directory. The dataset should be in a format compatible with torchvision.datasets.ImageFolder.

3. Run the Script: Execute the script using Python:

```bash
python k_fold_validation.py
```

4. Script Execution:

   1. The script will perform K-fold cross-validation on the dataset, training a new model for each fold.
   2. The early stopping mechanism will prevent overfitting by halting training if validation loss doesn't improve.
   3. After each fold, the script will save the model's state and print the performance metrics.

5. Results:

   1. Performance metrics for each fold, including accuracy, macro and micro precision, recall, and F1 score, will be printed.
   2. A summary of the average performance across all folds is calculated and displayed.
   3. These results are also saved to a CSV file named P3_k_validation_results.csv.

6. The script outputs a CSV file with detailed performance metrics for each fold and the averages across all folds. This file helps in assessing the model's performance and generalization capability.

## Image Labeling Tool
### Overview
This script, labelling.py, located in the label folder, provides a user interface for labeling images based on gender and age categories. It uses OpenCV for face detection and a trained model to suggest initial labels, which users can then manually adjust.

### NOTE
Avoid faceBox.py file

### Steps to Run the Labeling Tool
1. Install Required Libraries: Ensure Python is installed along with the OpenCV, Tkinter, and PIL libraries. These can be installed via pip.

2. Model and Proto Files: Place the OpenCV model files (opencv_face_detector_uint8.pb, age_net.caffemodel, gender_net.caffemodel) and their corresponding proto files (opencv_face_detector.pbtxt, age_deploy.prototxt, gender_deploy.prototxt) in the same directory as the script.

3. Prepare the Dataset: Organize your images in the ../dataset/datasplit/test directory, structured with category subfolders.

4. Run the Script:

```bash
python labelling.py
```

5. Using the Tool:

   1. The tool's GUI will display each image, along with detected gender and age range.
   2. Adjust the gender (m/f) and age category (1 for young, 2 for middle-aged, 3 for senior) as needed.
   3. Click Submit or press Enter to save your labels and move to the next image.
   4. The tool saves your progress, so you can resume labeling later if needed.

6. Output:

   1. Labels are saved in a CSV file labels.csv with columns: Image Path, Gender, Age Category.
   2. The tool also creates a progress.txt file to track the last processed image for resuming work.

### Features of the Script
* Face Detection: Uses OpenCV's DNN module to detect faces in images.
* Suggested Labels: Automatically suggests gender and age labels based on the models.
* Manual Input: Allows for manual correction of suggested labels.
* Progress Tracking: Keeps track of the last labeled image, allowing for easy resumption.

## Bias Analysis Tool
### Overview
This Python script is designed for facial expression recognition using a Convolutional Neural Network (CNN) and performs a bias analysis across different age and gender groups. It evaluates the model's performance in terms of accuracy, precision, recall, and F1-score for each group.

### Steps to Run the Script
1. Install Required Libraries: Ensure Python and the necessary libraries (PyTorch, pandas, Matplotlib, torchvision, scikit-learn) are installed.

2. Prepare Model and Data:

   1. Place the pre-trained model file facial_recognition_best_model.pth in the savemodel directory.
   2. Ensure the CSV file Part3_test_labels.csv with image paths and labels is available.

3. Run the Script:

```bash
python bias_analysis.py
```

4. Script Execution:

   1. The script will load the CNN model and the dataset from the CSV file.
   2. It performs bias analysis by evaluating the model separately on different age and gender groups.
   3. The performance metrics for each group, as well as the overall average, are calculated and printed.

5. Output:

   1. The results of the bias analysis are saved in a CSV file P3_bias_analysis_results.csv.
   2. The CSV file includes columns for Attribute (Age/Gender), Group, Accuracy, Precision, Recall, and F1-Score.
   3. Printed results provide a quick overview of the model's performance across different groups.

### Features of the Script
* Group-Specific Evaluation: Analyzes the model's performance for different demographic groups.
* Performance Metrics: Calculates and reports accuracy, precision, recall, and F1-score.
* Result Export: Outputs the analysis results to a CSV file for further examination.
