
# A.I.ducation Analytics Project Readme

Welcome to **A.I.ducation Analytics**, where we embark on the journey of revolutionizing academic feedback using Artificial Intelligence. In this project, we are shaping the future of AI-driven education with real-time analysis of student facial responses. Our innovative system empowers educators with insights to create dynamic and engaging learning experiences.

## Project Introduction

As AI lectures present complex algorithms, instructors are no longer in the dark about student engagement. Our system scrutinizes students' facial responses in real-time, distinguishing the curious from the overwhelmed. A sleek dashboard offers instructors immediate insights, such as 30% engaged, 20% neutral, and 10% nearing cognitive overload. Smart AI suggestions nudge adjustments in real-time, ensuring lectures evolve to meet learners' needs. As graduate students on this pioneering project, we are not just coding; we are sculpting the next phase of dynamic, AI-enhanced education.

## Project Objective

The primary objective of this project is to develop a Deep Learning Convolutional Neural Network (CNN) using PyTorch that can analyze images of students in a classroom or online meeting setting and categorize them into distinct states or activities. The system should be capable of recognizing four classes:

1. **Neutral**: Students displaying relaxed facial features with neither active engagement nor disengagement.
2. **Focused**: Students demonstrating signs of active concentration with sharp and attentive eyes.
3. **Tired**: Students exhibiting signs of weariness or a lack of interest, possibly with droopy eyes or vacant stares.
4. **Angry**: Signs of agitation or displeasure, which might manifest as tightened facial muscles, a tight-lipped frown, or narrowed eyes.

## Project Part I: Data Collection and Preparation

### Training Data

In this part, you will collect suitable training data and perform exploratory data analysis (EDA). Follow these guidelines:

- **Training Data**: Create datasets for training and testing your AI. Provide provenance information about the source of each image in your dataset. Reuse existing datasets, but ensure proper referencing.
- **Data Size**: Have a minimum of 1500 training images and 500 testing images (across all classes), totaling a minimum of 2000 images for the four classes, before applying data augmentation strategies. Ensure balanced datasets, with roughly the same number of images per class.
- **Use Real Data**: Use real training data; synthetic or generated data is not permitted.

### Data Cleaning

Images can vary in sizes, resolutions, or lighting conditions. Standardize the dataset by:

- Resizing images to a consistent dimension.
- Applying light processing for increased robustness (e.g., slight rotations, brightness adjustments, minor cropping).

### Labeling

If datasets are not pre-labeled or if there's ambiguity, manual labeling may be required. Map single or multiple classes from different datasets to suitable training classes for your system. Consider using platforms like Labelbox for assistance.

### Dataset Visualization

Visualize your dataset to ensure an even class distribution and to understand the data's nature. This is crucial before diving into model training, as imbalanced datasets can affect model performance. Use Matplotlib to show:

- Class distribution.
- A few sample images from different classes.
- Pixel intensity distribution for the images.

Gaining these insights early on will allow you to make informed decisions about any additional preprocessing or cleaning that your dataset might require.

## Get Started

## Purpose of Each File

1. **DataCleaning.py**: This Python script is responsible for cleaning and preprocessing image data. It resizes images to a consistent dimension, applies brightness adjustments, and saves the processed images. The purpose is to prepare the data for machine learning models.

2. **DataVisualisation.py**: This script is used for data visualization. It creates a class distribution bar graph and displays a grid of sample images from different classes, along with their pixel intensity distribution. This step is crucial to understand the dataset and its class distribution before model training.

3. **ConvertingPNGFilesToJPGFiles.py**: This script is designed to convert PNG image files to JPG format. It iterates through directories, identifies PNG files, converts them to JPG, and replaces the original PNG files.

4. **DataShrinking.py**: This script selects and copies a specific number of random images from a source directory to a target directory. It's useful for creating smaller subsets of a dataset for testing and experimentation.

## Data Cleaning

**DataCleaning.py** performs the following data cleaning tasks:

1. Resizes images to a consistent dimension (100x100 pixels) while preserving the aspect ratio and quality of the Images.
2. Applies brightness adjustments to images.
3. Standardizes the dataset to ensure uniformity in terms of size and quality.

To execute the data cleaning process:

**Step 1**: Set up your environment
Ensure that you have the required libraries installed. You'll need OpenCV (cv2) and NumPy. If you don't have them installed, you can use `pip` to install them:

```bash
pip install opencv-python-headless numpy
```

**Step 2**: Organize your directory structure
Make sure your dataset is organized in a directory structure similar to the one expected by the code. Specifically, you should have two directories: ../dataset/train/ (input directory) and ../datacleaning/train/ (output directory). The code will process images from the input directory and save the processed images in the output directory.

**Step 3**: Execute the code
Run the Python script or the notebook containing the code. You can execute it by running:

```bash
python data_cleaning.py
```

This will start the data cleaning process. The code will loop through the subdirectories in the input directory, resize the images to the specified target dimensions, apply brightness adjustments, and save the processed images in the output directory.

After executing these steps dataset will be standardized, with all images resized to a uniform dimension and brightness adjustments applied. The processed images will be stored in the output directory for further use in the project.

## Data Visualization

**DataVisualisation.py** is used for visualizing the dataset and provides insights into the data distribution. It includes the following visualizations:

1. Class distribution bar graph.
2. A 5x5 grid of random sample images from different classes.
3. Pixel intensity distribution in RGB channels for the sample images.

To execute the data visualization:

**Step 1:** Set up your environment
Ensure that you have the required libraries installed. You'll need OpenCV (cv2), NumPy, and Matplotlib. If you don't have them installed, you can use `pip` to install them:

```bash
pip install opencv-python-headless numpy matplotlib
```

**Step 2:** Organize your directory structure
Make sure you have already executed the data cleaning code and have our dataset ready in the expected directory structure. Because code assumes that you have the standardized dataset in the `../datacleaning/train/` directory.

**Step 3:** Execute the code
Run the Python script or the notebook containing the code. You can execute it by running:

```bash
python data_visualization.py
```

This will start the data visualization process. The code will generate a bar graph showing the class distribution, display 25 random sample images from different classes, and plot intensity distributions for these images.

After executing these steps, we will have a better understanding of your dataset's class distribution, content, and pixel intensity variations. The visualizations can help us identify any imbalances or anomalies in our dataset, which is valuable for further analysis and model training.


## Data Processing done before Data Cleaning
**NOTE**: Below processes are already done no need to do it. It's just for reference.
### Converting PNG Files to JPG Files

**ConvertingPNGFilesToJPGFiles.py** is designed to convert PNG images to JPG format. It iterates through directories, identifies PNG files, converts them to JPG, and replaces the original PNG files.

To execute the PNG to JPG conversion:

```bash
python ConvertingPNGFilesToJPGFiles.py
```

### Data Shrinking

**DataShrinking.py** is used for selecting and copying a specific number of random images from a source directory to a target directory. This is useful for creating smaller subsets of a dataset for testing and experimentation.

To execute the data shrinking process:

```bash
python DataShrinking.py
```

Please note that the effectiveness of these scripts depends on the specific dataset and requirements of your project. Make sure to adjust paths and parameters as needed for your dataset.

If you encounter any issues or have questions, feel free to reach out for support.
