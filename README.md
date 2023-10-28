
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
