import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import random
import matplotlib.image as mpimg

# Define the parent directory containing subdirectories with images
parent_directory = '../datacleaning/final/'

# Get a list of subdirectories
input_directory = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]

# Initialize Matplotlib for the grid
fig, axes = plt.subplots(5, 5, figsize=(8.5, 11), constrained_layout=True)

# Randomly select and display 25 images from different directories
for i in range(5):
    for j in range(5):
        # Randomly select a subdirectory
        random_subdirectory = random.choice(input_directory)

        # Get a list of image files in the selected subdirectory
        image_files = os.listdir(os.path.join(parent_directory, random_subdirectory))

        # Randomly select an image from the subdirectory
        random_image = random.choice(image_files)

        # Load and display the selected image
        image_path = os.path.join(parent_directory, random_subdirectory, random_image)
        image = mpimg.imread(image_path)
        axes[i, j].imshow(image)
        axes[i, j].set_title(f'Image {i * 5 + j + 1}')
        axes[i, j].axis('off')

# Show the grid
plt.show()
