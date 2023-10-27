import os
import random

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Define the parent directory containing subdirectories with class images
parent_directory = 'datacleaning/train'

# Get a list of subdirectories (classes)
subdirectories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]

# Class Distribution (Bar Graph)
class_counts = [len(os.listdir(os.path.join(parent_directory, sub))) for sub in subdirectories]
class_labels = [sub for sub in subdirectories]

plt.figure(figsize=(8, 8))
plt.bar(class_labels, class_counts)
plt.xticks(rotation=45, ha="right")
plt.title('Class Distribution')
plt.xlabel('Classes')
plt.ylabel('Number of Images')

# Initialize Matplotlib for the grid
fig, axes = plt.subplots(5, 5, figsize=(8.5, 11), constrained_layout=True)

# Create a list to store histograms for each channel
histograms = []

# Randomly select and display 25 images from different classes
for i in range(1):
    for j in range(1):
        # Randomly select a class for Sample Images
        random_subdirectory = random.choice(subdirectories)

        # Get a list of image files in the selected class
        image_files = os.listdir(os.path.join(parent_directory, random_subdirectory))

        # Randomly select an image from the class
        random_image = random.choice(image_files)

        # Load and display the selected image
        image_path = os.path.join(parent_directory, random_subdirectory, random_image)
        image = mpimg.imread(image_path)

        if image is not None:
            # Split the image into RGB channels
            b, g, r = cv2.split(image)

            # Compute histograms for each channel
            hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
            hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

            # Normalize the histograms
            hist_b = hist_b / hist_b.sum()
            hist_g = hist_g / hist_g.sum()
            hist_r = hist_r / hist_r.sum()

            # Store the histograms
            histograms.append((hist_b, hist_g, hist_r))

        axes[i, j].imshow(image)
        axes[i, j].set_title(f'{random_subdirectory}')
        axes[i, j].axis('off')

# Create a Matplotlib figure
plt.figure(figsize=(10, 8))

# Plot the histograms for each channel and image
for i, (hist_b, hist_g, hist_r) in enumerate(histograms):
    plt.plot(hist_b, color='blue', label=f'Image {i} (Blue Channel)')
    plt.plot(hist_g, color='green', label=f'Image {i} (Green Channel)')
    plt.plot(hist_r, color='red', label=f'Image {i} (Red Channel)')

# Set labels and legend
plt.title('Pixel Intensity Distribution')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
# plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the histogram
plt.show()
