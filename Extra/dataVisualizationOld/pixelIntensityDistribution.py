import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Define the parent directory containing subdirectories with class images
parent_directory = '../datacleaning/final'

# Get a list of subdirectories (classes)
subdirectories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]

# Initialize Matplotlib for the grid
fig, axes = plt.subplots(5, 5, figsize=(8.5, 11), constrained_layout=True)

# Initialize variables for pixel intensity distribution
pixel_intensity_data = []

# Randomly select and display 25 images from different classes
for i in range(5):
    for j in range(5):
        # Randomly select a class (subdirectory)
        random_subdirectory = random.choice(subdirectories)

        # Get a list of image files in the selected class
        image_files = os.listdir(os.path.join(parent_directory, random_subdirectory))

        # Randomly select an image from the class
        random_image = random.choice(image_files)

        # Load and display the selected image
        image_path = os.path.join(parent_directory, random_subdirectory, random_image)
        image = mpimg.imread(image_path)
        axes[i, j].imshow(image)
        axes[i, j].set_title(f'Image {i * 5 + j + 1}')
        axes[i, j].axis('off')

        # Calculate the pixel intensity distribution for the image and store it
        pixel_intensity = image.ravel()
        pixel_intensity_data.append(pixel_intensity)

# Plot the 5x5 grid of random images
plt.show()

# Plot the pixel intensity distribution for the same set of images
plt.figure(figsize=(10, 5))
plt.title('Pixel Intensity Distribution')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim(0, 256)

# Combine pixel intensity data from all images
pixel_intensity_data = np.concatenate(pixel_intensity_data)

# Calculate and plot the histogram
hist, bins = np.histogram(pixel_intensity_data, bins=256, range=(0, 256))
plt.plot(bins[0:-1], hist, color='gray', alpha=0.5, label='Pixel Intensity')

plt.legend()
plt.show()
