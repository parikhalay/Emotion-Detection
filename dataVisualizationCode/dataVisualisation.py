import os
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Defining the parent directory containing subdirectories with class images
parent_directory = '../datacleaning/train'

# Getting the list of subdirectories (classes)
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

# Generating 5*5 grid with random sample images and generating intensity distribution of the same images in RGB channels
# Initialize Matplotlib for the image grid
fig1, axes1 = plt.subplots(5, 5, figsize=(8.5, 11), constrained_layout=True) # Standard letter-sized page

# Initialize Matplotlib for the histogram grid
fig2, axes2 = plt.subplots(5, 5, figsize=(20, 25), constrained_layout=True)  # Standard letter-sized page

# Randomly select and display 25 images from different classes simultaneously plot intensity graph
for i in range(5):
    for j in range(5):
        # Randomly selecting a class for Sample Images
        random_subdirectory = random.choice(subdirectories)

        # Getting the list of image files in the selected class
        image_files = os.listdir(os.path.join(parent_directory, random_subdirectory))

        # Randomly selecting an image from the class
        random_image = random.choice(image_files)

        # Loading and displaying the selected image in pixels
        image_path = os.path.join(parent_directory, random_subdirectory, random_image)
        image = mpimg.imread(image_path)

        if image is not None:
            # [Separate the image into 3 channels (B, G, and R)]
            bgr_planes = cv2.split(image)

            # Compute the histograms for each channel
            hist_b = cv2.calcHist(bgr_planes, [0], None, [256], (0, 256), accumulate=False)
            hist_g = cv2.calcHist(bgr_planes, [1], None, [256], (0, 256), accumulate=False)
            hist_r = cv2.calcHist(bgr_planes, [2], None, [256], (0, 256), accumulate=False)

            # Normalize the histograms
            cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_g, hist_g, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_r, hist_r, 0, 1, cv2.NORM_MINMAX)

            # Plot histogram
            axes2[i, j].plot(hist_b, color='blue')
            axes2[i, j].plot(hist_g, color='green')
            axes2[i, j].plot(hist_r, color='red')

            # Set labels and grid
            axes2[i, j].set_title(f'{random_subdirectory}')
            axes2[i, j].set_xlabel('Pixel Intensity')
            axes2[i, j].set_ylabel('Frequency')
            axes2[i, j].grid(True, which='both', linestyle='--', linewidth=0.5)

        # plot the selected random image
        axes1[i, j].imshow(image)
        axes1[i, j].set_title(f'{random_subdirectory}')
        axes1[i, j].axis('off')

# Show the histogram
plt.show()
