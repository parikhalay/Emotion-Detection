from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt


def calculate_histograms_and_display(image_paths, rows, cols):
    # [Establish the number of bins]
    histSize = 256
    # [Establish the number of bins]

    # [Set the ranges (for B, G, R)]
    histRange = (0, 256)  # the upper boundary is exclusive
    # [Set the ranges (for B, G, R)]

    # [Set histogram param]
    accumulate = False
    # [Set histogram param]

    # Initialize Matplotlib for the histogram grid
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, image_path in enumerate(image_paths):
        if i >= rows * cols:
            break

        # Load the image
        src = cv.imread(image_path)
        if src is None:
            print('Could not open or find the image:', image_path)
            continue

        # [Separate the image into 3 channels (B, G, and R)]
        bgr_planes = cv.split(src)
        # [Separate the image into 3 channels (B, G, and R)]

        # [Compute the histograms]
        b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
        g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
        r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
        # [Compute the histograms]

        # [Normalize the result to (0, histImage.rows)]
        cv.normalize(b_hist, b_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        cv.normalize(g_hist, g_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        cv.normalize(r_hist, r_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        # [Normalize the result to (0, histImage.rows)]

        # Plot histograms for each channel
        axes[i // cols, i % cols].plot(b_hist, color='blue')
        axes[i // cols, i % cols].plot(g_hist, color='green')
        axes[i // cols, i % cols].plot(r_hist, color='red')

        # Set labels and title
        axes[i // cols, i % cols].set_title('Image {}'.format(i + 1))
        axes[i // cols, i % cols].set_xlabel('Pixel Intensity')
        axes[i // cols, i % cols].set_ylabel('Frequency')
        axes[i // cols, i % cols].legend(['Blue', 'Green', 'Red'])

    # Display the histograms in a grid
    plt.show()


if __name__ == "__main__":
    # Define a list of image directories
    image_directories = ['../dataset/train/neutral/ffhq_310.png', '../dataset/train/neutral/ffhq_5168.png', '../dataset/train/neutral/ffhq_5144.png']  # Add your directory paths

    # Set the number of rows and columns for the grid
    rows = 3
    cols = 3  # You can adjust this according to the number of directories

    # Initialize a list to store image paths
    image_paths = []

    # Collect image paths from each directory
    for directory in image_directories:
        image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_paths.extend(image_files)

    if not image_paths:
        print('No image files found in the specified directories.')
    else:
        calculate_histograms_and_display(image_paths, rows, cols)
