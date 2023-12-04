from PIL import Image
import numpy as np
import os
import cv2

def rotate_image(image, angle):
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

def adjust_brightness_contrast(image, alpha, beta):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

# Define a directory containing your input image files
input_directory = 'dataset/train/neutral/'

# Define a directory to save the resized images
output_directory = 'datacleaning/final/neutral/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Specify the target size for resizing
target_size = (45, 45)  # Change to your desired size

# Loop through the files in the input directory and process each image
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust the file extensions as needed
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        # Load the image
        image = cv2.imread(input_path)

        # Check if the image was successfully loaded
        if image is not None:
            # Resize the image
            resized_image = cv2.resize(image, target_size)
            # cv2.imwrite(output_path, resized_image)     # Save the resized image to the output directory

            # Rotate the image
            rotation_angle = np.random.randint(-10, 10)  # Random rotation between -10 and 10 degrees
            rotated_image = rotate_image(resized_image, rotation_angle)
            # cv2.imwrite(output_path, rotated_image)

            # Brighten the image
            alpha = 1  # Brightness adjustment factor
            beta = 10  # Contrast adjustment factor
            adjusted_image = adjust_brightness_contrast(rotated_image, alpha, beta)
            cv2.imwrite(output_path, adjusted_image)
