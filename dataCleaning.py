import os
import cv2
import numpy as np
import random

# Define the input and output directories
input_base_directory = 'dataset/train/'
output_base_directory = 'datacleaning/final/'

# Define the target dimensions for resizing
target_width = 45  # You can adjust these dimensions as needed
target_height = 45

# Define the range of rotation angles in degrees
min_rotation = -10
max_rotation = 10

# Define the range of brightness adjustments
min_brightness = 0.7  # Darken
max_brightness = 1  # Brighten

# Define the range for minor cropping
min_crop_percentage = 0.9
max_crop_percentage = 1.0

# Loop through subdirectories in the base input directory
for subdirectory in os.listdir(input_base_directory):
    input_directory = os.path.join(input_base_directory, subdirectory)
    output_directory = os.path.join(output_base_directory, subdirectory)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Loop through the images in the input directory
    for filename in os.listdir(input_directory):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        # Load the image
        image = cv2.imread(input_path)

        if image is not None:
            # Resize the image to the target dimensions
            image = cv2.resize(image, (target_width, target_height))

            # Apply slight rotation
            rotation_angle = random.uniform(min_rotation, max_rotation)
            rotation_matrix = cv2.getRotationMatrix2D((target_width / 2, target_height / 2), rotation_angle, 1)
            image = cv2.warpAffine(image, rotation_matrix, (target_width, target_height))

            # Apply brightness adjustment
            brightness_factor = random.uniform(min_brightness, max_brightness)
            image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

            # Apply minor cropping
            crop_percentage = random.uniform(min_crop_percentage, max_crop_percentage)
            crop_size = (int(target_width * crop_percentage), int(target_height * crop_percentage))
            image = image[(target_height - crop_size[1]) // 2:(target_height + crop_size[1]) // 2,
                    (target_width - crop_size[0]) // 2:(target_width + crop_size[0]) // 2]

            # Save the processed image to the output directory
            cv2.imwrite(output_path, image)

# You can add any additional processing steps as needed
