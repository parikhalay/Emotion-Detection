import os
import cv2
import numpy as np


# Define the input and output directories
input_base_directory = 'dataset/test/'
output_base_directory = 'datacleaning/test/'

# Define the target dimensions for resizing
target_width = 100
target_height = 100

# Define the range of brightness adjustments
brightness_factor = 1.1

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
            # Get the original dimensions of the image
            original_height, original_width, _ = image.shape

            # Calculate the new dimensions while preserving the aspect ratio
            if original_width > original_height:
                new_width = target_width
                new_height = int(original_height * (target_width / original_width))
            else:
                new_height = target_height
                new_width = int(original_width * (target_height / original_height))

            # Resizing the image to the new dimensions using high-quality interpolation
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # Create a canvas of the target dimensions
            canvas = 255 * np.ones((target_height, target_width, 3), dtype=np.uint8)

            # Calculate the position to paste the resized image in the canvas
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2

            # Paste the resized image onto the canvas
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

            # Apply brightness adjustment
            canvas = cv2.convertScaleAbs(canvas, alpha=brightness_factor, beta=0)

            # Save the processed image to the output directory
            cv2.imwrite(output_path, canvas)