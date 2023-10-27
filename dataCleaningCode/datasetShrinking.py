import os
import shutil
import random

# Define the source directory containing subdirectories with images
source_directory = '../dataset/data'

# Define the target directory where you want to save the selected images
target_directory = '../dataset/train'

# Number of images to select
num_images_to_select = 511

# Create the target directory if it doesn't exist
os.makedirs(target_directory, exist_ok=True)


# Function to select and copy random images from a source directory to a target directory
def select_and_copy_random_images(source_dir, target_dir, num_images):
    # Create a list of subdirectories in the source directory
    subdirectories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    # Randomly select images from each subdirectory
    for subdirectory in subdirectories:
        source_subdir = os.path.join(source_dir, subdirectory)
        target_subdir = os.path.join(target_dir, subdirectory)
        os.makedirs(target_subdir, exist_ok=True)
        image_files = os.listdir(source_subdir)

        # Ensure that the number of images to select is not greater than the number of available images
        num_images_to_select_from_subdir = min(num_images, len(image_files))

        # Randomly select images
        selected_images = random.sample(image_files, num_images_to_select_from_subdir)

        # Copy the selected images to the target directory
        for image in selected_images:
            source_path = os.path.join(source_subdir, image)
            target_path = os.path.join(target_subdir, image)
            shutil.copy(source_path, target_path)

    print(f'Selected and copied {num_images} random images to the target directory.')


# Call the function to select and copy random images
select_and_copy_random_images(source_directory, target_directory, num_images_to_select)
