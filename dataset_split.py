import os
import shutil
from sklearn.model_selection import train_test_split


def split_dataset(base_dir, output_dir, train_size=0.85):
    """
    Splits the dataset into training and testing sets, with a specified percentage for training.
    Remaining images are used for testing.

    :param base_dir: Directory containing the dataset with subdirectories for each class.
    :param output_dir: Output directory where the split dataset will be saved.
    :param train_size: Proportion of the dataset to include in the train split.
    """

    # Create output directories for train and test
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate over each category (subdirectory)
    for category in os.listdir(base_dir):
        # Create new subdirectories in output directories
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        # List all images in the category directory
        category_dir = os.path.join(base_dir, category)
        images = [f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))]

        # Adjust train_size to get a precise number of images for training
        total_images = len(images)
        train_count = int(total_images * train_size)
        test_count = total_images - train_count

        # Split the dataset
        train_images, test_images = train_test_split(images, train_size=train_count, test_size=test_count,
                                                     random_state=42)

        # Copy images to their respective directories
        for image in train_images:
            shutil.copy(os.path.join(category_dir, image), os.path.join(train_dir, category))
        for image in test_images:
            shutil.copy(os.path.join(category_dir, image), os.path.join(test_dir, category))

# Example usage
split_dataset('dataset/extra_images/data', 'dataset/extra_images/split')

