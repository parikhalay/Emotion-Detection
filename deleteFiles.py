import os

# Define the base directory containing subdirectories
base_directory = 'dataset/datacleaning/train'


# Function to delete files from subdirectories
def delete_files_in_subdirectories(directory):
    for subdir, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                # Delete the file
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")


# Call the function to delete files from subdirectories
delete_files_in_subdirectories(base_directory)
