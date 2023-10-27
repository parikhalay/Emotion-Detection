import shutil
from PIL import Image
import os

# Replace 'root_directory' with the path to the root directory where you want to search for PNG files.
root_directory = '../dataset'

# Iterate through all directories and subdirectories
for dirpath, dirnames, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename.lower().endswith('.png'):
            png_path = os.path.join(dirpath, filename)

            # Convert to JPG
            try:
                with Image.open(png_path) as img:
                    jpg_path = os.path.splitext(png_path)[0] + '.jpg'
                    img.convert('RGB').save(jpg_path, 'JPEG')
                    # Remove the original PNG file
                    os.remove(png_path)
                print(f"Converted and replaced: {png_path}")
            except Exception as e:
                print(f"Failed to convert and replace {png_path}: {str(e)}")
