import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import math

# Load the CSV file
csv_file = 'Part2_test_labels.csv'  # Replace with your actual CSV file path
data = pd.read_csv(csv_file)

# Function to display a grid of images
def display_image_grid(images, titles, rows=5, cols=5):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 15))
    axes = axes.ravel()

    for i in range(rows * cols):
        if i < len(images):
            img = Image.open(images[i])
            axes[i].imshow(img)
            axes[i].set_title(titles[i])
            axes[i].axis('off')
        else:
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == '__main__':
    num_images_per_grid = 25
    num_total_images = len(data)
    num_grids = math.ceil(num_total_images / num_images_per_grid)

    for grid in range(num_grids):
        start = grid * num_images_per_grid
        end = start + num_images_per_grid
        subset = data[start:end]

        images = subset['Image Path'].tolist()
        titles = subset.apply(lambda x: f"Gender: {x['Gender']}, Age: {x['Age Category']}", axis=1).tolist()

        display_image_grid(images, titles)
