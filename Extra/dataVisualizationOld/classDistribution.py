import os
import cv2
import matplotlib.pyplot as plt

# Define the parent directory containing subdirectories with images
parent_directory = '../datacleaning/final'

# Get a list of subdirectories
input_directory = [os.path.join(parent_directory, d) for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]

print(input_directory)
angry = []
neutral = []

# Loop through the files in the input directory and process each image
for image_directory in input_directory:
    for filename in os.listdir(image_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust the file extensions as needed
            input_path = os.path.join(image_directory, filename)

            # Load the image
            image = cv2.imread(input_path)
            if image_directory == '../datacleaning/final/angry':
                angry.append(image)
            if image_directory == '../datacleaning/final/neutral':
                neutral.append(image)

data = [len(angry), len(neutral)]

labels = ['Angry', 'Neutral']
# shuffled_data, shuffled_labels = shuffle(data, labels, random_state=42)  # Replace 'data' and 'labels' with your dataset and labels
# unique_labels, label_counts = np.unique(shuffled_labels, return_counts=True)
plt.bar(labels, data)
plt.xticks(labels, ['Angry', 'Neutral'])
plt.xlabel('Emotion Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution')
plt.show()