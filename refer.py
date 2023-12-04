import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from matplotlib import image as mp_image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class FacialExpressionCNN(nn.Module):
    def __init__(self, input_size):
        super(FacialExpressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Calculate the size of the flattened features after the conv layers
        self.flattened_size = self._get_conv_output_size(input_size)

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 4)  # 4 classes
        # Add dropout
        self.dropout = nn.Dropout(0.5)

    def _get_conv_output_size(self, input_size):
        # Pass a dummy input through the convolution layers to calculate the size
        dummy_input = torch.zeros(1, *input_size)
        output = self.conv1(dummy_input)
        output = F.max_pool2d(output, 2)
        output = self.conv2(output)
        output = F.max_pool2d(output, 2)
        output = self.conv3(output)
        output = F.max_pool2d(output, 2)
        return int(np.prod(output.size()))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)  # Flatten layer
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def generateBiasAnalysisTable(model_path):
    Category = [1, 2, 3]
    Gender = ["m", "f"]
    for i in Category:
        evaluateModelOnSpecificCategory('Age Category', i, model_path)

    for i in Gender:
        evaluateModelOnSpecificCategory('Gender', i, model_path)

def get_true_label(file_path):
    # Assuming your file path structure is consistent
    # You can split the path and get the label from the appropriate position
    label = str(file_path.split("/")[-2]).lower()  # Assumes the label is the second-to-last element in the path
    if(label=="angry"):
        return 0
    elif(label=="focused"):
        return 1
    elif(label=="neutral"):
        return 2
    elif(label=="tired"):
        return 3

def evaluateModelOnSpecificCategory(Column, value, model_path):
    # Load CSV file containing file_path, gender, and age
    csv_file_path = 'Part2_test_labels.csv'
    df = pd.read_csv(csv_file_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = df[df[Column] == value]['Image Path'].tolist()

    # Initialize your model and load the pre-trained weights
    model = FacialExpressionCNN((3, 90, 90)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Define loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Evaluate on the test set
    model.eval()
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for file_path in images:
            # file_path = file_path.replace("AppliedAI/", "")

            # Define transformations for the input image
            input_image = mp_image.imread(file_path)
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((90, 90)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
            ])
            input_tensor = transform(input_image).unsqueeze(0)
            # input_tensor = input_tensor.to(device)
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            test_predictions.append(predicted.item())
            # Get the true label from the file_path or dataframe

            true_label = get_true_label(file_path)  # Implement this function
            test_labels.append(true_label)

    # # Calculate evaluation metrics
    print("##########################")
    print("AFTER RESOLVING THE BIAS")
    # print("For Senior people")
    if Column == 'Age Category':
        if value == 1:
            print("For Young people")
        elif value == 2:
            print("For Middle Age people")
        elif value == 3:
            print("For Senior")
    else:
        print("For", Column, "-", value)

    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    f1 = f1_score(test_labels, test_predictions, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print()

if __name__ == '__main__':
    model_path = r'savemodel/facial_recognition_best_model.pth'
    print("The model path is: ", model_path)
    generateBiasAnalysisTable(model_path)