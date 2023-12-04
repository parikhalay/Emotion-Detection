import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# CNN Model Class Definition
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

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.annotations.iloc[index, 0]
        image = Image.open(img_path)
        gender = self.annotations.iloc[index, 1]  # m or f
        age = int(self.annotations.iloc[index, 2])  # 1, 2, or 3

        if self.transform:
            image = self.transform(image)

        return image, gender, age


# Function to Load Custom Dataset
def load_custom_dataset(csv_file):
    transform = transforms.Compose([
        transforms.Resize((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])
    return CustomDataset(csv_file=csv_file, transform=transform)


# Function to Evaluate Model for a Specific Group
def evaluate_model_group(model, device, dataset, group_condition):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for image, gender, age in dataset:
            if group_condition(gender, age):
                image = image.unsqueeze(0).to(device)
                output = model(image)
                _, pred = torch.max(output, 1)
                all_labels.append(age)
                all_preds.append(pred.cpu().item())

    return all_labels, all_preds


# Function to Calculate Metrics
def calculate_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return accuracy, precision, recall, f1


# Main Execution
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'savemodel/best_model.pth'
    csv_file = 'Part2_test_labels.csv'

    # Load Model
    model = FacialExpressionCNN((3, 90, 90)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load Dataset
    dataset = load_custom_dataset(csv_file)

    # Define Group Conditions
    age_groups = {'Young': lambda g, a: a == 1, 'Middle-aged': lambda g, a: a == 2, 'Senior': lambda g, a: a == 3}
    gender_groups = {'Male': lambda g, a: g == 'm', 'Female': lambda g, a: g == 'f'}

    results = []

    # Evaluate and Append Results for Each Group
    for attr, groups in [('Age', age_groups), ('Gender', gender_groups)]:
        group_results = []
        for group_name, group_condition in groups.items():
            labels, preds = evaluate_model_group(model, device, dataset, group_condition)
            accuracy, precision, recall, f1 = calculate_metrics(labels, preds)
            results.append([attr, group_name, accuracy, precision, recall, f1])
            group_results.append([accuracy, precision, recall, f1])

        # Calculate and append average for each attribute
        avg_results = np.mean(group_results, axis=0)
        results.append([attr, 'Average', *avg_results])

    # Calculate and append overall average
    overall_avg = np.mean([r[2:] for r in results], axis=0)
    results.append(['Overall', 'Average', *overall_avg])

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=['Attribute', 'Group', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    results_df.to_csv('bias_analysis_results.csv', index=False)

    # Print results
    print(results_df)
