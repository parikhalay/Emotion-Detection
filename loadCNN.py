import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import random
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image


# Your model definition must be exactly the same as in the training script
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

def load_model(model_path, input_size, device):
    model = FacialExpressionCNN(input_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_dataset(model, data_loader, device):
    predictions = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions


def predict_single_image(model, image_path, transform, device):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'facial_recognition_model.pth'
    data_path = 'dataset/datacleaning/train'  # Replace with your dataset path
    random_image_path = 'dataset/datacleaning/train/angry/image0020323.jpg'  # Replace with your image path

    # Class names mapping
    classes = {0: 'angry', 1: 'focused', 2: 'neutral', 3: 'tired'}

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])

    # Load the model
    model = load_model(model_path, (3, 90, 90), device)

    # Predict on dataset
    dataset = ImageFolder(data_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    dataset_predictions = predict_dataset(model, data_loader, device)

    # Convert numerical predictions to class names
    dataset_predictions = [classes[pred] for pred in dataset_predictions]
    print(f"Dataset Predictions: {dataset_predictions}")

    # Predict on a single image
    image_prediction = predict_single_image(model, random_image_path, transform, device)
    image_prediction = classes[image_prediction]
    print(f"Prediction for image {random_image_path}: {image_prediction}")


if __name__ == '__main__':
    main()
