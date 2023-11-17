import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Function to load facial image dataset with refined data augmentation
def facial_image_loader(batch_size, shuffle_test=False):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(100, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root="dataset/datacleaning/train", transform=transform)
    test_dataset = datasets.ImageFolder(root="dataset/datacleaning/test", transform=transform)

    train_size = int(0.8 * len(train_dataset))
    validation_size = len(train_dataset) - train_size
    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

    train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = td.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = td.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)

    return train_loader, validation_loader, test_loader

# Simplified CNN Model
class FacialRecognitionCNN(nn.Module):
    def __init__(self, output_size):
        super(FacialRecognitionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 6 * 6,
                             512)  # Adjust the input size based on the output of the last convolutional layer
        self.fc2 = nn.Linear(512, output_size)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    batch_size = 120
    output_size = 4  # Number of classes

    train_loader, validation_loader, test_loader = facial_image_loader(batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FacialRecognitionCNN(output_size).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    epochs = 11
    early_stopping_patience = 5
    min_epochs = 10
    epochs_without_improvement = 0
    best_validation_loss = float('inf')
    training_losses = []
    validation_losses = []


    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for instances, labels in validation_loader:
            instances, labels = instances.to(device), labels.to(device)
            outputs = model(instances)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')