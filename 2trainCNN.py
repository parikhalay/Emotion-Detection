import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import numpy as np

# Define the CNN model
class FacialRecognitionCNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 32, 3, padding=1, stride=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.layer3 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.Maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)  # Increased dropout
        self.fc1 = nn.Linear(128 * 12 * 12, 256)  # Additional dense layer
        self.fc2 = nn.Linear(256, output_size)  # Final output layer

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.layer1(x)))
        x = self.Maxpool(x)
        x = F.relu(self.batchnorm2(self.layer2(x)))
        x = self.Maxpool(x)
        x = F.relu(self.batchnorm3(self.layer3(x)))
        x = self.Maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        x = F.relu(self.fc1(x))  # First dense layer
        x = self.fc2(x)  # Output layer without relu for logits
        return F.log_softmax(x, dim=1)


# Function to automatically split dataset into training, validation, and test sets
def facial_image_loader(batch_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(100, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root="dataset/datacleaning/train", transform=transform)
    train_size = int(0.7 * len(full_dataset))
    validation_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - validation_size

    train_dataset, validation_dataset, test_dataset = random_split(full_dataset, [train_size, validation_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader


if __name__ == '__main__':
    batch_size = 64
    output_size = 4  # Number of classes: angry, focused, neutral, tired

    train_loader, validation_loader, test_loader = facial_image_loader(batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FacialRecognitionCNN(output_size).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    epochs = 15  # Maximum number of epochs
    early_stopping_patience = 5
    epochs_without_improvement = 0
    best_validation_loss = float('inf')
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for instances, labels in train_loader:
            instances, labels = instances.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(instances)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        training_losses.append(running_loss / len(train_loader))

        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for instances, labels in validation_loader:
                instances, labels = instances.to(device), labels.to(device)
                output = model(instances)
                loss = criterion(output, labels)
                validation_loss += loss.item()

        validation_losses.append(validation_loss / len(validation_loader))
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {training_losses[-1]}, Validation Loss: {validation_losses[-1]}')

        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == early_stopping_patience:
                print("Early stopping triggered")
                break

    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the trained model
    torch.save(model.state_dict(), 'facial_recognition_model.pth')

    # Optionally evaluate the model on the test set
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for instances, labels in test_loader:
            instances, labels = instances.to(device), labels.to(device)
            output = model(instances)
            test_loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')

