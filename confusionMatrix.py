import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
        super().__init__()
        self.layer1 = nn.Conv2d(3, 32, 3, padding=1, stride=1)
        self.layer2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.Maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(32 * 50 * 50, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.Maxpool(F.relu(self.layer2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return F.log_softmax(self.fc(x), dim=1)

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

    # for epoch in range(epochs):
    #     model.train()
    #     running_loss = 0
    #     for instances, labels in train_loader:
    #         instances, labels = instances.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         output = model(instances)
    #         loss = criterion(output, labels)
    #
    #         # L1 Regularization
    #         l1_lambda = 0.001
    #         l1_norm = sum(p.abs().sum() for p in model.parameters())
    #         loss = loss + l1_lambda * l1_norm
    #
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #
    #     training_losses.append(running_loss / len(train_loader))
    #
    #     model.eval()
    #     validation_loss = 0
    #     with torch.no_grad():
    #         for instances, labels in validation_loader:
    #             instances, labels = instances.to(device), labels.to(device)
    #             output = model(instances)
    #             loss = criterion(output, labels)
    #             validation_loss += loss.item()
    #
    #     average_validation_loss = validation_loss / len(validation_loader)
    #     validation_losses.append(average_validation_loss)
    #     print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {training_losses[-1]}, Validation Loss: {validation_losses[-1]}')
    #
    #     scheduler.step(average_validation_loss)
    #
    #     if epoch >= min_epochs - 1:
    #         if average_validation_loss < best_validation_loss:
    #             best_validation_loss = average_validation_loss
    #             epochs_without_improvement = 0
    #         else:
    #             epochs_without_improvement += 1
    #             if epochs_without_improvement == early_stopping_patience:
    #                 print("Early stopping triggered")
    #                 break
    #
    # # Plotting training and validation losses
    # plt.plot(training_losses, label='Training Loss')
    # plt.plot(validation_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for instances, labels in test_loader:
            instances, labels = instances.to(device), labels.to(device)
            output = model(instances)
            _, predictions = torch.max(output, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate and plot the confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Labels for each class
    classes = ['Angry', 'Bored', 'Focused', 'Neutral']  # Update these based on your actual class names

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Save the trained model
    # torch.save(model.state_dict(), 'facial_recognition_model.pth')
