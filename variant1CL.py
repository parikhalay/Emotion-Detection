import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# Function to load the dataset
def load_dataset(data_path, batch_size=64):
    # Data augmentation for the training set
    train_transform = transforms.Compose([
        transforms.Resize((90, 90)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])

    # Transformation for validation and test sets (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])

    # Load the full dataset
    full_dataset = ImageFolder(data_path, transform=train_transform)  # Initially set to train_transform
    train_size = int(0.7 * len(full_dataset))
    val_test_size = int(len(full_dataset)) - train_size
    val_size = test_size = val_test_size // 2

    # Splitting the dataset
    train_dataset, remaining_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_test_size])
    validation_dataset, test_dataset = torch.utils.data.random_split(remaining_dataset, [val_size, test_size])

    # Apply test_transform to validation and test datasets
    validation_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    # Data loaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# CNN Model Class
class FacialExpressionCNN(nn.Module):
    def __init__(self, input_size):
        super(FacialExpressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Additional layer

        # Adjusted flattened size calculation
        self.flattened_size = self._get_conv_output_size(input_size, more=True)

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 4)  # 4 classes
        self.dropout = nn.Dropout(0.5)

    def _get_conv_output_size(self, input_size, more=False):
        dummy_input = torch.zeros(1, *input_size)
        output = self.conv1(dummy_input)
        output = F.max_pool2d(output, 2)
        output = self.conv2(output)
        output = F.max_pool2d(output, 2)
        output = self.conv3(output)
        output = F.max_pool2d(output, 2)
        if more:
            output = self.conv4(output)
            output = F.max_pool2d(output, 2)
        return int(np.prod(output.size()))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))  # Forward through additional layer
        x = x.view(x.size(0), -1)  # Flatten layer
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()
    return val_loss / len(val_loader.dataset)


def test(model, device, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            correct += preds.eq(target.view_as(preds)).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    conf_matrix = confusion_matrix(all_targets, all_preds)
    accuracy = accuracy_score(all_targets, all_preds)
    precision_macro = precision_score(all_targets, all_preds, average='macro')
    recall_macro = recall_score(all_targets, all_preds, average='macro')
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    precision_micro = precision_score(all_targets, all_preds, average='micro')
    recall_micro = recall_score(all_targets, all_preds, average='micro')
    f1_micro = f1_score(all_targets, all_preds, average='micro')

    return conf_matrix, accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro, test_loss, test_accuracy

# Main Function
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FacialExpressionCNN((3, 90, 90)).to(device)

    # Adjust optimizer to include weight decay for L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Optional: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_loader, val_loader, test_loader = load_dataset(data_path='dataset/datacleaning/train')

    epochs = 15
    patience = 10
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer)
        val_loss = validate(model, device, val_loader)

        # Update learning rate
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save the model
    torch.save(model.state_dict(), 'facial_recognition_variant1.pth')

    plt.plot(range(len(train_losses)), train_losses, label='Training loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    conf_matrix, accuracy, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro, test_loss, test_accuracy = test(
        model, device, test_loader)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    print("\nMetrics Summary:")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1 Score: {f1_macro:.4f}")
    print(f"Micro Precision: {precision_micro:.4f}")
    print(f"Micro Recall: {recall_micro:.4f}")
    print(f"Micro F1 Score: {f1_micro:.4f}")
