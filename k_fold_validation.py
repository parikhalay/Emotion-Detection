import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import time


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


# CNN Model Class
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


def k_fold_cross_validation(model_class, dataset_path, k=10, batch_size=64):
    # Data augmentation for the training set
    train_transform = transforms.Compose([
        transforms.Resize((90, 90)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])

    # Transformation for the validation set (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])

    # Load the full dataset
    full_dataset = ImageFolder(dataset_path, transform=train_transform)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_micro': [],
        'recall_micro': [],
        'f1_micro': []
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
        start_time = time.time()
        print(f"Fold {fold + 1}")

        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)
        train_subset.dataset.transform = train_transform
        test_subset.dataset.transform = val_transform

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model_class((3, 90, 90)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        early_stopping = EarlyStopping(patience=5, verbose=True)
        best_val_loss = float('inf')
        best_model_path = f'savemodel/best_model_fold_{fold + 1}.pth'

        for epoch in range(15):
            train_loss = train(model, device, train_loader, optimizer)
            val_loss = validate(model, device, val_loader)
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        final_model_path = f'savemodel/final_model_fold_{fold + 1}.pth'
        torch.save(model.state_dict(), final_model_path)

        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                y_pred.extend(pred.view_as(target).cpu().numpy())
                y_true.extend(target.cpu().numpy())

        # Time taken for this fold
        fold_time = time.time() - start_time

        # Calculate metrics for this fold with formatting to 4 decimal places
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        precision_micro = precision_score(y_true, y_pred, average='micro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        f1_micro = f1_score(y_true, y_pred, average='micro')

        metrics['accuracy'].append(accuracy)
        metrics['precision_macro'].append(precision_macro)
        metrics['recall_macro'].append(recall_macro)
        metrics['f1_macro'].append(f1_macro)
        metrics['precision_micro'].append(precision_micro)
        metrics['recall_micro'].append(recall_micro)
        metrics['f1_micro'].append(f1_micro)

        print(f"Fold {fold + 1} completed in {fold_time:.2f} seconds. Accuracy: {accuracy:.4f}")

    # Prepare data for CSV with formatting to 4 decimal places
    data_for_csv = []
    for i in range(k):
        data_for_csv.append([
            f"Fold {i+1}",
            f"{metrics['accuracy'][i]:.4f}",
            f"{metrics['precision_macro'][i]:.4f}",
            f"{metrics['recall_macro'][i]:.4f}",
            f"{metrics['f1_macro'][i]:.4f}",
            f"{metrics['precision_micro'][i]:.4f}",
            f"{metrics['recall_micro'][i]:.4f}",
            f"{metrics['f1_micro'][i]:.4f}"
        ])

    # Append average data with formatting
    data_for_csv.append([
        "Average",
        f"{np.mean(metrics['accuracy']):.4f}",
        f"{np.mean(metrics['precision_macro']):.4f}",
        f"{np.mean(metrics['recall_macro']):.4f}",
        f"{np.mean(metrics['f1_macro']):.4f}",
        f"{np.mean(metrics['precision_micro']):.4f}",
        f"{np.mean(metrics['recall_micro']):.4f}",
        f"{np.mean(metrics['f1_micro']):.4f}"
    ])

    return data_for_csv


if __name__ == '__main__':
    cross_val_results = k_fold_cross_validation(FacialExpressionCNN, 'dataset/datacleaning/train')

    # Save to CSV
    with open('cross_validation_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fold', 'Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Micro Precision', 'Micro Recall', 'Micro F1'])
        writer.writerows(cross_val_results)

    # Print the CSV content
    with open('cross_validation_results.csv', 'r') as file:
        print(file.read())