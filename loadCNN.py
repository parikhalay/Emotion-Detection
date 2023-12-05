import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

def load_model(model_path, input_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FacialExpressionCNN(input_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # Add batch dimension

def get_true_class(file_path):
    return os.path.basename(os.path.dirname(file_path))

def predict_image_class(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()


class App:
    def __init__(self, root, model, device):
        self.root = root
        self.model = model
        self.device = device

        self.class_mapping = {0: "angry", 1: "focused", 2: "neutral", 3: "tired"}

        self.root.title("Facial Expression Classifier")
        self.root.geometry("800x600")

        self.label = tk.Label(root, text="Enter the path of the image:")
        self.label.pack()

        self.entry = tk.Entry(root, width=50)
        self.entry.bind("<Return>", self.on_submit)  # Bind enter key to submit
        self.entry.pack()

        self.submit_button = tk.Button(root, text="Submit", command=self.on_submit)
        self.submit_button.pack()

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

    def on_submit(self, event=None):
        image_path = self.entry.get()
        if image_path:
            true_class = get_true_class(image_path)
            image_tensor = preprocess_image(image_path)
            predicted_class_num = predict_image_class(self.model, image_tensor, self.device)
            predicted_class = self.class_mapping[predicted_class_num]

            self.display_image(image_path)
            self.result_label.config(text=f"True Class: {true_class}\nPredicted Class: {predicted_class}")

    def display_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((250, 250), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference to avoid garbage collection

def main():
    model_path = 'savemodel/facial_recognition_best_model.pth'
    input_size = (3, 90, 90)
    model = load_model(model_path, input_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = tk.Tk()
    app = App(root, model, device)
    root.mainloop()

if __name__ == "__main__":
    main()
