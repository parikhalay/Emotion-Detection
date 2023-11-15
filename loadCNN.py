import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader


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


def load_model(model_path, device):
    model = FacialRecognitionCNN(output_size=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def run_on_dataset(model, dataset_path, device, class_names):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    ])
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=120, shuffle=False)

    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            predicted_classes = [class_names[prediction] for prediction in predictions]
            print("Predicted classes:", predicted_classes)


def run_on_single_image(model, image_path, device, class_names):
    image = transform_image(image_path).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        probability = F.softmax(output, dim=1)
        prediction = torch.argmax(probability, dim=1)
        predicted_class = class_names[prediction.item()]
        print("Predicted class:", predicted_class)


if __name__ == '__main__':
    class_names = ['angry', 'focused', 'neutral', 'tired']
    model_path = 'facial_recognition_model.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device)

    # Test with dataset
    dataset_path = 'dataset/datacleaning/train'
    run_on_dataset(model, dataset_path, device, class_names)

    # Test with a single image
    image_path = 'dataset/datacleaning/test/tired/image0002843.jpg'
    run_on_single_image(model, image_path, device, class_names)
