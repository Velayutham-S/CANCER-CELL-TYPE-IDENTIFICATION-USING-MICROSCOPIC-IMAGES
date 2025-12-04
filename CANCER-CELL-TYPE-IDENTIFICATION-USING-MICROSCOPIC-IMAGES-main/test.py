import torch
import os
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 20  
model = models.resnet18(weights=None) 
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
model_path = "D:/Project/model.pth" 
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_path ="D:\Project\datasets\MMZ\MMZ_00629.jpg"
image = Image.open(image_path).convert('RGB')  
image = transform(image)
image = image.unsqueeze(0) 
image = image.to(device)
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

class_labels = [
    'ABE', 'ART', 'BAS', 'BLA', 'EBO', 'EOS', 'FGC', 'HAC', 'KSC', 'LYI',
    'LYT', 'MMZ', 'MON', 'MYB', 'NGB', 'NGS', 'NIF', 'OTH', 'PEB', 'PLM'
]  

predicted_label = class_labels[predicted.item()]
print(f"✅ Predicted Class: {predicted_label}")
test_dataset_path = "D:\\Project\\test"  
if os.path.exists(test_dataset_path):
    test_dataset = datasets.ImageFolder(root=test_dataset_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ Test Accuracy: {accuracy:.2f}%")
else:
    print("⚠️ No test dataset folder found. Skipping accuracy check.")
