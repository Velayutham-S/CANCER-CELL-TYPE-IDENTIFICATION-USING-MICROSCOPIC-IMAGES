import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset_path = r"D:\Project\datasets"  
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

weights = ResNet18_Weights.IMAGENET1K_V1   
model = resnet18(weights=weights)
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
model.train()

for epoch in range(epochs):
    start_time = time.time()

    running_loss = 0.0
    loop = tqdm(dataloader, total=len(dataloader), desc=f"Epoch [{epoch+1}/{epochs}]")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        
        loop.set_postfix(loss=loss.item())

    epoch_time = time.time() - start_time
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}, Time: {epoch_time:.2f} seconds")


save_path = r"D:\Project\model.pth"   
torch.save(model.state_dict(), save_path)
print(f"✅ Model saved to {save_path}")
