import torch
import torch.nn as nn

# Define your model architecture (replace with your actual model class)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Example layers
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

# Instantiate the model
model = MyModel()

# Load the state dict (weights)
state_dict = torch.load("D:/Project/model.pth")
model.load_state_dict(state_dict)

# Set model to evaluation mode
model.eval()
