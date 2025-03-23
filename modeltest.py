import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

# Define the same CNN model used for training
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input: [batch_size, 3, 50, 50], Output: [batch_size, 32, 50, 50]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: [batch_size, 64, 25, 25]
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 2)  # Output layer (for 2 classes: IDC and non-IDC)

    def forward(self, x):
        # Apply convolution and pooling layers
        x = self.pool(torch.relu(self.conv1(x)))  # Output: [batch_size, 32, 25, 25]
        x = self.pool(torch.relu(self.conv2(x)))  # Output: [batch_size, 64, 12, 12]
        x = x.reshape(x.size(0), -1)  # Flatten the tensor before feeding into fully connected layer
        x = torch.relu(self.fc1(x))  # Output: [batch_size, 128]
        x = self.fc2(x)  # Output: [batch_size, num_classes]
        return x

# Load the trained model with weights_only=True for added security
model = SimpleCNN()
model.load_state_dict(torch.load('simple_cnn.pth', weights_only=True))  # Load the trained model (secure mode)
model.eval()  # Set the model to evaluation mode

# Inference function for a single image
def infer_single_image(image_path, model, device):
    # Define transformations (resize, convert to tensor, normalize)
    transform = transforms.Compose([
        transforms.Resize((50, 50)),  # Resize image to 50x50
        transforms.ToTensor(),        # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to match model training
    ])
    
    # Open the image
    image = Image.open('test2.png').convert("RGB")  # Ensure image is RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension (1, 3, 50, 50)
    
    image = image.to(device)  # Move image to device (CPU or GPU)
    
    # Forward pass
    with torch.no_grad():  # No gradient calculation during inference
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class (IDC or non-IDC)
    
    # Return the prediction
    return 'IDC' if predicted.item() == 1 else 'Non-IDC'

# Choose the device (CPU or CUDA)
device = torch.device('cpu')  # Or 'cuda' if you have a GPU

# Path to the PNG image you want to test
image_path = 'test1.png'  # Provide the path to the image you want to test

# Perform inference
prediction = infer_single_image(image_path, model, device)
print(f"Prediction for the image:{image_path} {prediction}")
