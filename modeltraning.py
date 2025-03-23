import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Define a simple CNN model
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

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# Load your dataset
x_data = np.load('/X.npy')  # Corrected file path
y_data = np.load('/Y.npy')  # Corrected file path

# Ensure that the input data has the correct shape
x_data = np.transpose(x_data, (0, 3, 1, 2))  # Convert to shape [batch_size, channels, height, width] (e.g., [5547, 3, 50, 50])

# Split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Create Dataset and DataLoader for training and testing
train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, criterion, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Testing function
def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Main loop for training and testing
device = torch.device('cpu')  # You can change this to 'cuda' if you have a GPU available
print(f"Using device: {device}")

for epoch in range(20):
    print(f"Epoch {epoch+1}/20")
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test_model(model, test_loader, criterion, device)
    
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

# Save the trained model after the training loop
torch.save(model.state_dict(), 'simple_cnn.pth')  # Save the model's state_dict
print("Model saved to 'simple_cnn.pth'")

# Load the saved model (when needed for inference)
model = SimpleCNN()
model.load_state_dict(torch.load('simple_cnn.pth'))
model.eval()  # Set the model to evaluation mode
