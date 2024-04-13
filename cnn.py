import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# Define transforms for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to the range [-1, 1]
])

# Download and load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256*3, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256*3, shuffle=False)

import matplotlib.pyplot as plt
import numpy as np
# Get a batch of images from the data loader
batch_images, batch_labels = next(iter(train_loader))

# Convert the PyTorch tensor to numpy array for visualization
batch_images = batch_images.numpy()
batch_labels = batch_labels.numpy()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Now using device: ", device)

# Plot example images
plt.figure(figsize=(10, 5))
for i in range(10):  # Change the range as needed
    plt.subplot(2, 5, i+1)
    plt.imshow(batch_images[i].squeeze(), cmap='gray')
    plt.title(f"Number: {batch_labels[i]}")
    plt.axis('off')
plt.savefig('example_images.png')  # Save the plot as an image file


# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = CNN().to(device)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Function to train the models
def train(model, optimizer, criterion, train_loader):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Function to evaluate the models
def evaluate(model, criterion, test_loader):
    model.eval()
    correct = 0
    total = 0
    # with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

criterion = nn.CrossEntropyLoss()

# Training loop for CNN model
train_loss_history_cnn = []
test_accuracy_history_cnn = []

start_time = time.time()
for epoch in range(5):  # Adjust number of epochs as needed
    train_loss = train(cnn_model, cnn_optimizer, criterion, train_loader)
    test_accuracy = evaluate(cnn_model, criterion, test_loader)

    # Append loss and accuracy values to lists
    train_loss_history_cnn.append(train_loss)
    test_accuracy_history_cnn.append(test_accuracy)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")
end_time = time.time()
print(f"CNN Training Time: {end_time - start_time} seconds")

# Plot the training loss and testing accuracy
plt.figure(figsize=(10, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(range(1, 6), train_loss_history_cnn, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epoch')

# Plot testing accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, 6), test_accuracy_history_cnn, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Testing Accuracy')
plt.title('Testing Accuracy vs. Epoch')

plt.tight_layout()
plt.savefig('training_loss.png')  # Save the plot as an image file


import math

# Define a function to visualize the channels
def visualize_channels(conv_layer):
    # Get the weights of the convolutional layer
    weights = conv_layer.weight.data.cpu().numpy()

    # Normalize the weights to [0, 1]
    weights -= weights.min()
    weights /= weights.max()

    # Plot each channel
    num_channels = weights.shape[0]
    num_cols = 4
    num_rows = math.ceil(num_channels / num_cols)
    plt.figure(figsize=(num_cols * 2, num_rows * 2))
    for i in range(num_channels):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(weights[i, 0], cmap='gray')
        plt.axis('off')
        plt.title(f'Channel {i + 1}')
    plt.savefig('channels.png')  # Save the plot as an image file

# Create an instance of the CNN model
cnn_model = CNN()

# Visualize the channels of the first convolutional layer
visualize_channels(cnn_model.conv1)
# visualize_channels(cnn_model.conv2)