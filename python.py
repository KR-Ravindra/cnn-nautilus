# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time

# Load MNIST dataset
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Now using device: ", device)

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # Increase the number of neurons
        self.fc2 = nn.Linear(512, 256)  # Increase the number of neurons
        self.fc3 = nn.Linear(256, 128)  # Increase the number of neurons
        self.fc4 = nn.Linear(128, 64)  # Add more layers
        self.fc5 = nn.Linear(64, 10)  # Add more layers

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))  # Add more layers
        x = self.fc5(x)  # Add more layers
        return x

# Define a more complex CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)  # Increase the number of filters
        self.conv2 = nn.Conv2d(64, 128, 3)  # Add more convolutional layers
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 5 * 5, 512)  # Increase the number of neurons and adjust the input size
        self.fc2 = nn.Linear(512, 256)  # Increase the number of neurons
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))  # Add more convolutional layers
        x = x.view(-1, 128 * 5 * 5)  # Adjust the input size
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

mlp_model = MLP().to(device)
cnn_model = CNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_mlp = optim.SGD(MLP().parameters(), lr=0.001, momentum=0.9)
optimizer_cnn = optim.SGD(CNN().parameters(), lr=0.001, momentum=0.9)

# Train MLP and CNN models
def train_model(model, optimizer):
    model = model.to(device)
    start_time = time.time()
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
    end_time = time.time()
    print('Finished Training')
    print('Training time: %.3f seconds' % (end_time - start_time))

# Train MLP
print("Training MLP...")
train_model(mlp_model, optimizer_mlp)

# Train CNN
print("Training CNN...")
train_model(cnn_model, optimizer_cnn)