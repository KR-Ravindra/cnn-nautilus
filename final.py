import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# Define the generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device: ", device)

# Initialize the generator and discriminator
latent_dim = 100
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Define the training parameters
batch_size = 128
lr = 0.0002
beta1 = 0.5
epochs = 20

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Define loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training loop
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        real_images, _ = data
        real_images = real_images.to(device)  # Move images to the device
        batch_size = real_images.size(0)

        # Train the discriminator
        discriminator.zero_grad()
        real_label = torch.full((batch_size,), 1, dtype=torch.float32, device=device)  # Move labels to the device
        fake_label = torch.full((batch_size,), 0, dtype=torch.float32, device=device)  # Move labels to the device


        # Train with real images
        output = discriminator(real_images)
        loss_real = criterion(output.view(-1), real_label)
        loss_real.backward()
        D_x = output.mean().item()

        # Train with fake images
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)  # Generate noise on the device
        fake_images = generator(noise)
        output = discriminator(fake_images.detach())
        loss_fake = criterion(output.view(-1), fake_label)
        loss_fake.backward()
        D_G_z1 = output.mean().item()

        # Update discriminator
        optimizer_D.step()

        # Train the generator
        generator.zero_grad()
        output = discriminator(fake_images)
        loss_G = criterion(output.view(-1), real_label)
        loss_G.backward()
        D_G_z2 = output.mean().item()

        # Update generator
        optimizer_G.step()

        # Print training stats
        if i % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], Batch [{i}/{len(dataloader)}], '
                  f'D Loss: {loss_real.item() + loss_fake.item():.4f}, '
                  f'G Loss: {loss_G.item():.4f}, '
                  f'D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

    # Save generated images
    torchvision.utils.save_image(fake_images.detach(), f'fake_samples_epoch_{epoch}.png', normalize=True)

print('Training finished')


import torch
import torchvision.utils
import matplotlib.pyplot as plt

# Load the trained generator
latent_dim = 100
generator = Generator(latent_dim).to(device)
generator.load_state_dict(torch.load('generator.pth'))  # Load the trained generator state dict

# Generate images
num_images = 10
noise = torch.randn(num_images, latent_dim, 1, 1, device=device)  # Generate noise on the device
generated_images = generator(noise)

# Denormalize and display the generated images
plt.figure(figsize=(10, 1))
for i in range(num_images):
    plt.subplot(1, num_images, i+1)
    plt.imshow(generated_images[i].squeeze().detach().cpu().numpy(), cmap='gray')
    plt.axis('off')
plt.savefig('output.png')